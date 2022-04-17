from typing import Dict
from nbeats import MyNBeatsModel, generate_batch, create_data_and_label
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import sys
from torch import nn
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STOCK_SCALER = None
COV_SCALER = None
SAVEPATH = "model_save/save1.pth"

def timeseries_to_numpy(target_ts, cov_ts):
    # Shape: N, variable, feature
    target_numpy = target_ts.all_values()
    cov_numpy = cov_ts.all_values()

    return np.concatenate([target_numpy, cov_numpy], 1)
    
def preprocess_data(hourly_data):
    global STOCK_SCALER
    global COV_SCALER

    # Some dirty processing
    cov_cols = list(hourly_data['A'].columns)
    cov_cols.remove('Date')
    cov_cols.remove('Close')
    cov_cols.remove('first_month')
    cov_cols.remove('ticker')

    # Start
    stock_ts = {}
    train_stock_ts = {}
    test_stock_ts = {}
    stock_scaler = {}
    cov_ts = {}
    train_cov_ts = {}
    test_cov_ts = {}
    cov_scaler = {}
    for name, base_data in hourly_data.items():
        temp_stock_df = base_data['Close'].copy()
        if len(temp_stock_df) <= 200:
            continue
        temp_stock_ts = TimeSeries.from_series(temp_stock_df)
        for cov_name in cov_cols:
            temp_cov_df = base_data[f'{cov_name}'].copy()
            temp_cov_ts = TimeSeries.from_series(temp_cov_df)
            if cov_name == cov_cols[0]:
                stack_cov = temp_cov_ts
            else:
                stack_cov = stack_cov.stack(temp_cov_ts)
                
        temp_stock_scaler = Scaler()
        temp_stock_ts_scaled = temp_stock_scaler.fit_transform(temp_stock_ts)
        train_stock, val_stock = temp_stock_ts_scaled.split_before(len(temp_stock_ts_scaled)-50)
        STOCK_SCALER = temp_stock_scaler
        
        temp_cov_scaler = Scaler()
        stack_cov_scaled = temp_cov_scaler.fit_transform(stack_cov)
        train_cov, val_cov = stack_cov_scaled.split_before(len(stack_cov_scaled)-50)
        COV_SCALER = temp_cov_scaler
        
        # stock_ts[name] = temp_stock_ts_scaled
        train_stock_ts[name] = train_stock
        test_stock_ts[name] = val_stock
        # stock_scaler[name] = temp_cov_scaler
        
        # cov_ts[name] = stack_cov_scaled
        train_cov_ts[name] =train_cov
        test_cov_ts[name] = val_cov
        # cov_scaler[name] = temp_cov_scaler

    return train_stock_ts, test_stock_ts, train_cov_ts, test_cov_ts

def train(model:MyNBeatsModel, train_x, train_y, epoch, lr, batchsize):
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    n_sample = len(train_x)
    assert n_sample == len(train_y)

    for e in range(epoch):
        print("Epoch {}: ".format(e), file=sys.stderr)
        model.train()
        train_loss = []

        iteration = 0
        for batch_x, batch_y in generate_batch(train_x, train_y, batchsize):
            optimizer.zero_grad()
            batch_x = torch.from_numpy(batch_x)
            batch_x.to(DEVICE)
            batch_y = [torch.from_numpy(_y).to(DEVICE) for _y in batch_y]
            logit = model(batch_x)

            loss = model.get_loss(logit, batch_y)
            if iteration % 4 == 0:
                print(
                    "{}/{}: current loss is {:.6f}".format(batchsize*iteration, n_sample, loss.item()),
                    end="\r",
                    flush=True,
                    file=sys.stderr
                )
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            iteration += 1
        
        train_loss = np.mean(train_loss)
        print("\nEnd of Epoch {}, average train loss: {:.6f} ".format(e, train_loss), file=sys.stderr)

def test(model:MyNBeatsModel, test_x, test_y):
    model.eval()
    bs = 50
    
    overall_acc = []
    
    for batch_x, batch_y in generate_batch(test_x, test_y, bs):
        batch_x = torch.from_numpy(batch_x)
        batch_x.to(DEVICE)
        logit = model(batch_x)

        logit_stack = []
        for batch_logit in logit:
            b_numpy = batch_logit.detach().cpu().numpy()
            logit_stack.append(b_numpy)
        
        acc_nday = []
        for ls, by in zip(logit_stack, batch_y):
            pred = np.argmax(ls, axis=-1)
            acc = accuracy_score(by, pred)
            acc_nday.append(acc)
        
        overall_acc.append(acc_nday)

    overall_acc = np.array(overall_acc)
    acc_avg = np.mean(overall_acc, axis=0)

    return acc_avg

def predict(model:MyNBeatsModel, x):
    model.eval()

    x = torch.from_numpy(x).to(DEVICE)
    logit = model(x)

    logit_stack = []
    for batch_logit in logit:
        b_numpy = batch_logit.detach().cpu().numpy()
        logit_stack.append(b_numpy)

    return logit_stack

def main():
    # Load data
    hourly_data = {}
    tickers = [n for n, ext in map(os.path.splitext, os.listdir('darts_data')) if ext == '.csv']

    for i in range(0,len(tickers)):
        ticker = tickers[i]
        stock_data = pd.read_csv(f'darts_data/{ticker}.csv', index_col = 0, parse_dates=True)
        stock_data = stock_data.dropna()
        stock_data.sort_values('Date', axis = 0, inplace = True)
        stock_data.index = np.arange(0, len(stock_data))
        hourly_data[ticker] = stock_data

    for name, base_data in hourly_data.items():
        base_data.set_index(pd.RangeIndex.from_range(range(len(base_data))), inplace=True)

    # Process the raw data to dart time series
    train_stock_ts, test_stock_ts, train_cov_ts, test_cov_ts = preprocess_data(hourly_data)

    # Dart ts to numpy
    train_stock_numpys = []
    for com in train_stock_ts.keys():
        train_stock_numpys.append( timeseries_to_numpy(train_stock_ts[com], train_cov_ts[com]) )

    test_stock_numpys = []
    for com in train_stock_ts.keys():
        test_stock_numpys.append( timeseries_to_numpy(test_stock_ts[com], test_cov_ts[com]) )

    # Set Hyper-parameters
    input_step = 30
    nday = 5
    input_dim = train_stock_numpys[0].shape[1]

    # Create training data that ready to feed into model
    train_x = []
    train_y = []
    for com_seq in train_stock_numpys:
        _x, _y = create_data_and_label(com_seq.squeeze(-1), input_step=input_step, nday=nday)
        train_x.append(_x)
        train_y.append(_y)

    train_x = np.concatenate(train_x, 0).astype(np.float32)
    train_y = np.concatenate(train_y, 0).astype(int)

    # Create testing data that ready to feed into model
    test_x = []
    test_y = []
    for com_seq in test_stock_numpys:
        _x, _y = create_data_and_label(com_seq.squeeze(-1), input_step=input_step, nday=nday)
        test_x.append(_x)
        test_y.append(_y)

    test_x = np.concatenate(test_x, 0, dtype=np.float32)
    test_y = np.concatenate(test_y, 0, dtype=int)

    assert len(train_x) == len(train_y)
    assert len(test_x) == len(test_y)
    print("Number of training samples: {}".format(len(train_x)), file=sys.stderr)
    print("Number of testing samples: {}".format(len(test_x)), file=sys.stderr)
    
    model = MyNBeatsModel(
        input_chunk_length=input_step,
        output_chunk_length=nday,
        input_dim=input_dim,
        nr_params=1,
        generic_architecture=True,
        num_stacks=10,
        num_blocks=3,
        num_layers=4,
        layer_widths=512
    )
    model.to(DEVICE)

    # Load
    # print("Loading model from {}".format(SAVEPATH), file=sys.stderr)
    # model = torch.load(SAVEPATH)
    # model.to(DEVICE)

    # Train
    train(model, train_x, train_y, epoch=10, lr=1e-4, batchsize=32) # Hypter parameter

    # Save
    print("Saving model to {}".format(SAVEPATH), file=sys.stderr)
    torch.save(model, SAVEPATH)

    # Test
    acc_nday = test(model, test_x, test_y)
    print(acc_nday)


if __name__ == "__main__":
    main()