from typing import Dict
from nbeats import MyNBeatsModel, generate_batch, create_data_and_label
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import pandas as pd
import numpy as np
import torch
import sys
from torch import nn
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def timeseries_to_numpy(target_ts, cov_ts):
    # Shape: N, variable, feature
    target_numpy = target_ts.all_values()
    cov_numpy = cov_ts.all_values()

    return np.concatenate([target_numpy, cov_numpy], 1)
    
def preprocess_data(hourly_data):
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
        if len(temp_stock_df) == 0:
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
        
        temp_cov_scaler = Scaler()
        stack_cov_scaled = temp_cov_scaler.fit_transform(stack_cov)
        train_cov, val_cov = stack_cov_scaled.split_before(len(stack_cov_scaled)-50)
        
        # stock_ts[name] = temp_stock_ts_scaled
        train_stock_ts[name] = train_stock
        test_stock_ts[name] = val_stock
        # stock_scaler[name] = temp_cov_scaler
        
        # cov_ts[name] = stack_cov_scaled
        train_cov_ts[name] =train_cov
        test_cov_ts[name] = val_cov
        # cov_scaler[name] = temp_cov_scaler

    return train_stock_ts, test_stock_ts, train_cov_ts, test_cov_ts

def train(model:nn.Module, train_x, train_y, epoch, lr, batchsize):
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
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
            batch_y = torch.from_numpy(batch_y)
            batch_y.to(DEVICE)
            logit = model(batch_x)

            loss = loss_fn(logit.softmax(dim=-1), batch_y)
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

    # test_stock_numpys = []
    # for com in train_stock_ts.keys():
    #     test_stock_numpys.append( timeseries_to_numpy(test_stock_ts[com], test_cov_ts[com]) )

    # Set Hyper-parameters
    input_step = 30
    avg_n = 7
    input_dim = train_stock_numpys[0].shape[1]

    # Create training data that ready to feed into model
    train_x = []
    train_y = []
    for com_seq in train_stock_numpys:
        _x, _y = create_data_and_label(com_seq.squeeze(-1), input_step=input_step, avg_n=avg_n)
        train_x.append(_x)
        train_y.append(_y)

    train_x = np.concatenate(train_x, 0, dtype=np.float32)
    train_y = np.concatenate(train_y, 0, dtype=int)

    assert len(train_x) == len(train_y)
    print("Number of training samples: {}".format(len(train_x)), file=sys.stderr)
    
    model = MyNBeatsModel(
        input_chunk_length=input_step,
        output_chunk_length=avg_n,
        input_dim=input_dim,
        nr_params=1,
        generic_architecture=True,
        num_stacks=10,
        num_blocks=3,
        num_layers=4,
        layer_widths=512
    )
    model.to(DEVICE)

    # Train
    train(model, train_x, train_y, epoch=10, lr=1e-4, batchsize=32)


if __name__ == "__main__":
    main()