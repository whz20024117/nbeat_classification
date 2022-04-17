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

def predict(model:MyNBeatsModel, x):
    model.eval()

    x = torch.from_numpy(x).to(DEVICE)
    logit = model(x)

    logit_stack = []
    for batch_logit in logit:
        b_numpy = batch_logit.detach().cpu().numpy()
        logit_stack.append(b_numpy)

    return logit_stack


 # Predict
# result = predict(model, test_x[0:1])
# print(result)