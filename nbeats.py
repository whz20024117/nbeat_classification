from typing import List
from darts.models.forecasting.nbeats import _GType, _Stack
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F

class MyNBeatsModel(nn.Module):
    def __init__(self,
        input_chunk_length: int,
        output_chunk_length: int, # Length of output of original Nbeats model to set up stacks
        input_dim: int,
        nr_params: int,
        generic_architecture: bool = True,
        num_stacks: int = 30,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths = 256,
        expansion_coefficient_dim: int = 5,
        trend_polynomial_degree: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.nr_params = nr_params
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length_multi = self.input_chunk_length * input_dim
        self.target_length = self.output_chunk_length * input_dim
        
        layer_widths = [layer_widths] * num_stacks

        if generic_architecture:
            self.stacks_list = [
                _Stack(
                    num_blocks,
                    num_layers,
                    layer_widths[i],
                    nr_params,
                    expansion_coefficient_dim,
                    self.input_chunk_length_multi,
                    self.target_length,
                    _GType.GENERIC,
                )
                for i in range(num_stacks)
            ]
        else:
            num_stacks = 2
            trend_stack = _Stack(
                num_blocks,
                num_layers,
                layer_widths[0],
                nr_params,
                trend_polynomial_degree + 1,
                self.input_chunk_length_multi,
                self.target_length,
                _GType.TREND,
            )
            seasonality_stack = _Stack(
                num_blocks,
                num_layers,
                layer_widths[1],
                nr_params,
                -1,
                self.input_chunk_length_multi,
                self.target_length,
                _GType.SEASONALITY,
            )
            self.stacks_list = [trend_stack, seasonality_stack]

        self.stacks = nn.ModuleList(self.stacks_list)
        self.output = nn.Linear(self.target_length * self.nr_params, 3) # dec, slight dec/slight inc, inc

        # setting the last backcast "branch" to be not trainable (without next block/stack, it doesn't need to be
        # backpropagated). Removing this lines would cause logtensorboard to crash, since no gradient is stored
        # on this params (the last block backcast is not part of the final output of the net).
        self.stacks_list[-1].blocks[-1].backcast_linear_layer.requires_grad_(False)
        self.stacks_list[-1].blocks[-1].backcast_g.requires_grad_(False)


    def forward(self, x):

        # if x1, x2,... y1, y2... is one multivariate ts containing x and y, and a1, a2... one covariate ts
        # we reshape into x1, y1, a1, x2, y2, a2... etc
        x = torch.reshape(x, (x.shape[0], self.input_chunk_length_multi, 1))
        # squeeze last dimension (because model is univariate)
        x = x.squeeze(dim=2)

        # One vector of length target_length per parameter in the distribution
        y = torch.zeros(
            x.shape[0],
            self.target_length,
            self.nr_params,
            device=x.device,
            dtype=x.dtype,
        )

        for stack in self.stacks_list:
            # compute stack output
            stack_residual, stack_forecast = stack(x)

            # add stack forecast to final output
            y = y + stack_forecast

            # set current stack residual as input for next stack
            x = stack_residual

        y = y.view(y.shape[0], -1)
        y = self.output(y)
        y = F.softmax(y)

        return y

def generate_batch(x, y, bs):
    if (len(x) != len(y)):
        raise ValueError("X and Y should contain same number of samples")
    
    batches = []
    for i in range(0, len(x), bs):
        batches.append((x[i:i + bs], y[i:i + bs]))
    
    for batch in batches:
        yield batch

def create_data_and_label(data, input_step, avg_n):
    """
    Prepare the data for the model.
    Parameters:
        data: original sequence data in np.ndarray. First var need to be the target var.
        input_step: number of steps in input (input_chunk_length)
        avg_n: number of samples to average for comparison
    """
    # Data shape: [Timestep, variable]
    if len(data.shape) > 2:
        raise ValueError("Original data should have shape as [Timestep, variable]")
    
    x = []
    y = []
    for i in range(input_step, len(data) - avg_n):
        # Data
        _x = data[i - input_step:i]
        x.append(_x)

        # Label
        _data_to_avg = data[i:i + avg_n]
        _data_avg = np.mean(_data_to_avg[:, 0]) # We assume first var is Close

        _diff = _data_avg - _x[-1][0]
        _percentage_diff = _diff / (_x[-1][0] + 1e-6)  # We assume first var is Close. Avoid zero division

        if _percentage_diff <= -0.05:
            y.append(0)
        elif _percentage_diff < 0.05 and _percentage_diff > -0.05:
            y.append(1)
        else:
            y.append(2)

    return np.array(x), np.array(y)

