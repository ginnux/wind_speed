import torch
import torch.nn as nn


# Batch_first = True,
# input (Length, Num, inputsize)
# output (Length, Num, Hiddensize)


class MinimalRNNCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        batch_first=True,
        dropout=0.0,
        device="cpu",
    ):
        super(MinimalRNNCell, self).__init__()
        latent_size = hidden_size

        # 定义输入层的参数
        self.input_layer = nn.Linear(input_size, latent_size, device=device)

        # 定义门的参数
        self.hidden_size = hidden_size
        # self.U_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # self.U_z = nn.Parameter(torch.randn(input_size, hidden_size))
        # self.b_u = nn.Parameter(torch.zeros(hidden_size))
        self.gate_layer = nn.Linear(
            latent_size + hidden_size, hidden_size, device=device
        )

        # 定义输出层的参数
        self.output_layer = nn.Linear(hidden_size, output_size, device=device)

        # 定义是否为batch_first
        self.batch_first = batch_first

    def forward(self, inputs, h=None):

        isBatch = True

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            self.batch_first = True
            isBatch = False

        outputs = []
        # RNN的前向传播
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        # input (Num, Length, input_size)

        length = inputs.size(0)
        batch_size = inputs.size(1)
        input_size = inputs.size(2)
        latent_size = self.hidden_size
        output_size = input_size

        if h is None:
            h = torch.zeros(
                batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device
            )
            # (batch_size, hidden_size)

        assert h.size() == torch.Size([batch_size, self.hidden_size]), (
            str(h.size()) + "," + str(batch_size) + "," + str(self.hidden_size)
        )

        for x in inputs:
            assert x.size() == torch.Size([batch_size, input_size])

            z_t = self.input_layer(x)
            assert z_t.size() == torch.Size([batch_size, latent_size])

            u_inputs = torch.cat([h, z_t], dim=1)
            u_t = torch.sigmoid(self.gate_layer(u_inputs))

            h = u_t * h + (1 - u_t) * z_t

            y = self.output_layer(h)

            outputs.append(y.unsqueeze(0))

        outs = torch.cat(outputs, dim=0)

        if isBatch is False:
            outs = outs.squeeze(1)
        else:
            if self.batch_first:
                outs = outs.transpose(0, 1)

        return outs[-1, :]


# 将MinimalRNNCell扩展为多层RNN
class MinimalRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        batch_first=True,
        dropout=0.0,
        device="cpu",
    ):
        super(MinimalRNN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            MinimalRNNCell(
                input_size, hidden_size, output_size, batch_first, dropout, device
            )
        )
        for i in range(1, num_layers):
            self.layers.append(
                MinimalRNNCell(
                    hidden_size, hidden_size, output_size, batch_first, dropout, device
                )
            )
        self.batch_first = batch_first

    def forward(self, inputs, h=None):
        isBatch = False if inputs.dim() == 2 else True

        if h is None:
            h = [None] * self.num_layers
        else:
            h = [h[i] for i in range(self.num_layers)]

        for i in range(self.num_layers):
            inputs, h[i] = self.layers[i](inputs, h[i])

        if isBatch is True:
            h = torch.stack(h, dim=0)
        else:
            h = torch.cat(h, dim=0)

        return inputs, h
