import torch
from torch import nn


class LoraLinear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(LoraLinear, self).__init__()

        row, column = weight.shape

        # restore Linear
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # create LoRA weights (with initialization)
        self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
        nn.init.kaiming_normal_(self.lora_right)  # , a=math.sqrt(5)
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, row))

    def forward(self, inputs):
        w_inputs, u_i_inputs = inputs
        x = self.linear(w_inputs)
        y = u_i_inputs @ self.lora_right @ self.lora_left
        y = torch.mean(y, dim=1).unsqueeze(1).repeat((1, x.shape[1], 1))

        return x + y
