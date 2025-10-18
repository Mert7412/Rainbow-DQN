from torch.nn import Conv2d,ReLU,Flatten,Linear
from torch import nn
import torch

class DQN(nn.Module):
    def __init__(self,in_channel,action_space):
        super().__init__()
        self.cnn_layers = nn.Sequential(Conv2d(in_channel,32,kernel_size=8,stride=4),
                                 ReLU(),
                                 Conv2d(32,64,kernel_size=4,stride=2),
                                 ReLU(),
                                 Conv2d(64,64,kernel_size=3),
                                 ReLU())

        self.flatten = Flatten()

        with torch.no_grad():
            dummy_input_shape = self.cnn_layers(torch.zeros(1,in_channel,84,84)).shape
            input_size = dummy_input_shape[1]*dummy_input_shape[2]*dummy_input_shape[3]

        self.linear_layers = nn.Sequential(Linear(input_size,256),
                                           ReLU(),
                                           Linear(256,256),
                                           ReLU(),
                                           Linear(256,action_space))

    def forward(self,x):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x
    
