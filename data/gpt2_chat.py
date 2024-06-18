import torch
from torch import nn

class TestModel(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(TestModel,self).__init__()
        self.encode_module=nn.Sequential(nn.Linear(input_size, latent_dim),
                                    nn.LayerNorm(latent_dim),
                                    nn.ReLU(),
                                    )
    def forward(self, input):
        return self.encode_module(input)

if __name__ == '__main__':

    inputs=torch.ones([6,128])
    model= TestModel(128, 64)
    print(model(inputs).shape)