import torch

class MLP:
    def __init__(self,input_size,layer1_size,layer2_size):
        self.layer1 = torch.rand(input_size, layer1_size)
        self.layer2 = torch.rand(layer1_size, layer2_size)
