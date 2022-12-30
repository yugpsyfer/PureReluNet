import torch.nn as nn
from collections import OrderedDict

"""
Linear model whihc only has relu as activation function.
inputSize: integer
outputSize: integer
hiddenLayers: list of int, where length of list number of layers and each integer is size of the layer
"""

class PureReluNet(nn.Module):

    def __init__(self, inputSize, hiddenLayers, outputSize) -> None:
        super().__init__()
        self.inputSize = inputSize
        
        if type(hiddenLayers) != list:
            print("Failed to build model, Check comments to understand the paramter's type")
            return
        self.hiddenLayers = hiddenLayers

        self.model = self.__make_model__()
    
    def __make_model__(self):
        model_dict= OrderedDict()
        model_dict['input'] = nn.Linear(in_features=self.inputSize, out_features=self.hiddenLayers[0])
        model_dict['relu'] = nn.ReLU()

        previous_output_size = self.hiddenLayers[0]

        for layer in range(len(self.hiddenLayers)-1):
            model_dict['layer_'+str(layer)] = nn.Linear(in_features=previous_output_size, out_features=self.hiddenLayers[layer+1])
            model_dict['relu_'+str(layer)] = nn.ReLU()
            previous_output_size = self.hiddenLayers[layer+1]
        
        
        return nn.Sequential(model_dict)
    
    def forward(self,x):
        return self.model(x)
