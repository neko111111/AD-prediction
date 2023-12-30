import torch
import torch.nn as nn


# 定义脉冲神经元模型
class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = 0

    def forward(self, x):
        self.membrane_potential += x
        print(self.membrane_potential.shape)
        spike = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spike) * self.decay
        return spike


# 定义脉冲神经网络
class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpikingNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = SpikingNeuron()
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_spike):
        spike1 = self.input_layer(input_spike)
        spike2 = self.hidden_layer(spike1)
        output = self.output_layer(spike2)
        return output
