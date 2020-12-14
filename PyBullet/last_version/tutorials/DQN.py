import torch

class DQN(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        
        hidden_layer_shape = 24
        self.dense1 = torch.nn.Linear(in_features=observation_space,  out_features=hidden_layer_shape)
        self.dense2 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=hidden_layer_shape)
        self.dense3 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=action_space)
        self.relu1   = torch.nn.ReLU()
        self.relu2   = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.dense3(x)
        return x