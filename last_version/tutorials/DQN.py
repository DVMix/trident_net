import torch

class DQN(torch.nn.Module):
    def __init__(self, observation_space, action_space, base=True):
        super(DQN, self).__init__()
        self.base = base
        hidden_layer_shape = 24
        self.dense1 = torch.nn.Linear(in_features=observation_space,  out_features=hidden_layer_shape)
        self.dense2 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=hidden_layer_shape)
        self.dense3 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=hidden_layer_shape)
        self.dense4 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=action_space)
        self.relu   = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(1)
    
    def forward(self, x):
        if self.base:
            x = self.relu(self.dense1(x))
            x = self.relu(self.dense2(x))
            # x = self.relu(self.dense3(x))# self.bn()
            x = self.dense4(x)
        else:
            x = x.unsqueeze(1)
            x = self.bn(self.relu(self.dense1(x)))
            x = self.bn(self.relu(self.dense2(x)))
            # x = self.relu(self.dense3(x))# self.bn()
            x = self.dense4(x)
            x = x.squeeze(1)
        return x