import torch

class DQN(torch.nn.Module):
    def __init__(self, observation_space, action_space, base=True):
        super(DQN, self).__init__()
        self.base = base
        self.multiplier = 2
        hidden_layer_shape = observation_space * self.multiplier
        self.dense1 = torch.nn.Linear(in_features=observation_space,  out_features=hidden_layer_shape)
        self.dense2 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=hidden_layer_shape)
        self.dense3 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=hidden_layer_shape)
        self.dense4 = torch.nn.Linear(in_features=hidden_layer_shape, out_features=action_space)
        self.bn1    = torch.nn.BatchNorm1d(hidden_layer_shape)
        self.bn2    = torch.nn.BatchNorm1d(hidden_layer_shape)
        self.bn3    = torch.nn.BatchNorm1d(hidden_layer_shape)
        self.relu   = torch.nn.ReLU()
        
    def forward(self, X):
        x = X.view(X.shape[0], -1)
        # =-=-=-=-=-=-=-=-=-=-
        x = self.dense1(x)
        try:
            x = self.bn1(x)
        except:
            pass
        x = self.relu(x)
        # =-=-=-=-=-=-=-=-=-=-
        x = self.dense2(x)
        try:
            x = self.bn2(x)
        except:
            pass
        x = self.relu(x)
        # =-=-=-=-=-=-=-=-=-=-
        x = self.dense3(x)
        try:
            x = self.bn3(x)
        except:
            pass
        x = self.relu(x)
        # =-=-=-=-=-=-=-=-=-=-
        x = self.dense4(x)
        x = self.relu(x)
            
        # x = self.relu(self.dense1(x))
        # x = self.relu(self.dense2(x))
        # x = self.relu(self.dense3(x))
        # x = self.dense4(x)
        return x