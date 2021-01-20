import torch

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.hw = 'Hello, World!'
        self.Linear_1 = torch.nn.Linear(26 ,  52)
        self.Linear_2 = torch.nn.Linear(52 , 104)
        self.Linear_3 = torch.nn.Linear(104, 208)
        self.Linear_4 = torch.nn.Linear(208, 104)
        self.Linear_5 = torch.nn.Linear(104,  52)
        self.classifier = torch.nn.Linear(52 ,26)
        self.regressor  = torch.nn.Linear(52 ,26)
        
    def forward(self, x):
        x = self.Linear_1(x)
        x = self.Linear_2(x)
        x = self.Linear_3(x)
        x = self.Linear_4(x)
        x = self.Linear_5(x)
        cls = self.classifier(x)
        rgr = self.regressor(x)
        return cls, rgr
    
def optimizer_init(net, lr  = 1e-3,  momentum = 0.9, 
                   wght_dcy = 5e-4,  epsilon  = 1e-8, beta1 = 0.9,  beta2 = 0.999, 
                   amsgrad  = False, is_SGD   = True, custom_params = True):
    if custom_params:
        biases, not_biases = list(), list()
        for param_name, param in net.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        params = [{'params': biases, 'lr': 2*lr}, {'params': not_biases}]
    else:
        params = net.parameters()
    if is_SGD:
        optimizer = torch.optim.SGD(
            params       = params,
            lr           = lr,
            momentum     = momentum,
            weight_decay = wght_dcy)
    else:
        optimizer = torch.optim.Adam(
            params       = params, 
            lr           = lr, 
            betas        = (beta1,beta2), 
            eps          = epsilon,
            weight_decay = wght_dcy, 
            amsgrad      = amsgrad)
    return optimizer