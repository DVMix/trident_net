import torch
import numpy as np
from .DQN import DQN
from collections import deque
import random

class DQNSolver:
    def __init__(self, observation_space, action_space, 
                 base              = True,
                 EXPLORATION_MIN   = 0.01,
                 EXPLORATION_MAX   = 1.0,
                 EXPLORATION_DECAY = 0.995,
                 GAMMA             = 0.95,
                 LEARNING_RATE     = 0.001,
                 MEMORY_SIZE       = 1000000,
                 BATCH_SIZE        = 150
                ):
        self.exploration_rate  = EXPLORATION_MAX
        self.EXPLORATION_DECAY = EXPLORATION_DECAY
        self.EXPLORATION_MIN   = EXPLORATION_MIN
        
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA      = GAMMA
        
        self.action_space = action_space
        self.MEMORY_SIZE = MEMORY_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_net = DQN(observation_space, action_space, base).to(self.device)
            
        self.target_net = DQN(observation_space, action_space, base).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
                
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.policy_net.parameters(), lr=LEARNING_RATE)
        
    def save(self, name_template):
        # fix format parameters
        torch.save(self.policy_net.state_dict(), name_template.format()) 
            
    def load(self, path2checkpoints):
        files = [int(file.split('_')[-1].replace('.pth','')) 
                 for file in os.listdir(path2checkpoints) if '.pth' in file]
        if files!= []:
            file = 'best_'+str(max(files))+'.pth'
            if file in os.listdir():
                self.policy_net.load_state_dict(torch.load(file))
                print('Weights loaded!')
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        
        q_values = self.policy_net(state.unsqueeze(0).to(self.device))
        return torch.argmax(q_values).item()
    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = random.sample(self.memory, self.BATCH_SIZE)
        non_final_mask = ~torch.stack([b[4] for b in batch])
        non_final_next_states = torch.stack([b[3] for b in batch if not b[4]])

        state_batch  = torch.stack([state[0].to(self.device)  for state  in batch])
        action_batch = torch.stack([action[1] for action in batch]).reshape(self.BATCH_SIZE,1)
        action_batch = action_batch.to(self.device)
        
        reward_batch = torch.stack([reward[2] for reward in batch]).to(self.device)
        state_action_values = self.policy_net(state_batch)
        
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device = self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.exploration_rate *= self.EXPLORATION_DECAY
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)