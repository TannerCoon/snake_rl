from re import L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, saved_model=None):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        if hidden_size2:
            self.linear2 = nn.Linear(hidden_size1, hidden_size2)
            self.linear3 = nn.Linear(hidden_size2, output_size)
        else:
            self.linear2 = nn.Linear(hidden_size1, output_size)
            self.linear3 = None
        if saved_model:
            self.model = self.load_state_dict(torch.load(saved_model))
            print("weights loaded")

    def forward(self, x):
        x = F.relu(self.linear1(x))
        if self.linear3:
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
        else:
            x = self.linear2(x)
            # x = F.normalize(self.linear2(x))
            # x = F.instance_norm(self.linear2(x))
        return x

    def save(self, dt_str, run_name, record):
        file_name = f'{run_name}__{record}.pth'
        model_folder_path = f'./model/{dt_str}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class CNNet(nn.Module):
    def __init__(self, input_channels=3, output_features=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=20, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        # self.conv2 = nn.Conv2d(in_channels=20, out_channels= 256, kernel_size=(2,2))
        # self.flat = nn.Flatten()
        self.lin1 = nn.Linear(720, 64)
        self.lin2 = nn.Linear(64, output_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.flat(x)
        x = x.view(-1,720)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def save(self, dt_str, run_name, record):
        file_name = f'{run_name}__{record}.pth'
        model_folder_path = f'./model_cnn/{dt_str}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        


    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        if type(state) == tuple:
            state = np.array(state)
            next_state = np.array(next_state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        print(len(state.shape))
        if len(state.shape) == 1 or len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        # 1: get predicted Q values with current state
        # if len(state.shape) == 3:
        #     pred = self.model(state)[0]
        # if len(state.shape) == 4:
        #     print(self.model(state))
        #     pred = [self.model(s)[0] for s in state]
        # else:
        pred = self.model(state)
        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        target = pred.clone()
        print('prev:', target)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        print('reward:', reward)
        print('new:', target)

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()