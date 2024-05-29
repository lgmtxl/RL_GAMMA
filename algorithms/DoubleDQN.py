import numpy as np
import torch
import torch.nn.functional as F
from utils import rl_utils
from torch.cuda.amp import GradScaler, autocast



class DoubleDQN:
    ''' DoubleDQN算法 解决过高估计 '''

    def __init__(self,  action_dim, learning_rate, gamma,
                 epsilon, target_update, device, nets):
        self.action_dim = action_dim
        self.q_net = nets[0].to(device)  # Q网络
        # 目标网络
        self.target_q_net = nets[1].to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.scaler = GradScaler()

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # print(self.epsilon)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # print(self.device)
            state = state.to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = rl_utils.mat2tensor(transition_dict['states']).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = rl_utils.mat2tensor(transition_dict['next_states']).to(self.device)
        # next_states = transition_dict['next_states'].to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        with autocast():
            q_values = self.q_net(states).gather(1, actions)  # Q值
            # 下个状态的最大Q值
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                    )  # TD误差目标
            dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()
        self.scaler.scale(dqn_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # print(f'loss: {dqn_loss}')
        # self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        # dqn_loss.backward()  # 反向传播更新参数
        # self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
        loss_item = dqn_loss.item()
        return loss_item, torch.mean(q_values).item(), torch.mean(q_targets).item()