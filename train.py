from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
import random

from algorithms.DQN import DQN
from env.ImgEnv import ImgEnv
from utils import rl_utils as rl_utils
import matplotlib.pyplot as plt
import os
from utils.ReplayBuffer import ReplayBuffer

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(config):
    #train parameters
    return_list = []
    lr = float(config['train']['lr'])
    num_steps = config['train']['num_steps']
    num_episodes = config['train']['num_episodes']
    partition = config['train']['partition']
    gamma = config['train']['gamma']
    epsilon = config['train']['epsilon']
    target_update = config['train']['target_update']
    batch_size = config['train']['batch_size']
    action_space_range_left = config['train']['action_space_range_left']
    action_space_range_right = config['train']['action_space_range_right']
    action_space_partition_num = config['train']['action_space_partition_num']
    #data parameters
    dataSetDir = config['DataSet']['DataSetDir']
    dataPath = config['DataSet']['DataPath']
    #env parameters
    env_name = config['env']['name']
    #algorithm parameters
    algorithm_name = config['algorithm']['name']
    #model parameters
    model_name = config['model']['name']
    hidden_dim = config['model']['hidden_dim']
    #buffer_size parameters
    buffer_name = config['ReplayBuffer']['name']
    buffer_size = config['ReplayBuffer']['buffer_size']
    minimal_size = config['ReplayBuffer']['minimal_size']


    # 设置设备
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    action_list = np.linspace(action_space_range_left, action_space_range_right, num=action_space_partition_num, endpoint=True)[1:]
    print(action_list)
    action_dim = len(action_list)
    if env_name == 'ImgEnv':
        env = ImgEnv(dataPath, action_list)

    if algorithm_name == 'DQN':
        agent = DQN(hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    if buffer_name == 'ReplayBuffer':
        replay_buffer = ReplayBuffer(buffer_size)

    # 获取当前时间
    now = datetime.now()
    # 将当前时间转换为字符串，格式为 YYYY-MM-DD HH:MM:SS
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    for i in range(partition):
        with tqdm(total=int(num_episodes / partition), desc='Iteration %d' % i) as pbar:
            if(i >= 3):
                agent.setEpsilon(0.45)
            if(i >= 6):
                agent.setEpsilon(0.3)
            if(i >= 8):
                agent.setEpsilon(0.1)
            for i_episode in range(int(num_episodes / 10)):
                # print(f'i_episode: {i_episode}')
                episode_return = 0
                state = env.reset()
                done = False
                step_num = 0
                while step_num < num_steps:
                    # print(f'step_num: {step_num}')
                    state_tensor = rl_utils.mat2tensor(state)
                    action = agent.take_action(state_tensor)
                    next_state, reward, done = env.step(action)
                    # if((step_num+1) % 2 == 0):
                    dir_path = f'./exp_imgs/{time_str}/iter_{i}/episode{i_episode}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    cv2.imwrite(f'{dir_path}img_step{step_num}.jpg',next_state)
                    # cv2.waitKey(0)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    print(f'reward: {reward},   episode_return: {episode_return}    action_item: {action}   action: {action_list[action]}')
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                        # print(b_r)
                    step_num+=1
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / partition * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
                
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on test1.jpg')
    plt.show()
    plt.savefig('exp1.jpg')
    
    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on test.jpg')
    plt.show()
    plt.savefig('exp2.jpg')
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    config = load_config('./config/config.yaml')
    train(config)