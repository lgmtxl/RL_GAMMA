import logging
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

from algorithms.DQN import DQN
from env.ImgEnv import ImgEnv
from model.QNet import Qnet
from utils import rl_utils as rl_utils
import matplotlib.pyplot as plt
import os
from utils.ReplayBuffer import ReplayBuffer

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

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
    exp_results_path = config['train']['exp_results_path']
    exp_log_path = config['train']['exp_log_path']
    exp_imgs = config['train']['exp_imgs']
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
    assert minimal_size>=batch_size, "minimal_size must be greater than batch_size "
    # 获取当前时间
    now = datetime.now()
    # 将当前时间转换为字符串，格式为 YYYY-MM-DD HH:MM:SS
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    #设置logger
    if not os.path.exists(exp_log_path+time_str+"/"):
        os.makedirs(exp_log_path+time_str+"/")
    exp_logFile_path = f'{exp_log_path}{time_str}/exp.log'
    train_logger = get_logger(exp_logFile_path, 1, 'train')
    #设置SummaryWriter
    if not os.path.exists(exp_results_path+time_str+"/"):
        os.makedirs(exp_results_path+time_str+"/")
    summary_writer_path = f'{exp_results_path}{time_str}/exp_run_log'
    summary_writer = SummaryWriter(summary_writer_path)
    tags = ['reward_time_steps', 'reward_episode', 'loss_time_steps', 'loss_episode']


    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_logger.info(f'device: {device}')

    action_list = np.linspace(action_space_range_left, action_space_range_right, num=action_space_partition_num, endpoint=True)[1:]
    train_logger.info(f'action_list: {action_list}')
    action_dim = len(action_list)
    if env_name == 'ImgEnv':
        env = ImgEnv(dataPath, action_list)

    if model_name == 'QNet':
        nets = [Qnet(hidden_dim,action_dim).to(device),Qnet(hidden_dim,action_dim).to(device)]

    #tensorboard中添加网络结构图
    init_img = torch.zeros((1,3, 1706, 1280) ,device=device)
    summary_writer.add_graph(nets[0], init_img)

    if algorithm_name == 'DQN':
        agent = DQN(action_dim, lr, gamma, epsilon, target_update, device, nets)
    if buffer_name == 'ReplayBuffer':
        replay_buffer = ReplayBuffer(buffer_size)


    for i in range(partition):
        train_logger.info(f'{"="*20}Iteration {i} {"="*20}')
        with tqdm(total=int(num_episodes / partition), desc='Iteration %d' % i) as pbar:
            if(i >= 2):
                agent.setEpsilon(0.45)
            if(i >= 4):
                agent.setEpsilon(0.3)
            if(i >= 7):
                agent.setEpsilon(0.1)
            for i_episode in range(int(num_episodes / partition)):
                train_logger.info(f'{"=" * 10}i_episode {i_episode} {"=" * 10}')
                episode_return = 0
                dqn_loss_sum = 0
                state = env.reset()
                done = False
                step_num = 0
                while not done and step_num < num_steps:
                    state_tensor = rl_utils.mat2tensor(state)
                    action = agent.take_action(state_tensor)
                    next_state, reward, done = env.step(action)
                    dir_path = f'{exp_imgs}/{time_str}/iter_{i}/episode{i_episode}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    if step_num==0 or (step_num + 1)%5==0:
                        cv2.imwrite(f'{dir_path}img_step{step_num}.jpg', next_state)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    summary_writer.add_scalar(tags[0], reward, (i*(num_episodes / partition)+i_episode)*num_steps+step_num)
                    train_logger.info(f'reward: {reward},   episode_return: {episode_return}    action_item: {action}   action: {action_list[action]}')

                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() >= minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        dqn_loss = agent.update(transition_dict)
                        dqn_loss_sum += dqn_loss
                        summary_writer.add_scalar(tags[2], dqn_loss, (i*(num_episodes / partition)+i_episode)*num_steps+step_num)
                    step_num+=1
                return_list.append(episode_return)
                summary_writer.add_scalar(tags[1],episode_return,i*(num_episodes / partition)+i_episode)
                summary_writer.add_scalar(tags[3],dqn_loss_sum/(step_num+1),i*(num_episodes / partition)+i_episode)
                # summary_writer.add_histogram(tag="last_fc",
                #                              values=agent.q_net.fc1.weight,
                #                              global_step=i*(num_episodes / partition)+i_episode)
                if (i_episode + 1) % 5 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / partition * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-5:])
                    })
                pbar.update(1)

    dir_path = f'{exp_results_path}{time_str}/model_pth/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(agent.q_net.state_dict(), f'{dir_path}q_net.pth')
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on test1.jpg')
    dir_path = f'{exp_results_path}{time_str}/rewardImg/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f'{dir_path}exp1.jpg')
    
    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on test1.jpg')
    plt.savefig(f'{dir_path}exp2.jpg')
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    config = load_config('./config/config.yaml')
    train(config)