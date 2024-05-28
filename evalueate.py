import cv2
import numpy as np
import torch
import yaml

from env.ImgEnv import ImgEnv
from model.QNet import Qnet
from utils import rl_utils


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def evalueate():
    config = load_config("./config/config.yaml")
    pth_path = config['evalueate']["pth_path"]
    test_data_dir_path = config['evalueate']["test_data_dir_path"]
    test_data_path = f'{test_data_dir_path}/test3.jpg'
    hidden_dim = config['model']["hidden_dim"]
    action_space_range_left = config['train']['action_space_range_left']
    action_space_range_right = config['train']['action_space_range_right']
    action_space_partition_num = config['train']['action_space_partition_num']
    action_list = np.linspace(action_space_range_left, action_space_range_right, num=action_space_partition_num, endpoint=True)[1:]
    action_dim = len(action_list)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Qnet(hidden_dim, action_dim).to(device)
    net.load_state_dict(torch.load(pth_path, map_location=device))
    net.eval()

    env = ImgEnv(test_data_path,action_list)

    for i in range(10):
        img_tensor = rl_utils.mat2tensor(env.imgState)
        img_tensor = img_tensor.to(device)
        out = net(img_tensor)
        value_max, indice = torch.max(out, dim=1)
        action_item = indice.item()
        action = action_list[action_item]
        next_state, reward, done = env.step(action_item)
        print(f'value_max: {value_max.item()}, action: {action}, reward: {reward}')


        cv2.imshow('next_state', next_state)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    evalueate()



