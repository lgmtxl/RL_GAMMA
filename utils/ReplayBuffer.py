import collections
import random
import numpy as np
import pickle
import os


class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class ReplayBuffer_Disk:
    def __init__(self, buffer_size, file_path='buffer_data'):
        self.buffer_size = buffer_size
        self.file_path = file_path
        self.buffer = collections.deque(maxlen=buffer_size)  # 队列,先进先出
        self.position = 0
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def push(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.buffer_size
        self.save_to_disk()

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def save_to_disk(self):
        with open(os.path.join(self.file_path, f'buffer_{self.position}.pkl'), 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_from_disk(self, position):
        with open(os.path.join(self.file_path, f'buffer_{position}.pkl'), 'rb') as f:
            self.buffer = pickle.load(f)

