import cv2
import numpy as np
import utils.util as util

class ImgEnv:
    def __init__(self, imgPath: str, action_list: np.array) -> None:
        self.ori_img = cv2.imread(imgPath)
        self.imgState = cv2.imread(imgPath)
        self.action_list = action_list

    def step(self, action):
        reward = self.compute_reward()
        done = False
        if(reward <= 5):
            reward = -20
            done = True
        reward = util.scale_to_0_1(reward,max_val=20,min_val=-20)
        self.imgState = self.adjust_gamma(self.action_list[action])
        return self.imgState, reward, done

    def reset(self):
        self.imgState = self.ori_img
        return self.imgState

    def adjust_gamma(self, gamma=1.0):
        # 建立映射表
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        # 应用 gamma 校正
        return cv2.LUT(self.imgState, table)

    def compute_reward(self):
        hist = self.compute_hist()
        reward_entropy = self.loss_entropy(hist)
        reward_contrast = self.loss_contrast(hist)
        reward = reward_entropy + reward_contrast / 10
        return reward

    def compute_hist(self):
        image = cv2.cvtColor(self.imgState, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return hist

    def loss_contrast(self, hist):
        def calculate_contrast(hist):
            # 计算直方图的标准差作为对比度的一种度量
            hist = hist.astype(np.float32)  # 确保使用浮点数
            mean = np.sum(hist.reshape(256) * np.arange(256)) / np.sum(hist)
            variance = np.sum(((np.arange(256) - mean) ** 2) * hist.reshape(256)) / np.sum(hist)
            return np.sqrt(variance)

        # 计算对比度
        contrast = calculate_contrast(hist)
        return contrast

    def loss_entropy(self, hist):
        hist_norm_ori = hist.ravel() / hist.sum()
        entropy = -np.sum(hist_norm_ori * np.log2(hist_norm_ori + np.finfo(float).eps))
        return entropy