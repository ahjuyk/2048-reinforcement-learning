# dqn.py
# https://geektutu.com
from collections import deque
import random
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers


class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 200  # 模型更新频率
        self.replay_size = 2000  # 训练集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_conv_model()
        self.target_model = self.create_conv_model()

    def create_model(self):
        """创建一个隐藏层为100的神经网络"""
        board_size=4
        board_layers=16
        ACTION_DIM = 4
        model = models.Sequential([
            layers.Input(shape=(board_size * board_size * board_layers,)),
            layers.Dense(100, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def create_conv_model(self):
        print("Tensorflow version: {}".format(tf.__version__))
        print("Tensorflow keras version: {}".format(tf.keras.__version__))
        filters = 128
        filter2 = 256
        board_size=4
        board_layers=16
        ACTION_DIM = 4
        # Functional API model
        inputs = layers.Input(shape=(board_size * board_size * board_layers,))
        x = layers.Reshape((board_size, board_size, board_layers))(inputs)

        # Initial convolutional block
        conv_a = layers.Conv2D(filters=filters, kernel_size=(2, 1), padding='valid')(x)
        conv_a = layers.BatchNormalization()(conv_a)
        conv_aa = layers.Conv2D(filters=filter2, kernel_size=(2, 1), padding='valid')(conv_a)
        conv_aa = layers.BatchNormalization()(conv_aa)
        conv_ab = layers.Conv2D(filters=filter2, kernel_size=(1, 2), padding='valid')(conv_a)
        conv_ab = layers.BatchNormalization()(conv_ab)

        conv_b = layers.Conv2D(filters=filters, kernel_size=(1, 2), padding='same')(x)
        conv_b = layers.BatchNormalization()(conv_b)
        conv_ba = layers.Conv2D(filters=filter2, kernel_size=(2, 1), padding='same')(conv_b)
        conv_ba = layers.BatchNormalization()(conv_ba)
        conv_bb = layers.Conv2D(filters=filter2, kernel_size=(1, 2), padding='same')(conv_b)
        conv_bb = layers.BatchNormalization()(conv_bb)

        c_x = layers.concatenate([layers.Flatten()(x) for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]])
        d_x = layers.Dense(256, activation='relu')(c_x)
        d_x = layers.Dense(64, activation='relu')(d_x)
        predictions = layers.Dense(ACTION_DIM, activation='linear')(d_x)
        model = models.Model(inputs=inputs, outputs=predictions)
        model.summary()
        # Create model
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def act(self, s, epsilon=0.1):
        """预测动作"""
        # 刚开始时，加一点随机成分，产生更多的状态
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2, 3])
        return self.model.predict(np.reshape(s.astype('float32'), (-1, 256)))[0]

    def save_model(self, file_path='2048-v0-dqn.hdf5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward, obs, max_score = False):
        """當盤面最大值在角落时给额外的reward，快速收敛"""
        if obs[0][0] == np.max(obs) or obs[3][0] == np.max(obs) or obs[0][3] == np.max(obs) or obs[3][3] == np.max(obs):
            reward *= 1.2
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # 每 update_freq 步，将 model 的权重赋值给 target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([np.reshape(replay[0].astype('float32'), (-1, 256)) for replay in replay_batch])
        s_batch = np.reshape(s_batch, (64,256))
        next_s_batch = np.array([np.reshape(replay[2].astype('float32'), (-1, 256)) for replay in replay_batch])
        next_s_batch = np.reshape(next_s_batch, (64, 256))

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        # 传入网络进行训练
        self.model.fit(s_batch, Q, verbose=0)
    def set_model(self, model):
        self.model = model
