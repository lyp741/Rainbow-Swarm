# -*- coding: utf-8 -*-

import chainer
import copy
import numpy as np
from chainer import cuda, Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import pickle as pickle


class QNet:
    # Hyper-Parameters
    gamma = 0.95  # Discount factor
    timestep_per_episode = 5000
    initial_exploration = timestep_per_episode * 1  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    hist_size = 2  # original: 4
    data_index = 0
    data_flag = False
    loss_log = '../playground/Assets/log/'

    def __init__(self, use_gpu, enable_controller, cnn_input_dim,
                 feature_dim, agent_count, other_input_dim, model):
        self.use_gpu = use_gpu
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller
        self.cnn_input_dim = cnn_input_dim
        self.feature_dim = feature_dim
        self.agent_count = agent_count
        self.other_input_dim = other_input_dim
        self.data_size = self.timestep_per_episode
        self.loss_log_file = self.loss_log + "loss.log"
        self.loss_per_episode = 0
        self.time_of_episode = 0

        print("Initializing Q-Network...")

        if model == 'None':
            self.model = Chain(
                conv1=L.Convolution2D(3 * self.hist_size, 32, 4, stride=2),
                bn1=L.BatchNormalization(32),
                conv2=L.Convolution2D(32, 32, 4, stride=2),
                bn2=L.BatchNormalization(32),
                conv3=L.Convolution2D(32, 32, 4, stride=2),
                bn3=L.BatchNormalization(32),
#                 conv4=L.Convolution2D(64, 64, 4, stride=2),
#                 bn4=L.BatchNormalization(64),
                l1=L.Linear(self.feature_dim + self.other_input_dim * self.hist_size, 128),
                l2=L.Linear(128, 128),
                l3=L.Linear(128, 96),
                l4=L.Linear(96, 64),
                q_value=L.Linear(64, self.num_of_actions)
            )
        else:
            with open(model, 'rb') as i:
                self.model = pickle.load(i)
                self.data_size = 0
        if self.use_gpu >= 0:
            self.model.to_gpu()

        self.optimizer = optimizers.RMSpropGraves()
        self.optimizer.setup(self.model)

        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.d = [np.zeros((self.agent_count, self.data_size, self.hist_size, 128, 128, 3), dtype=np.uint8),
                  np.zeros((self.agent_count, self.data_size, self.hist_size, self.other_input_dim), dtype=np.uint8),
                  np.zeros((self.agent_count, self.data_size), dtype=np.uint8),
                  np.zeros((self.agent_count, self.data_size, 1), dtype=np.float32),
                  np.zeros((self.agent_count, self.data_size, 1), dtype=np.bool)]

    def _reshape_for_cnn(self, state, batch_size, hist_size, x, y):
        
        state_ = np.zeros((batch_size, 3 * hist_size , 128, 128), dtype=np.float32)
        for i in range(batch_size):
            if self.hist_size == 1 :
                state_[i] = state[i][0].transpose(2, 0, 1)
            elif self.hist_size == 2:
                state_[i] = np.c_[state[i][0], state[i][1]].transpose(2, 0, 1)
            elif self.hist_size == 4:
                state_[i] = np.c_[state[i][0], state[i][1], state[i][2], state[i][3]].transpose(2, 0, 1)
        
        return  state_

    def forward(self, state_cnn, state_other, action, reward,
                state_cnn_dash, state_other_dash, episode_end):
        
        num_of_batch = state_cnn.shape[0]
        s_cnn = Variable(state_cnn)
        s_oth = Variable(state_other)
        s_cnn_dash = Variable(state_cnn_dash)
        s_oth_dash = Variable(state_other_dash)

        q = self.q_func(s_cnn, s_oth)  # Get Q-value
        
        max_q_dash_ = self.q_func(s_cnn_dash, s_oth_dash)
        if self.use_gpu >= 0:
            tmp = list(map(np.max, max_q_dash_.data.get()))
        else:
            tmp = list(map(np.max, max_q_dash_.data))
        max_q_dash = np.asanyarray(tmp, dtype=np.float32)
        if self.use_gpu >= 0:
            target = np.array(q.data.get(), dtype=np.float32)
        else:
            target = np.array(q.data, dtype=np.float32)

        for i in range(num_of_batch):
            tmp_ = reward[i] + (1 - episode_end[i]) * self.gamma * max_q_dash[i]

            action_index = self.action_to_index(action[i])
            target[i, action_index] = tmp_
        
        if self.use_gpu >= 0:
            loss = F.mean_squared_error(Variable(cuda.to_gpu(target)), q)
        else:
            loss = F.mean_squared_error(Variable(target), q)


        
        return loss, q

    def stock_experience(self, time,
                        state_cnn, state_other, action, reward,
                        state_cnn_dash, state_other_dash, episode_end_flag):

        for i in range(self.agent_count):
            self.d[0][i][self.data_index] = state_cnn[i].copy()
            self.d[1][i][self.data_index] = state_other[i].copy()
            self.d[2][i][self.data_index] = action[i].copy()
            self.d[3][i][self.data_index] = reward[i].copy()
            self.d[4][i][self.data_index] = episode_end_flag
                
        self.data_index += 1
        if self.data_index >= self.data_size:
            self.data_index -= self.data_size
            self.data_flag = True

    def experience_replay(self, time):
        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            replayRobotIndex = np.random.randint(0, self.agent_count, self.replay_size)
            if not self.data_flag:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, self.data_index, self.replay_size)
            else:
                replay_index = np.random.randint(0, self.data_size, self.replay_size)

            s_cnn_replay = np.ndarray(shape=(self.replay_size, self.hist_size, 128, 128, 3), dtype=np.float32)
            s_oth_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.other_input_dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_cnn_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, 128, 128, 3), dtype=np.float32)
            s_oth_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.other_input_dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
                
            for i in range(self.replay_size):
                s_cnn_replay[i] = np.asarray((self.d[0][replayRobotIndex[i]][replay_index[i]]), dtype=np.float32)
                s_oth_replay[i] = np.asarray((self.d[1][replayRobotIndex[i]][replay_index[i]]), dtype=np.float32)
                a_replay[i] = self.d[2][replayRobotIndex[i]][replay_index[i]]
                r_replay[i] = self.d[3][replayRobotIndex[i]][replay_index[i]]
                if (replay_index[i] + 1 >= self.data_size):
                    s_cnn_dash_replay[i] = np.array((self.d[0][replayRobotIndex[i]][replay_index[i] + 1 - self.data_size]), dtype=np.float32)
                    s_oth_dash_replay[i] = np.array((self.d[1][replayRobotIndex[i]][replay_index[i] + 1 - self.data_size]), dtype=np.float32)
                else:
                    s_cnn_dash_replay[i] = np.array((self.d[0][replayRobotIndex[i]][replay_index[i] + 1]), dtype=np.float32)
                    s_oth_dash_replay[i] = np.array((self.d[1][replayRobotIndex[i]][replay_index[i] + 1]), dtype=np.float32)
                episode_end_replay[i] = self.d[4][replayRobotIndex[i]][replay_index[i]]
    
            s_cnn_replay = self._reshape_for_cnn(s_cnn_replay, self.replay_size,
                                                 self.hist_size, 128, 128)
            s_cnn_dash_replay = self._reshape_for_cnn(s_cnn_dash_replay, self.replay_size,
                                                 self.hist_size, 128, 128)
                
            s_cnn_replay /= 255.0
            s_oth_replay /= 255.0
            s_cnn_dash_replay /= 255.0
            s_oth_dash_replay /= 255.0
    
            if self.use_gpu >= 0:
                s_cnn_replay = cuda.to_gpu(s_cnn_replay)
                s_oth_replay = cuda.to_gpu(s_oth_replay)
                s_cnn_dash_replay = cuda.to_gpu(s_cnn_dash_replay)
                s_oth_dash_replay = cuda.to_gpu(s_oth_dash_replay)

            # Gradient-based update
            loss, _ = self.forward(s_cnn_replay, s_oth_replay, a_replay, r_replay,
                                       s_cnn_dash_replay, s_oth_dash_replay,
                                       episode_end_replay)
            send_loss = loss.data
            with open(self.loss_log_file, 'a') as the_file:
                the_file.write(str(time) + "," + str(send_loss) + "\n")
            self.loss_per_episode += loss.data
            self.time_of_episode += 1
            self.model.zerograds()
            loss.backward()
            self.optimizer.update()

    def q_func(self, state_cnn, state_other):
        if self.use_gpu >= 0:
            num_of_batch = state_cnn.data.get().shape[0]
        else:
            num_of_batch = state_cnn.data.shape[0]

        h1 = F.tanh(self.model.bn1(self.model.conv1(state_cnn)))
        h2 = F.tanh(self.model.bn2(self.model.conv2(h1)))
        h3 = F.tanh(self.model.bn3(self.model.conv3(h2)))
#         h4 = F.tanh(self.model.bn4(self.model.conv4(h3)))
#         h5 = F.tanh(self.model.bn5(self.model.conv5(h4)))
        
        h4_ = F.concat((F.reshape(h3, (num_of_batch, self.feature_dim)),
                        F.reshape(state_other, (num_of_batch, self.other_input_dim * self.hist_size))), axis=1)
 
        h6 = F.relu(self.model.l1(h4_))
        h7 = F.relu(self.model.l2(h6))
        h8 = F.relu(self.model.l3(h7))
        h9 = F.relu(self.model.l4(h8))
        q = self.model.q_value(h9)
        return q

    def e_greedy(self, state_cnn, state_other, epsilon, reward):
        s_cnn = Variable(state_cnn)
        s_oth = Variable(state_other)
        q = self.q_func(s_cnn, s_oth)
        q = q.data        
        if self.use_gpu >= 0: 
            q_ = q.get()
        else:
            q_ = q
            
        index_action = np.zeros((self.agent_count), dtype=np.uint8)
        
        print(("agent"), end=' ')
        for i in range(self.agent_count):
            if np.random.rand() < epsilon:
                index_action[i] = np.random.randint(0, self.num_of_actions)
                print(("[%02d] Random(%2d)reward(%06.2f)" % (i, index_action[i], reward[i])), end=' ')
            else:
                index_action[i] = np.argmax(q_[i])
                print(("[%02d]!Greedy(%2d)reward(%06.2f)" % (i, index_action[i], reward[i])), end=' ')
            if i % 5 == 4:
                print(("\n     "), end=' ')
        
        del q_
        
        return self.index_to_action(index_action), q

    def index_to_action(self, index_of_action):
        index = np.zeros((self.agent_count), dtype=np.uint8)
        for i in range(self.agent_count):
            index[i] = self.enable_controller[index_of_action[i]]
        return index

    def action_to_index(self, action):
        return self.enable_controller.index(action)
