# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np
from chainer import cuda
import gc

from q_net import QNet
from PIL import Image



class CnnDqnAgent(object):
    policy_frozen = False
    epsilon_delta = 1.0 / (6 * 10 ** 4)
    min_eps = 0.1

    actions = [0, 1, 2, 3, 4, 5, 6, 7]

    image_feature_dim = 14 * 14  
    image_feature_count = 32
    image_dim = 128 * 128
    avgloss_log = '/home/ohk/Documents/playground/Assets/log/'

    def _observation_to_state_cnn(self, observation):
        temp = []
        for i in range(len(observation["image"])):
            temp.append(np.r_[observation["image"][i]])
        return np.r_[temp]
    
    def _observation_to_state_other(self, observation):
        temp = []
        # change in another network structure
        for i in range(len(observation["ir"])):
            temp.append(np.r_[observation["ir"][i],
                              observation["compass"][i],
                              observation["target"][i]])
        return np.r_[temp]
    
    def _reshape_for_cnn(self, state, hist_size, x, y):
        
        state_ = np.zeros((self.agent_count, 3 * hist_size , 128, 128), dtype=np.float32)

        for i in range(self.agent_count):
            if hist_size == 1 :
                state_[i] = state[i][0].transpose(2, 0, 1)
            elif hist_size == 2:
                state_[i] = np.c_[state[i][0], state[i][1]].transpose(2, 0, 1)
            elif hist_size == 4:
                state_[i] = np.c_[state[i][0], state[i][1], state[i][2], state[i][3]].transpose(2, 0, 1)
        
        return  state_

    def agent_init(self, **options):
        self.use_gpu = options['use_gpu']
        self.agent_count = options['agent_count']
        self.image_count = options['rgb_image_count']
        self.depth_image_dim = options['depth_image_dim']
        self.ir_idm = options['ir_dim']
        self.ground_dim = options['ground_dim']
        self.compass_dim = options['compass_dim']
        self.target_dim = options['target_dim']
        self.model = options['model']

        self.cnn_input_dim = self.image_dim * self.image_count
        self.feature_dim = self.image_feature_dim * self.image_feature_count
        self.other_input_dim = self.depth_image_dim + self.ir_idm + self.ground_dim + self.compass_dim + self.target_dim
        
        self.time = 1
        self.epsilon = 1.0
        self.avgloss_log_file = self.avgloss_log + "avg_loss.log"
        
        if self.model != 'None':
            self.policy_frozen = False
            self.epsilon = 0.5

            
        self.q_net = QNet(self.use_gpu, self.actions, self.cnn_input_dim,
                          self.feature_dim, self.agent_count, self.other_input_dim, self.model)

    def agent_start(self, observation, reward):
        obs_cnn_array = self._observation_to_state_cnn(observation)
        obs_other_array = self._observation_to_state_other(observation)

        # Initialize State
        self.state_cnn = np.zeros((self.agent_count, self.q_net.hist_size, 128, 128, 3),
                                  dtype=np.uint8)
                
        for i in range(self.agent_count):
            self.state_cnn[i][self.q_net.hist_size - 1] = obs_cnn_array[i]
        state_cnn_ = self._reshape_for_cnn(self.state_cnn, self.q_net.hist_size, 128, 128)
        state_cnn_ /= 255.0
        
        self.state_other = np.zeros((self.agent_count, self.q_net.hist_size, self.other_input_dim), dtype=np.uint8)
        for i in range(self.agent_count):
            self.state_other[i][self.q_net.hist_size - 1] = obs_other_array[i]
        state_other_ = np.asanyarray(self.state_other.reshape(self.agent_count, self.q_net.hist_size * self.other_input_dim), dtype=np.float32)
        state_other_ /= 255.0

        if self.use_gpu >= 0:
            state_cnn_ = cuda.to_gpu(state_cnn_)
            state_other_ = cuda.to_gpu(state_other_)

        if self.policy_frozen is False:  # Learning ON/OFF
            if self.q_net.initial_exploration <= self.time:
                self.epsilon -= self.epsilon_delta
                if self.epsilon < self.min_eps:
                    self.epsilon = self.min_eps
                eps = self.epsilon
                print(("\naTraining Now. Time step : %d Epsilon : %.6f" % (self.time, eps)))
            else:  # Initial Exploation Phase
                eps = 1.0
                print(("\naInitial Exploration S : %d/%d Epsilon : %.6f" % (self.time, self.q_net.initial_exploration, eps)))

        # Generate an Action e-greedy
        action, q_now = self.q_net.e_greedy(state_cnn_, state_other_, self.epsilon, reward)

        # Update for next step
        self.last_action = action.copy()  
        self.last_state_cnn = self.state_cnn.copy()
        self.last_state_other = self.state_other.copy()
        
        del state_cnn_, state_other_, obs_cnn_array, obs_other_array
        gc.collect()

        self.time += 1
        
        return action, q_now

    def agent_step(self, reward, observation):
        obs_cnn_array = self._observation_to_state_cnn(observation)
        obs_other_array = self._observation_to_state_other(observation)
        
#         img = observation["image"][0]
#         img.save("img.png")

        # Compose State : 4-step sequential observation
        for i in range(self.agent_count):
            if self.q_net.hist_size == 4:
                self.state_cnn[i] = np.asanyarray([self.state_cnn[i][1], self.state_cnn[i][2],
                                                   self.state_cnn[i][3], obs_cnn_array[i]], dtype=np.uint8)
                if(obs_other_array.size != 0):
                    self.state_other[i] = np.asanyarray([self.state_other[i][1], self.state_other[i][2],
                                                         self.state_other[i][3], obs_other_array[i]], dtype=np.uint8)
            elif self.q_net.hist_size == 2:
                self.state_cnn[i] = np.asanyarray([self.state_cnn[i][1], obs_cnn_array[i]], dtype=np.uint8)
                if(obs_other_array.size != 0):
                    self.state_other[i] = np.asanyarray([self.state_other[i][1], obs_other_array[i]], dtype=np.uint8)
            elif self.q_net.hist_size == 1:
                self.state_cnn[i] = np.asanyarray([obs_cnn_array[i]], dtype=np.uint8)
                if(obs_other_array.size != 0):
                    self.state_other[i] = np.asanyarray([obs_other_array[i]], dtype=np.uint8)
            else:
                print("self.DQN.hist_size err")

        state_cnn_ = self._reshape_for_cnn(self.state_cnn, self.q_net.hist_size, 128, 128)
        state_cnn_ /= 255.0
        
        state_other_ = np.asanyarray(self.state_other.reshape(self.agent_count, self.q_net.hist_size * self.other_input_dim), dtype=np.float32)
        state_other_ /= 255.0

        if self.use_gpu >= 0:
            state_cnn_ = cuda.to_gpu(state_cnn_)
            state_other_ = cuda.to_gpu(state_other_)

        # Exploration decays along the time sequence
        if self.policy_frozen is False:  # Learning ON/OFF
            if self.q_net.initial_exploration <= self.time:
                self.epsilon -= self.epsilon_delta
                if self.epsilon < self.min_eps:
                    self.epsilon = self.min_eps
                eps = self.epsilon
                print(("\nbTraining Now. Time step : %d Epsilon : %.6f" % (self.time, eps)))
            else:  # Initial Exploation Phase
                eps = 1.0
                print(("\nInitial Exploration : %d/%d Epsilon : %.6f" % (self.time, self.q_net.initial_exploration, eps)))
        else:  # Evaluation
            eps = 0.05
            print(("\nPolicy is Frozen. Time step : %d Epsilon : %.6f" % (self.time, eps)))


        # Generate an Action by e-greedy action selection
        action, q_now = self.q_net.e_greedy(state_cnn_, state_other_, eps, reward)
                
        del state_cnn_, state_other_, obs_cnn_array, obs_other_array
        gc.collect()
        
        return action, eps, q_now

    def agent_step_update(self, reward, action, eps):
        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state_cnn, self.last_state_other,
                                        self.last_action, reward, self.state_cnn, self.state_other,
                                        False)
            self.q_net.experience_replay(self.time)

        if self.policy_frozen is False:
            self.last_action = action.copy()  # copy.deepcopy(action)
            self.last_state_cnn = self.state_cnn.copy()
            self.last_state_other = self.state_other.copy()

        self.time += 1

    def agent_end(self, reward):  # Episode Terminated
        print(('episode finished. Time step : %d' % self.time))
        
        print(("agent"), end=' ')
        for i in range(self.agent_count):
            print(("[%02d]        ( )reward(%06.2f)" % (i, reward[i])), end=' ')
            if i % 5 == 4:
                print(("\n     "), end=' ')

        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state_cnn, self.last_state_other,
                                        self.last_action, reward,
                                        self.last_state_cnn, self.last_state_other,
                                        True)
            self.q_net.experience_replay(self.time)
        avg_episode_loss = 0
        if self.q_net.time_of_episode != 0:
            avg_episode_loss = self.q_net.loss_per_episode / self.q_net.time_of_episode
        self.q_net.loss_per_episode = 0
        self.q_net.time_of_episode = 0
        with open(self.avgloss_log_file, 'a') as the_file:
                the_file.write(str(self.time) + "," + str(avg_episode_loss) + "\n")
        # Time count
#         if self.policy_frozen is False:
        self.time += 1
