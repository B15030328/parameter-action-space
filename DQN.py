import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ReplayBuffer import ReplayBuffer
import configparser
from utils import plotLearning
import time
HIDDEN1_UNITS = 25
HIDDEN2_UNITS = 20
class DQN:
    def __init__(self, sess, env,  ttype : float, trader_type, trader_id, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        #self.sess = sess
        #self.env = env
        self.type = ttype
        self.config = configparser.ConfigParser()
        self.config.read("./config.ini")
        self.trader_type = trader_type
        self.trader_id = trader_id
        self.gamma = 0.95
        self.discount = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = float(self.config.get('Agent', 'LearningRateBR_a'))
        self.mini_batch = int(self.config.get('Agent', 'MiniBatchSize'))
        self.buffer_size = int(self.config.get('Utils', 'Buffersize'))
        self.update_target_every = 5
        self.loss = []
        self.grad = []
        self.test = []
        self.replay_memeory = deque(maxlen=self.buffer_size)
        #self._rl_memory = ReplayBuffer(self.buffer_size)
        # create predict model(main model)
        self.model_prediction = self.create_model()
        # target model
        self.model_target = self.create_model()
        self.model_target.set_weights(self.model_prediction.get_weights()) # set the same weight
        # Used to count when to update target network with prediction network's weights
        self.target_update_counter = 0


    def create_model(self):
        model = Sequential()
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu", input_dim=self.state_dim))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=self.action_dim, activation="linear"))
        print("model_summary:", model.summary())
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get Q value ==> prediction_model's Q value
    def get_qs(self, state): # state type is list
        state = np.reshape(state, (1, self.state_dim))
        sss = self.model_prediction.predict(state)
        return sss[0]

    # add to replay_buffer
    def remember_for_rl(self, state, action, reward, next_state):
        #self._rl_memory.add(state, action, reward, next_state)
        transition = (state, action, reward, next_state)
        self.replay_memeory.append(transition)

    # train network, target_model is no need train, it's weight is copy from prediction_model
    def update_train(self):
        if len(self.replay_memeory) > self.mini_batch:
            batch = random.sample(self.replay_memeory, self.mini_batch)
            # Get current states from minibatch, then query NN model_prediction for current Q values
            #qs_list = self.model_prediction.predict(np.reshape(states, [1, self.state_dim]))
            states = np.array([e[0] for e in batch])
            #states = np.reshape(states, (self.mini_batch, self.state_dim))
            qs_list = self.model_prediction.predict(states)
            actions = np.array([e[1] for e in batch])

            # Get next_states from minibatch, then query NN model_target for target Q values
            # When using target network, query it, otherwise main network should be queried
            next_states = np.array([e[3] for e in batch])
            taret_qs_list = self.model_target.predict(next_states)

            x = []
            y = []
            # 遍历batch
            for index, (state, action, reward, next_state) in enumerate(batch):
                if True:
                   max_target_q = np.max(taret_qs_list[index])
                   new_q = reward + self.discount * max_target_q
                else:
                    new_q = reward
                #用给的state 更新Q值
                current_qs = qs_list[index]  # 用model_prediction(state)预测出的
                current_qs[action] = new_q   # update Q value

                #添加到我们的训练结果中
                x.append(state)   # tuple （2,）
                y.append(current_qs)  #tuple (4, )

            self.model_prediction.fit(np.array(x), np.array(y), batch_size=self.mini_batch, verbose=0)  # input => state, output=> 此state下
                                                                                                        # 每个离散action的Q值
            self.target_update_counter += 1
            #更新计数器
            #如果是最后一步，需要计数+1 todo

            #如果计数达到，就更新target network, with weight of main network
            if self.target_update_counter > self.update_target_every:
                self.model_target.set_weights(self.model_prediction.get_weights())
                self.target_update_counter = 0

            if self.epsilon > self.epsilon_min: # epsilon 需要衰减
                self.epsilon *= self.epsilon_decay

    def chooseAction(self, state, trader_type, type, count):
        if random.random() > self.epsilon:
            action = np.argmax(self.get_qs(state))   #model.predict()
        else:
            action = np.random.randint(0, self.action_dim) # random 产生一个action
        return action

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(1)
    done = False
    agent = DQN(None, None, 0.45, 'S', 3, 6, 2)
    env = gym.make('Pendulum-v0')
    np.random.seed(0)
    score_history = []
    for i in range(1000):
        s = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.chooseAction(s, 0, 0, 0)
            print("act:", act)
            new_state, reward, done, info = env.step(act)
            agent.remember_for_rl(s, act, reward, new_state)
            agent.update_train()
            score += reward
            obs = new_state
            env.render()
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score,
              'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

    filename = 'Pendulum-alpha00005-beta0005-800-600-optimized.png'
    plotLearning(score_history, filename, window=100)
