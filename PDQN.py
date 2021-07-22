# 混合DQN
import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
from ReplayBuffer import ReplayBuffer
from keras.layers import Dense, Flatten, Input, merge, Lambda, normalization, concatenate
import configparser
from ActorNetwork import ActorNetwork
HIDDEN1_UNITS = 25
HIDDEN2_UNITS = 20

class PDQN:
    def __init__(self, sess, env, ttype : float, trader_type, trader_id, state_dim, action_dim):
        self.sess = sess
        self.env = env
        self.type = ttype
        self.trader_type = trader_type
        self.trader_id = trader_id
        self.config = configparser.ConfigParser()
        self.config.read("./config.ini")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = 0.99
        self.epsilon = 1
        self.tau = float(self.config.get('Agent', 'TAU'))
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.learning_rate = float(self.config.get('Agent', 'LearningRateBR_a'))
        self.mini_batch = int(self.config.get('Agent', 'MiniBatchSize'))
        self.buffer_size = int(self.config.get('Utils', 'Buffersize'))
        self.update_target_every = 5
        self.loss = []
        self.grad = []
        self.test = []
        self._rl_memory = ReplayBuffer(self.buffer_size)
        # create predict model(main model)
        self.model_q_prediction, _, _ = self.create_model()
        self.model_q_target, _, _ = self.create_model()
        # target model
        #self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, self.mini_batch, self.tau, self.learning_rate)
        self.model_continuous_prediction, _, _ = self.create_actor_network(self.state_dim, self.action_dim)
        # Used to count when to update target network with prediction network's weights
        self.target_update_counter = 0
        self.market_number = int(self.config.get('Agent', 'Market'))
        self.alpha = 0.001 # satisfied Robbins-Monro condition
        self.beta = 0.003

    def create_model(self):
        '''
        model = Sequential()
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu", input_shape=(self.state_dim, self.action_dim[1])))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=self.action_dim[0], activation="linear"))
        print("model_summary:", model.summary())
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        '''
        S = Input(shape=[1, self.state_dim], name='input_s')
        bn_s = normalization.BatchNormalization()(S)
        A = Input(shape=[1, self.action_dim[0] - 1], name='input_a')  # todo  input continuous action dim
        bn_a = normalization.BatchNormalization()(A)
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(bn_s)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(bn_a)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = concatenate([h1, a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        #h4 = Dense(HIDDEN1_UNITS, activation='relu')(h3)
        V = Dense(self.action_dim[0], activation='linear')(h3) #output，2个离散维度,
        model = Model(inputs=[S, A], outputs=V)
        print("model_summary_discrete:", model.summary())
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

    def create_target_model(self):
        model = Sequential()
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu", input_dim=self.state_dim[0]))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=self.action_dim[0], activation="linear"))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def create_actor_network(self, state_dim, action_dim): #输入 state， 输出 连续动作参数
        S = Input(shape=[1, state_dim])
        bn =normalization.BatchNormalization()(S)
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(bn)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        out = Dense(action_dim[1], activation='sigmoid')(h1)
        model = Model(inputs=S, outputs=out)
        print("model_summary_continuous:", model.summary())
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam) #todo 是什么损失函数
        return model, model.trainable_weights, S


    #大Q   Q(s,k,xk)的最大值==> output是Q值最大的K (也就是离散动作)
    def get_qs(self, state, continuous_action_param, discrete_action): #discrete_action [0, 1, 2]
        state = np.reshape(state, (1, 1, self.state_dim))
        continuous_action_param = np.reshape(continuous_action_param, (1, 1, self.action_dim[0] - 1))  # 两个市场，3个维度
        discrete_result = self.model_q_prediction.predict([state, continuous_action_param])  #input tate
        return discrete_result[0]   #应该是list

    def compute_action_parameter(self, state, discre_k): # x_k # 产生连续的动作参数
        continuous_action = self.model_continuous_prediction.predict(state)   # 计算actor
        return continuous_action



    def chooseAction(self, state, trader_type, type, count):
        action = [0, 0]
        state = np.reshape(state, (1, 1, self.state_dim))
        if random.random() > self.epsilon:
            continuous_action_param = self.compute_action_parameter(state, 0)
            continuous_action_param = continuous_action_param[0][0][0]
            f = open('%s_%f.txt' %(self.trader_type, self.type), 'a+')
            print("continuous_action_param:", continuous_action_param, file=f)
            a1 = self.get_qs(state, continuous_action_param, 0)  # 计算此状态下，每个动作的Q值，type=list
            a1 = np.argmax(a1)
            action[0] = a1
            action[1] = continuous_action_param
        else:
            #不进入市场的概率太大，导致学到的太多不进入市场的概率,从概率分布中选择一个
            if random.random() < 0.15:   #no enter market
                act_1 = 0
            else:
                act_1 = random.randint(1, self.market_number) #在多个市场中选择
            act_2 = 0
            if act_1 != 0:
                act_2 = random.uniform(0, 1)
            action[0] = act_1
            action[1] = act_2

        if self.epsilon > self.epsilon_min:  # epsilon 需要衰减
            self.epsilon *= self.epsilon_decay

        return action

        # add to replay_buffer,action 2 dim
    def remember_for_rl(self, state, action, reward, next_state):
        self._rl_memory.add(state, action, reward, next_state)

    # train network, target_model is no need train, it's weight is copy from prediction_model
    def update_train(self):
        if self._rl_memory.size() > self.mini_batch:
            w_t = self.model_q_prediction.get_weights()
            theta_t = self.model_continuous_prediction.get_weights()
            batch = self._rl_memory.getBatch(self.mini_batch) #get batch
            states = np.array([e[0] for e in batch])   # shape(batch, 1, 6)

            x_list = self.model_continuous_prediction.predict(states)
            qs_list = self.model_q_prediction.predict([states, x_list])  # 每个state下，所有离散action的Q值

            next_states = np.array([e[3] for e in batch])
            #next_states = np.reshape(next_states, (self.mini_batch, self.state_dim))
            x_k_list = self.model_continuous_prediction.predict(next_states)  # 根据next_state, 求出连续性参数
            target_q_list = self.model_q_target.predict([next_states, x_k_list])  #根据连续性参数，求最大的q,

            x_s = []
            x_param = []
            y = []
            z = []
            # 遍历batch
            for index, (state, action, reward, next_state) in enumerate(batch): # todo action can't be none
                if True:    # shape is (1, 6) (1, 2)..
                    max_q = np.max(target_q_list[index])
                    new_q = reward + self.discount * max_q
                else:
                    new_q = reward
                #用给的state 更新Q值
                current_qs = qs_list[index]  # tuple
                current_qs = np.reshape(current_qs, (2, ))
                current_qs = current_qs.tolist()
                i = int(action[0][0])  # 0 or 1
                #print("action:", action)
                current_qs[i] = new_q

                #添加到我们的训练结果中
                state = np.reshape(state, (self.state_dim,))
                state = state.tolist()
                x_s.append(state)
                x_k = np.reshape(x_list[index], (self.action_dim[1], ))
                x_k = x_k.tolist()[0]
                x_param.append(x_k)
                y.append(current_qs)
                z.append(action[0][1]) #todo 连续的动作

            x_s = np.reshape(x_s, (self.mini_batch, 1, self.state_dim))
            x_param = np.reshape(x_param, (self.mini_batch, 1, 1))
            y = np.reshape(y, (self.mini_batch, 1, self.action_dim[0]))  #每个state下，所有离散action的Q值

            # input => state, output=> 此state下 每个action的Q值
            self.model_q_prediction.fit([x_s, x_param], np.array(y), batch_size=self.mini_batch, verbose=0) # train...

            z = np.reshape(z, (self.mini_batch, 1, self.action_dim[1]))
            # input state output=>param[0-1]

            self.model_continuous_prediction.fit(np.array(x_s), np.array(z), batch_size=self.mini_batch, verbose=0)
            if self.target_update_counter > self.update_target_every:
                self.train_discrete(w_t)
                self.train_continuous(theta_t)
                self.model_q_target.set_weights(self.model_q_prediction.get_weights())
                self.target_update_counter = 0


    def train_discrete(self, w_t):
        target_weights = self.model_q_prediction.get_weights() # w_t
        for i in range(len(target_weights)):
            target_weights[i] = w_t - self.alpha * target_weights[i]
        self.model_q_prediction.set_weights(target_weights)

    def train_continuous(self, theta_t):
        target_theta = self.model_continuous_prediction.get_weights()
        for i in range(len(target_theta)):
            target_theta[i] = theta_t - self.beta * target_theta[i]
        self.model_continuous_prediction.set_weights(target_theta)


