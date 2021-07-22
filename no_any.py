from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
import configparser
import numpy as np
from OU import OU
from OU import OUNoise
from keras.models import Sequential, Model
from keras.layers import Dense, Input,normalization, concatenate
import random
from keras.optimizers import Adam
import copy
import tensorflow as tf
import math
import sys

HIDDEN1_UNITS = 25
HIDDEN2_UNITS = 20
class Agent:
    def __init__(self, sess, env, ttype : float, trader_type, id : int, state_dim, action_dim):
        self.sess = sess
        self.env = env
        self.config = configparser.ConfigParser()
        self.config.read("./config.ini")
        self.type = ttype
        self.trader_type = trader_type
        self.id = id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 1
        self.discount = 0.99
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.update_target_every = 5
        self.batch_size = int(self.config.get('Agent', 'MiniBatchSize'))
        self.var = float(self.config.get('Agent', 'VAR'))
        self.learning_rate = float(self.config.get('Agent', 'LearningRateAR'))
        self.gamma = float(self.config.get('Agent', 'Gamma'))
        self.tau = float(self.config.get('Agent', 'TAU'))      #Target Network HyperParameters
        self.lra = float(self.config.get('Agent', 'LearningRateBR_a'))      #Learning rate for Actor
        self.lrc = float(self.config.get('Agent', 'LearningRateBR_c'))      #Lerning rate for Critic
        self._rl_memory = ReplayBuffer(int(self.config.get('Agent', 'MRLSize')))
        self.buffer_size = int(self.config.get('Agent', 'MRLSize'))
        self.alpha = 0.001  # satisfied Robbins-Monro condition
        self.beta = 0.003
        self.episodes = int(self.config.get('Common', 'Episodes'))

        # create predict model(main model)
        self.model_prediction, _, _ = self.create_model()
        # target model
        self.model_target, _, _ = self.create_model()
        self.model_target.set_weights(self.model_prediction.get_weights())  # set the same weight
        # Used to count when to update target network with prediction network's weights
        self.target_update_counter = 0
        #continuous part
        self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim[1] - 1, self.batch_size, self.tau, self.lra)
        #self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim[1], self.batch_size, self.tau, self.lrc)
        self.loss = []
        self.grad = []
        self.test = []
        self.num = 0
        #此处actor和critic network是否需要修改

    def _create_model(self):
        model = Sequential()
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu", input_dim=self.state_dim))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=HIDDEN1_UNITS, activation="relu"))
        model.add(Dense(units=self.action_dim[0], activation="linear"))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def create_model(self):
        S = Input(shape=[1, self.state_dim], name='input_s')
        bn_s = normalization.BatchNormalization()(S)
        A = Input(shape=[1, self.action_dim[1]], name='input_a')
        bn_a = normalization.BatchNormalization()(A)
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(bn_s)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(bn_a)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = concatenate([h1, a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        # h4 = Dense(HIDDEN1_UNITS, activation='relu')(h3)
        V = Dense(self.action_dim[0], activation='linear')(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


    # train network, target_model is no need train, it's weight is copy from prediction_model
    def update_train(self, pricing_1, pricing_2):
        if self._rl_memory.size() > self.batch_size:
            batch = self._rl_memory.getBatch(self.batch_size)
            # Get current states from minibatch, then query NN model_prediction for current Q values
            states = np.array([e[0] for e in batch])
            actions = np.array(e[1] for e in batch)
            next_states = np.array([e[3] for e in batch])
            # todo 更新连续Net
            continuouse_paprams = [[[0, 0]]] * self.batch_size
            next_continuouse_paprams = [[[0, 0]]] * self.batch_size
            continuous_action1, continuous_action2 = 0, 0
            continuous_action3, continuous_action4 = 0, 0
            if self.trader_type == 'S':
                self.actor.model.load_weights('./data/market_' + str(pricing_1) + '_seller_' + str(self.type) + '.h5')
                continuous_action1 = self.actor.model.predict(states)
                continuous_action3 = self.actor.model.predict(next_states)
                self.actor.model.load_weights('./data/market_' + str(pricing_2) + '_seller_' + str(self.type) + '.h5')
                continuous_action2 = self.actor.model.predict(states)
                continuous_action4 = self.actor.model.predict(next_states)

            if self.trader_type == 'B':
                self.actor.model.load_weights('./data/market_' + str(pricing_1) + '_buyer_' + str(self.type) + '.h5')
                continuous_action1 = self.actor.model.predict(states)
                continuous_action3 = self.actor.model.predict(next_states)
                self.actor.model.load_weights('./data/market_' + str(pricing_2) + '_buyer_' + str(self.type) + '.h5')
                continuous_action2 = self.actor.model.predict(states)
                continuous_action4 = self.actor.model.predict(next_states)

            for i in range(len(continuous_action1)):
                continuouse_paprams[i][0][0] = continuous_action1[i][0][0]
                continuouse_paprams[i][0][1] = continuous_action2[i][0][0]
                next_continuouse_paprams[i][0][0] = continuous_action3[i][0][0]
                next_continuouse_paprams[i][0][1] = continuous_action4[i][0][0]
            #continuouse_paprams.append(continuous_action)
            #continuouse_paprams = self.actor.model.predict(states)  #求连续参数
            #self.actor.model.fit(loss, batch_size=self.batch_size, verbose=0)
            continuouse_paprams = np.reshape(continuouse_paprams, (self.batch_size, 1, self.action_dim[1]))
            qs_list = self.model_prediction.predict([states, continuouse_paprams])  # 求qs_list,每个list 是此时state对应所有Q值
            # qs_list type tuple (64,1,2)
            #self.actor.sum_loss(qs_list)  # 入参是batch长度的Q值列表
            #loss = np.sum(np.reshape(qs_list, (self.batch_size, self.action_dim[0])), axis=1)
            #loss = loss * -1
            #self.model_target.fit([[np.array(x), np.array(np.reshape(continuouse_paprams, (self.batch_size, 1, self.action_dim[1])))],])
            #self.actor.model.fit(x=np.array(states), y=None, batch_size=self.batch_size, verbose=0)  # 此处输入的是(1,2)
            #todo  有问题
            #loss = np.reshape(loss, (self.batch_size, 1, 1))
            #self.actor.train(states, loss)
            # Get next_states from minibatch, then query NN model_target for target Q values
            # When using target network, query it, otherwise main network should be queried
            #next_continuous_param = self.actor.target_model.predict(next_states)

            #next_continuous_param = self.actor.target_model.predict(next_states)
            next_continuouse_paprams = np.reshape(next_continuouse_paprams, (self.batch_size, 1, self.action_dim[1]))
            target_qs_list = self.model_target.predict([next_states, next_continuouse_paprams])  # output所有的Q值
            # 连续网络如何更新参数
            # action continuous， 作为输入， 获得k parameter。组合起来，
            # 放入Q target网络，计算Q值
            x = []
            y = []
            params = []
            # 遍历batch
            for index, (state, action, reward, next_state) in enumerate(batch):  # todo 此处action是2维
                if True:  # action (1,3)
                   max_target_q = np.max(target_qs_list[index])
                   new_q = reward + self.discount * max_target_q
                else:
                    new_q = reward

                #用给的state 更新Q值
                current_qs = qs_list[index].tolist()[0]  # 用model_prediction(state)预测出的
                dis_action = int(action[0][0])
                current_qs[dis_action] = new_q   # update Q value  type (1,2)

                #添加到我们的训练结果中
                x.append(np.array(state).flatten())
                y.append(np.reshape(current_qs, (self.action_dim[0], )))
                #params.append()
            x = np.reshape(x, (self.batch_size, 1, self.state_dim))
            y = np.reshape(y, (self.batch_size, 1, self.action_dim[0]))
            self.model_prediction.fit([np.array(x), np.array(np.reshape(continuouse_paprams, (self.batch_size, 1, self.action_dim[1])))],  np.array(y), batch_size=self.batch_size, validation_data=([np.array(x),  np.reshape(continuouse_paprams, (self.batch_size, 1, self.action_dim[1]))], np.array(y)), verbose=0)  # input => state, output=> 此state下
                                                                                                        # 每个离散action的Q值
            self.target_update_counter += 1
            #更新计数器
            #如果计数达到，就更新target network, with weight of main network
            if self.target_update_counter > self.update_target_every:
                self.model_target.set_weights(self.model_prediction.get_weights())
                self.target_update_counter = 0

            if self.epsilon > self.epsilon_min: # epsilon 需要衰减
                self.epsilon *= 0.995

    # get Q value ==> prediction_model's Q value
    def get_qs(self, state, continuous_actions : list): # state type is list
        #state = np.reshape(state, (1, self.state_dim))
        continuous_actions = np.reshape(continuous_actions, (1, 1, self.action_dim[1]))
        sss = self.model_prediction.predict([state, continuous_actions])   #todo 此处输入是state，输出是每个action的Q值
        return sss[0]

    def remember_for_rl(self, state, action, reward, next_state):
        self._rl_memory.add(state, action, reward, next_state)


    def remember_best_response(self, state, action):
        pass


    def act_best_response(self, state):
        action_original = self.actor.model.predict(state)
        return action_original


    def train_discrete(self, w_t):
        target_weights = self.model_prediction.get_weights() # w_t
        for i in range(len(target_weights)):
            target_weights[i] = w_t - self.alpha * target_weights[i]
        self.model_prediction.set_weights(target_weights)

    def train_continuous(self, theta_t):
        target_theta = self.actor.model.get_weights()
        for i in range(len(target_theta)):
            target_theta[i] = theta_t - self.beta * target_theta[i]
        self.actor.model.set_weights(target_theta)
    #更新连续的网络
    # todo, 连续NN, 输入是【state, action(1个维度)】
    def update_strategy1(self):
        if self._rl_memory.size() > self.batch_size: #todo
            loss = 0
            batch = self._rl_memory.getBatch(self.batch_size)
            states = np.array([e[0] for e in batch])  # type = (1,1,8)
            actions = np.array([e[1] for e in batch])  # type (1,1,3)
            concat_state = []  # list, type为7维state
            continous_actions = []
            for i in range(len(states)):
                continous_actions.append(actions[i][0][1])  # 离散的动作
                temp = []
                a = actions[i][0][0]   # 离散的动作
                s = states[i]          # 将state 与 离散action结合
                s = np.reshape(s, (self.state_dim, ))
                for num in range(len(s)):
                    temp.append(s[num])

                a = 0.3 if a == 0 else 0.6 if a == 1 else 0.9
                temp.append(a)
                temp = np.reshape(temp, (1, 1, self.state_dim + 1))  # todo 验证是否为(1,1,9)
                concat_state.append(temp)

            # 将每次用于训练的batch_size print出来
            rewards = np.array([e[2] for e in batch])
            new_states = np.array([e[3] for e in batch])
            concat_new_states = []  # new_state,  7维度state
            for j in range(len(new_states)):
                temp = []
                a = actions[j][0][0]  # 离散的动作
                s = new_states[j]     # 将state 与离散的action结合
                s = np.reshape(s, (self.state_dim, ))
                for num in range(len(s)):
                    temp.append(s[num])

                param_discrete_action = 0.3 if a == 0 else 0.6 if a == 1 else 0.9
                temp.append(param_discrete_action)
                temp = np.reshape(temp, (1, 1, self.state_dim + 1))
                concat_new_states.append(temp)


            concat_state = np.reshape(concat_state, (self.batch_size, 1, self.state_dim + 1))   # todo check， 将其reshape成(32,1,9)
            concat_new_states = np.reshape(concat_new_states, (self.batch_size, 1, self.state_dim + 1))
            continous_actions = np.reshape(continous_actions, (self.batch_size, 1, 1))
            y_t = continous_actions
            target = self.critic.target_model.predict([concat_new_states, self.actor.target_model.predict(concat_new_states, batch_size=self.batch_size)], batch_size=self.batch_size)
           # target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            for k in range(len(batch)):
                if y_t[k] is None or target[k] is None or rewards[k] is None:
                    continue
                else:
                    y_t[k] += rewards[k] + self.gamma * target[k]
            y_t = np.reshape(y_t, (self.batch_size, 1, 1))
            loss += self.critic.model.train_on_batch([concat_state, continous_actions], y_t)  # y_t == target q
            self.loss.append(loss)
            a_for_grad = self.actor.model.predict(concat_state)
            grads = self.critic.gradients(self.sess, concat_state, a_for_grad) # loss 上的梯度
            self.grad.append(grads)
            # update actor policy using sampled gradient
            self.actor.train(self.sess, concat_state, grads)  # 根据loss 的梯度求得的极值
            self.actor.target_train()
            self.critic.target_train()
        return

    def update_strategy(self):  #连续动作
        if self._rl_memory.size() > self.batch_size: #todo
            loss = 0
            batch = self._rl_memory.getBatch(self.batch_size)
            sss = []
            states = np.array([e[0] for e in batch])  #type  (batch,1,8)
            actions = np.array([e[1] for e in batch])
            discrete_actions = []
            continuous_actions = []
            for i in range(len(actions)):
                temp = [actions[i][0][1], actions[i][0][2]]
                discrete_actions.append(actions[i][0][0])
                continuous_actions.append(np.reshape(temp, (1, 1, self.action_dim[1])))
            continuous_actions = np.reshape(continuous_actions, (self.batch_size, 1, self.action_dim[1]))

            # 将每次用于训练的batch_size print出来
            rewards = np.array([e[2] for e in batch])
            new_states = np.array([e[3] for e in batch])
            y_t = copy.copy(continuous_actions)
            target = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states, batch_size=self.batch_size)], batch_size=self.batch_size)
           # target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            for k in range(len(batch)):
                if y_t[k] is None or target[k] is None or rewards[k] is None:
                    continue
                else:
                    #index_y = int(discrete_actions[k])
                    #temp_y_t = np.array(y_t[k]).tolist()[0]  #  tuple (1,2)
                    #temp_reward = rewards[k]  # tuple (1,)
                    #temp_target = np.array(target[k]).tolist()[0]  # tuple (1, 2)
                    #temp_y_t[index_y] = temp_reward + self.gamma * temp_target[index_y]
                    #y_t[k] = tuple(temp_y_t)
                    temp_reward = rewards[k]
                    temp_target = target[k]
                    y_t[k] = temp_reward + self.gamma * temp_target  # todo 这里一次更新2个维度，思考是否一次更新选中的那个维度
                    #y_t[k] = rewards[k] + self.gamma * target[k]
            loss += self.critic.model.train_on_batch([states, continuous_actions], y_t)  # y_t == target q
            self.loss.append(loss)
            a_for_grad = self.actor.model.predict(states)
            grads = self.critic.gradients(states, a_for_grad)  # loss 上的梯度
            self.grad.append(grads)
            # update actor policy using sampled gradient
            self.actor.train(states, grads)  # 根据loss 的梯度求得的极值
            self.actor.target_train()
            self.critic.target_train()
        return
    '''
    此处的思路：先选择离散的action，然后将其作为state的一个维度，
    基于新的state选择动作。此处也是要加noise.
    这样的话，每个agent是能够分清楚不同的离散选择应该做出合适的动作
    '''
    def chooseAction(self, state, trader_type, type, count, market_number, pricing_1, pricing_2):
        #先选择连续的动作，再选择离散的动作
        #1 根据state输出最佳连续的参数，注意此处是对应输出所有离散的动作的连续参数
        discrete_action = 0
        continuous_action = [0] * self.action_dim[1]
        action = [0] * (self.action_dim[1] + 1)  # 对应的是选择离散动作0，则连续参数为a[0]...
        state = np.reshape(state, (1, 1, self.state_dim))
        # get 每个离散动作的 连续参数
        s1 = 's'
        s2 = 's'
        if trader_type == 'S':
            self.actor.model.load_weights('./data/market_' + str(pricing_1) + '_seller_' + str(type) + '.h5')
            continuous_action[0] = self.actor.model.predict(state)[0][0][0]
            self.actor.model.load_weights('./data/market_' + str(pricing_2) + '_seller_' + str(type) + '.h5')
            continuous_action[1] = self.actor.model.predict(state)[0][0][0]
        if trader_type == 'B':
            self.actor.model.load_weights('./data/market_' + str(pricing_1) + '_buyer_' + str(type) + '.h5')
            continuous_action[0] = self.actor.model.predict(state)[0][0][0]
            self.actor.model.load_weights('./data/market_' + str(pricing_2) + '_buyer_' + str(type) + '.h5')
            continuous_action[1] = self.actor.model.predict(state)[0][0][0]

        #continuous_action[0], continuous_action[1], original_actions, noises = self.choose_continuous_action(state, trader_type, type, count, market_number)
        # 根据此，选择合适的离散action
        if random.random() > self.epsilon:
            discrete_action = np.argmax(self.get_qs(state, continuous_action))  # 选择Q值最大的index
        else:
            discrete_action = np.random.randint(0, self.action_dim[0])  # 随机选择 0 or 1
        # action = {k, xk} xk 是所有的参数
        action[0] = discrete_action
        action[1] = continuous_action[0]
        action[2] = continuous_action[1]
        #return action, original_actions, noises, 0  # 二维
        return action, 0, 0, 0  # 二维

        '''
        # todo 此处拼接new_state
        continuous_state = np.copy(state)
        continuous_state = np.reshape(continuous_state, (self.state_dim, ))
        new_continuous_state = []
        for i in continuous_state:
            new_continuous_state.append(i)  #todo 需要将离散动作映射到[0,1]
        new_continuous_state.append(discrete_action)
        #选择连续动作
        continuous_action, original_action, noise = self.choose_continuous_action0(new_continuous_state, trader_type, type, count, market_number)  # todo 此处入参state是 7维
        action[0] = discrete_action
        action[1] = continuous_action
        '''

    def choose_continuous_action0(self, state, trader_type, type, count, market_number):
        a = self.act_best_response(np.reshape(state, (1, 1, self.state_dim + 1)))   # type (1,1,7) => (1,1,1)
        a_original = a
        a = np.random.normal(a, self.var)  # normal加噪音
        noise = a - a_original
        if self._rl_memory.size() >= self._rl_memory.buffer_size:
            self.var *= 0.9999
        a = np.clip(a, 0, 1)
        return a[0][0][0], a_original[0][0][0], noise[0][0][0]

    def choose_continuous_action(self, state, trader_type, type, count : int, market_number):  # count 代表第多少个周期
        a = self.act_best_response(np.reshape(state, (1, 1, self.state_dim)))  # type (1,1,2)
        a_original = copy.copy(a)
        noise = [0] * self.action_dim[1]
        a_new = [0] * self.action_dim[1]
        #a = np.random.normal(a, self.var)  # normal加噪音
        #noise = a - a_original
        #if self._rl_memory.size() >= self._rl_memory.buffer_size:
            #  self.num += 1
            #  if self.num >= 3:
        #    self.var *= 0.9999
            # self.num = 0
        # 此处分析如何加noise
        left_episode = self.episodes - count
        k = left_episode / self.episodes   # k 从1 - 0
        #k = 1
        if trader_type == 'S':
            for i in range(self.action_dim[1]):  # 对每个维度的action判断
                temp_a = a[0][0][i]
                if temp_a < type:  # 原始出价不在合理范围内
                    if random.random() < 0.3:  # 一定的概率不加噪音，就探索不合理的action
                        noise[i] = 0
                        a_new[i] = temp_a
                    else:  # 一定的概率还是要探索合理的范围
                        noise[i] = np.random.uniform(type - temp_a, (type - temp_a) + (1 - type) * k)
                        a_new[i] = temp_a + noise[i]
                else:  # 原始出价在合理范围内
                    if random.random() < 0.8:  # 向type靠近
                        noise[i] = np.random.uniform(type - temp_a, (type - temp_a) * (1 - k))
                        a_new[i] = temp_a + noise[i]
                    else:
                        noise[i] = np.random.uniform(0, (1 - temp_a) * k)
                        a_new[i] = temp_a + noise[i]

        if trader_type == 'B':  # 对每个维度判断
            for j in range(self.action_dim[1]):
                temp_a = a[0][0][j]
                if temp_a > type:  # 原始出价不合理
                    if random.random() < 0.3:  # 探索不合理的action
                        noise[j] = 0
                        a_new[j] = temp_a
                    else:  # 探索合理的
                        noise[j] = np.random.uniform((type - temp_a) - type * k, type - temp_a)
                        a_new[j] = temp_a + noise[j]
                else:  # 原始出价合理[0, type]
                    if random.random() < 0.8:
                        noise[j] = np.random.uniform(0 + (type - temp_a) * (1 - k), type - temp_a)  # 向type靠近
                        a_new[j] = temp_a + noise[j]
                    else:
                        noise[j] = np.random.uniform(-temp_a * k, 0)
                        a_new[j] = noise[j] + temp_a

        return a_new[0], a_new[1], np.array(a_original[0][0]).tolist(), noise

        '''
                if random.random() > 0.25: # 加噪音，随机探索
            a_original = a
            a = np.random.normal(a, self.var)  # normal加噪音
            noise = a - a_original
            if self._rl_memory.size() >= self._rl_memory.buffer_size:
                self.var *= 0.9999

            a = np.clip(a, 0, 1)
            return a[0][0][0], a_original[0][0][0], noise[0][0][0]
        else: #不探索
            return a[0][0][0], a[0][0][0], 0
        '''


    def chooseAction1(self, state, trader_type, type, count):
        action = 0
        number = self._rl_memory.size()
        batch_size = self.batch_size
        if self._rl_memory.size() <= self.batch_size:
            sig = random.random()
            if sig < 0.25:
                action = None
            else:
                if trader_type == 'B':# buyer
                    action = random.uniform(0, type)
                else:# seller
                    action = random.uniform(type, 1)
            return action, action, 0
        else: # size > batch_size
            a = self.act_best_response(np.reshape(state, (1, 1, 6)))
            origin_a = a
            # add noise, new action
            a = np.random.normal(a, self.var)
            if self._rl_memory.size() >= self._rl_memory.buffer_size:
                self.var *= 0.99999
            a = np.reshape(a, (1, ))
            if a < 0 or a > 1:# not enter market
                return None, None, self.var
            else:# enter market
                return a[0], origin_a[0][0][0], self.var

    def action(self, state, trader_type, type):
        a = self.act_best_response(np.reshape(state, (1, 1, 6)))
        a = np.reshape(a, (1, ))
        #a[0] = np.round(a[0], 2)
        return a[0]


    #此处是测试action 2维度，是否可以表示混合action
    def chooseAction2(self, state, trader_type, type, count):
        a = self.act_best_response(np.reshape(state, (1, 1, 6)))   # type (1,1,2)
        a = np.reshape(a, (2,))
        original_action = a
        if a[0] >= 0 and a[0] < 0.5: #不进入
            print("action out of range [0, 1],not enter the market")
            f = open('no_market.txt', 'a+')
            print("=====Agent %d======" % self.id, "trader_type:", self.trader_type, '\n',
                  'type:', self.type, file=f)
            return a, a, 0
        else: #进入市场
            noise_t = max((1 - 0.00001 * count), 0) * OU.function(a[1], 0.0, 0.2, 0.3)
            a[1] = a[1] + noise_t
            a[1] = np.clip(a[1], 0, 1)
            # action[0] = np.round(action[0], 2)
            return a, original_action, noise_t[0]

        '''
        a = np.random.normal(a, self.var)# normal加噪音
        noise = a - a_original
        a = np.clip(a, 0, 1) # 控制在[0, 1]之间.
        if self._rl_memory.size() >= self._rl_memory.buffer_size:
            self.num += 1
            if self.num >= 4:
                self.var *= 0.9999
                self.num = 0
        return a[0][0][0], a_original[0][0][0], noise[0][0][0]
        '''





