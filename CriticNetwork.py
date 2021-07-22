import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation,normalization, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 20  #150
HIDDEN2_UNITS = 40  #300

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.output = self.model.output
        #self.sess.run(tf.global_variables_initializer())


    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    '''
    def gradients(self, sess, states, actions):
        K.set_session(sess)
        return sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]
    '''
    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network1(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim],name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

    '''
    state+离散+连续作为input
    '''
    def create_critic_network(self, state_dim, action_dim):
        S = Input(shape=[1, state_dim], name='input_s')
        bn_s = normalization.BatchNormalization()(S)
        A = Input(shape=[1, action_dim[0] + action_dim[1]], name='input_dis_continuous')
        bn_a = normalization.BatchNormalization()(A)
        #A_continous = Input(shape=[1, action_dim[1]], name='input_a_cont')
        #bn_b = normalization.BatchNormalization()(A_continous)
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(bn_s)
        a1 = Dense(HIDDEN2_UNITS, activation='relu')(bn_a)
        #b1 = Dense(HIDDEN2_UNITS, activation='relu')(bn_b)

        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        #A = concatenate([a1, b1])
        h2 = concatenate([h1, a1])  # 连接
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        h4 = Dense(HIDDEN1_UNITS, activation='relu')(h3)
        V = Dense(action_dim[0] + action_dim[1], activation='linear')(h4)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S   # todo  返回的是什么动作