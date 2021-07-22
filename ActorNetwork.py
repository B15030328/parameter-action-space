import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, normalization, concatenate
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 25
HIDDEN2_UNITS = 20

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.state_size = state_size
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, 1, action_size[0] + action_size[1]])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        #self.sess.run(tf.global_variables_initializer())

    #其中self.model.output对self.weights求导，
    # self.action_gradient对model.trainable_weights中的每个元素的求导加权重
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads,
        })
    '''
    def train(self, sess, states, action_grads):
        sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })
    '''

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network0(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[1, state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Acceleration = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Brake = Dense(1, activation='sigmoid', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        V = merge([Steering, Acceleration,Brake],mode='concat')
        model = Model(input=S, output=V)
        return model, model.trainable_weights, S

    def sum_loss(self, y_true, y_pred, e=0.1):  # 手动构造损失函数
        return K.sum(y_true, axis=1) * -1

    '''
    input：state 
    output: outputs values for each of the four discrete actions as well as
    six continuous parameters.
    '''
    def create_actor_network(self, state_dim, action_dim):
        S = Input(shape=[1, state_dim])
        bn = normalization.BatchNormalization()(S)
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(bn)  # 1
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)   # 2
        x = Dense(action_dim[0] + action_dim[1], activation='sigmoid')(h1)   # 输出是2个维度
        # 构建第一个输出,discrete_action
        #x = Model(inputs=S, outputs=x)
        #h2 = Dense(HIDDEN1_UNITS, activation='relu')(h1)
        # 构建第二个输出，continuous_action
        #y = Dense(action_dim[1], activation='sigmoid')(h1)
        #y = Model(inputs=S, outputs=y)
        #z = x + y
        model = Model(inputs=S, outputs=x)
        #combined = concatenate([x.output, y.output])
        #out = Dense(action_dim, activation='sigmoid')(h1)
        #model = Model(inputs=S, outputs=out)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, model.trainable_weights, S

    def create_actor_network1(self, state_dim, action_dim):
        S = Input(shape=[1, state_dim])
        bn = normalization.BatchNormalization()(S)
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(bn)  # 1
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)   # 2
        x = Dense(action_dim[1], activation='sigmoid')(h1)   # 输出是2个维度 params
        # 构建第一个输出,discrete_action
        #x = Model(inputs=S, outputs=x)
        #h2 = Dense(HIDDEN1_UNITS, activation='relu')(h1)
        # 构建第二个输出，continuous_action
        y = Dense(action_dim[0], activation='linear')(x)  # 输出的是离散动作的Q值
        #y = Dense(action_dim[1], activation='sigmoid')(h1)
        #y = Model(inputs=S, outputs=y)
        model = Model(inputs=S, outputs=[y, x])
        #combined = concatenate([x.output, y.output])
        #out = Dense(action_dim, activation='sigmoid')(h1)
        #model = Model(inputs=S, outputs=out)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, model.trainable_weights, S

