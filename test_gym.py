import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import losses,optimizers,metrics
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from tqdm import tqdm

class DQNAgent:

    def __init__(self):

        # Replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Prediction Network (the main Model)
        self.model_prediction = create_model()

        # Target Network
        self.model_target = create_model()
        self.model_target.set_weights(self.model_prediction.get_weights())

        # Used to count when to update target network with prediction network's weights
        self.target_update_counter = 0

    # Adds step's data to a memory replay array
    # (current_state, action, reward, next_state, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries prediction network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model_prediction.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state, step):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model_prediction for current Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model_prediction.predict(current_states)

        # Get next_states from minibatch, then query NN model_target for target Q values
        # When using target network, query it, otherwise main network should be queried
        next_states = np.array([transition[3] for transition in minibatch])
        target_qs_list = self.model_target.predict(next_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_target_q = np.max(target_qs_list[index])
                new_q = reward + DISCOUNT * max_target_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

            # Fit on all samples as one batch, log only on terminal state
        self.model_prediction.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0,
                                  shuffle=False if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.model_target.set_weights(self.model_prediction.get_weights())
            self.target_update_counter = 0
env = gym.make("MountainCar-v0")
ACTION_SPACE_SIZE = env.action_space.n
def create_model():
        model = models.Sequential()
        model.add(Dense(16, input_shape=(env.observation_space.shape)))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(ACTION_SPACE_SIZE))
        model.add(Activation('linear'))

        print(model.summary())

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

EPISODES =1000
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
DISCOUNT = 0.99


# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001
ep_rewards = []

AGGREGATE_STATS_EVERY = 50
MIN_EPSILON = 0.001
if __name__ == '__main__':
    agent = DQNAgent()
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

    env.reset()

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

        #     # Update tensorboard step every episode
        #     agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, ACTION_SPACE_SIZE)

            next_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            #         if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            #             env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, next_state, done))
            agent.train(done, step)

            current_state = next_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

            # rewards recording
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min_reward)
            aggr_ep_rewards['max'].append(max_reward)

        #         agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        #         if min_reward >= MIN_REWARD:
        #             agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    recorder = {"epsode": [], "epsilon": []}

    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
    plt.legend(loc='upper left')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    agent.model_prediction.save('dqn_1.model')
























