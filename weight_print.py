from agent1 import Agent as Agent
import tensorflow as tf
import env as game
from keras.models import Sequential, Model
from keras.models import model_from_json
import numpy as np
if __name__ == '__main__':
    with tf.Session() as sess:
        env = game.Env('bal', 2, 0.6, 0.4, 0.1)
        agent = Agent(sess, env, 0.2, 'S', 1, 8, [2, 2])
        for i in range(1, 2):
            print("\033[32;1m agent %d \033[0m\n" % i)
            #agent.model_prediction.load_weights("q_prediction_" + str(i) + ".h5")
            agent.actor.model.load_weights("continuous_actor_" + str(i) + ".h5")
            print("q_prediction_model")
            '''
            for j in range(1, 4):
                test_model = Model(inputs=agent.model_prediction.inputs, outputs=agent.model_prediction.get_layer(index=j).output)
                #print(test_model.to_json())
                buy_state = [0.4, 0.1, 0.51, 0.5, 0.35, 0.45, 0.65, 0.8]
                ac = test_model.predict(np.reshape(buy_state, (1, 8)))
                print("action by layer_%d" % j, ac)
            '''
            print("continuous_actions:\n")
            for k in range(1, 5):
                test_model = Model(inputs=agent.actor.model.inputs, outputs=agent.actor.model.get_layer(index=k).output)
                buy_state = [0.4, 0.1, 0.51, 0.5, 0.35, 0.45, 0.65, 0.8]
                ac = test_model.predict(np.reshape(buy_state, (1, 1, 8)))
                print("\033[32;1m action by layer_%d \033[0m\n" %k, ac)

        #print(agent.actor.model.weights)
        '''
                for i in range(5):
            weight_Dense_1 = agent.actor.model.get_layer(index=i)
            print(weight_Dense_1.output)
            print(weight_Dense_1.get_weights())
        '''


