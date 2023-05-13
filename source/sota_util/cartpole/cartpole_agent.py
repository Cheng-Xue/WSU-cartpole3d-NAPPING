import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.utils import to_categorical

from keras.optimizers import Adam
import tensorflow as tf
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.models import Model

import numpy as np


class Simple:

    def __init__(self):
        self.agent = None
        self.observation_space = np.asarray(range(10))
        self.action_space = range(5)
        self.nb_actions = len(self.action_space)
        self.default_session = self.load_model()

        return None

    def load_model(self):

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!

        with tf.Graph().as_default() as g:
            config = tf.ConfigProto()
            sess = tf.Session(graph=g, config=config)
            with sess.as_default():
                model = Sequential()
                model.add(Flatten(input_shape=(1,) + self.observation_space.shape))
                model.add(Dense(512))
                model.add(Activation('relu'))
                model.add(Dense(512))
                model.add(Activation('relu'))
                model.add(Dense(self.nb_actions))
                model.add(Activation('linear'))
                print(model.summary())
                memory = SequentialMemory(limit=50000, window_length=1)
                policy = BoltzmannQPolicy()
                dqn = DQNAgent(model=model, nb_actions=self.nb_actions, memory=memory, nb_steps_warmup=10,
                               target_model_update=1e-2, policy=policy)
                dqn.compile(Adam(lr=1e-3), metrics=['mae'])
                dqn.load_weights('sota_util/cartpole/cartpole_weights.h5f')
                dqn.trainable_model._make_train_function()

                dqn.forward(np.random.normal(0,1,(10,)))
                dqn.target_model._make_predict_function()
                layer_output = model.layers[-3].output

                self.intermediate_model = Model(inputs=model.inputs, outputs=layer_output)
                self.intermediate_model.predict(np.random.normal(0,1,(1,1,10)))

        # Save to self
        self.agent = dqn
        return sess

    def predict(self, feature_vector, if_model_obs = False):
        obs = self.format_obs(feature_vector)
        with self.default_session.as_default():
            prediction = self.agent.forward(obs)
            action_response = self.format_response(prediction)
            if if_model_obs:
                model_obs = self.intermediate_model.predict(np.array(obs).astype(np.float32).reshape(1, 1, -1))
                return action_response, model_obs.squeeze(0)
        return action_response

    def format_obs(self, state):
        obs_state = list()
        # Add cart
        obs_state.append(state['cart']['x_position'])
        obs_state.append(state['cart']['y_position'])
        # obs_state.append(state['cart']['z_position'])
        obs_state.append(state['cart']['x_velocity'])
        obs_state.append(state['cart']['y_velocity'])
        # obs_state.append(state['cart']['z_velocity'])

        # Add pole
        obs_state.append(state['pole']['x_velocity'])
        obs_state.append(state['pole']['y_velocity'])
        obs_state.append(state['pole']['z_velocity'])

        '''
        # Add blocks max 4
        for i in list(range(4)):
            if i < len(state['blocks']):
                obs_state.append(state['blocks'][i]['x_position'])
                obs_state.append(state['blocks'][i]['y_position'])
                obs_state.append(state['blocks'][i]['z_position'])
                obs_state.append(state['blocks'][i]['x_velocity'])
                obs_state.append(state['blocks'][i]['y_velocity'])
                obs_state.append(state['blocks'][i]['z_velocity'])
            else:
                obs_state.append(-10.0)
                obs_state.append(-10.0)
                obs_state.append(-10.0)
                obs_state.append(-10.0)
                obs_state.append(-10.0)
                obs_state.append(-10.0)
        '''

        for ind, val in enumerate(obs_state):
            obs_state[ind] = round((val + 10.0) / 20.0, 2)

        # Add pole angle
        x, y, z = self.quaternion_to_euler(state['pole']['x_quaternion'], state['pole']['y_quaternion'],
                                           state['pole']['z_quaternion'], state['pole']['w_quaternion'])

        obs_state.append(round((x + 180) / 360, 2))
        obs_state.append(round((y + 180) / 360, 2))
        obs_state.append(round((z + 180) / 360, 2))

        return obs_state

    def format_response(self, prediction):
        if prediction == 0:
            prediction = {'action': 'nothing'}
        elif prediction == 1:
            prediction = {'action': 'right'}
        elif prediction == 2:
            prediction = {'action': 'left'}
        elif prediction == 3:
            prediction = {'action': 'forward'}
        elif prediction == 4:
            prediction = {'action': 'backward'}
        else:
            print('Bad action sent by sota cartpole')
            prediction = {'action': 'nothing'}

        return prediction

    def quaternion_to_euler(self, x, y, z, w):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2 > +1.0, +1.0, t2)
        # t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2 < -1.0, -1.0, t2)
        # t2 = -1.0 if t2 < -1.0 else t2
        Y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z
