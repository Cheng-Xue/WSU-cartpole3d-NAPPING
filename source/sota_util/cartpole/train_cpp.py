# https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py
import time

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import WandbLogger

from env_generator.test_handler import TestHandler


# Wrapper class for keras-rl dqn learning
class CWrapper:

    def __init__(self, novelty, difficulty, seed=123):
        # Parameters
        self.domain = 'cartpole'
        self.novelty = novelty
        self.trial_novelty = novelty
        self.difficulty = difficulty
        self.seed = seed

        # Internal vars
        self.env = None
        self.path = "env_generator/envs/"
        self.use_img = False
        self.use_gui = False

        # REQUIRED:
        self.action_space = range(5)
        self.observation_space = np.asarray(range(10))

        return None

    # Converts the dictionary to a fixed list
    def transform(self, state):
        obs_state = list()

        # Add cart
        obs_state.append(state['cart']['x_position'])
        obs_state.append(state['cart']['y_position'])
        #obs_state.append(state['cart']['z_position'])
        obs_state.append(state['cart']['x_velocity'])
        obs_state.append(state['cart']['y_velocity'])
        #obs_state.append(state['cart']['z_velocity'])

        # Add pole
        obs_state.append(state['pole']['x_velocity'])
        obs_state.append(state['pole']['y_velocity'])
        obs_state.append(state['pole']['z_velocity'])

        # Add blocks max 4
        ''' Ignore blocks for now (too much for the network)
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

    def step(self, action):
        obs, reward, done, info = self.env.test.env.step(action)

        if done:
            self.env.test.env.close()

        return self.transform(obs), 1.0, done, info

    def reset(self):
        self.seed = int(time.time() * 1000) % 268435456

        # Create a new instance of the environment
        self.env = TestHandler(domain=self.domain, novelty=self.novelty, trial_novelty=self.novelty,
                               difficulty=self.difficulty, seed=self.seed, use_img=self.use_img, path=self.path,
                               use_gui=self.use_gui)

        return self.transform(self.env.get_feature_vector())

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


def main():
    # Bool for train or skip
    train = False

    # Setup test hander
    env = CWrapper(200, 'easy', seed=123)
    nb_actions = len(env.action_space)

    # Simple model
    model = Sequential()
    print(env.observation_space.shape)
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Setup keras-rl dqn
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Train or skip training
    if train:
        filepath = 'my_best_model.hdf5'
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        callbacks = [checkpoint]
        train_hist = dqn.fit(env, nb_steps=500000, visualize=False, verbose=1, callbacks=callbacks)
        dqn.save_weights('cartpole.h5f', overwrite=True)

    else:
        dqn.load_weights('cartpole.h5f')

    # Collect data samples for data plot
    r = dqn.test(env, nb_episodes=500, visualize=False)
    print(np.mean(np.asarray(r.history['episode_reward']) / 200.0))
    plt.hist(np.asarray(r.history['episode_reward']) / 200.0, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('performance')
    plt.ylabel('Amount')
    plt.title('Cartpole++ Sota')
    plt.xlim(0.0, 1.0)
    plt.grid(True)
    plt.show()

    return None


if __name__ == "__main__":
    main()
