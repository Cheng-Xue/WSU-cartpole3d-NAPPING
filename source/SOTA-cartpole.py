#!/usr/bin/env python3
# ************************************************************************************************ #
# **                                                                                            ** #
# **    AIQ-SAIL-ON SOTA Agent Example                                                          ** #
# **                                                                                            ** #
# **        Brian L Thomas, 2020                                                                ** #
# **                                                                                            ** #
# **  Tools by the AI Lab - Artificial Intelligence Quotient (AIQ) in the School of Electrical  ** #
# **  Engineering and Computer Science at Washington State University.                          ** #
# **                                                                                            ** #
# **  Copyright Washington State University, 2020                                               ** #
# **  Copyright Brian L. Thomas, 2020                                                           ** #
# **                                                                                            ** #
# **  All rights reserved                                                                       ** #
# **  Modification, distribution, and sale of this work is prohibited without permission from   ** #
# **  Washington State University.                                                              ** #
# **                                                                                            ** #
# **  Contact: Brian L. Thomas (bthomas1@wsu.edu)                                               ** #
# **  Contact: Larry Holder (holder@wsu.edu)                                                    ** #
# **  Contact: Diane J. Cook (djcook@wsu.edu)                                                   ** #
# ************************************************************************************************ #


from sota_util.cartpole.cartpole_agent import Simple
import optparse
import random
import time
from importlib.util import spec_from_file_location, module_from_spec

from objects.TA2_logic import TA2Logic
import numpy as np
import math

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

class TA2Agent(TA2Logic):
    def __init__(self):
        super().__init__()

        # This variable can be set to true and the system will attempt to end training at the
        # completion of the current episode, or sooner if possible.
        self.end_training_early = True

        # This variable is checked only during the evaluation phase.  If set to True the system
        # will attempt to cleanly end the experiment at the conclusion of the current episode,
        # or sooner if possible.
        self.end_experiment_early = False

        # If you need values from the command line, you can get values from your custom options
        # here.  Set custom options in the _add_ta2_command_line_options() function.
        options = self._get_command_line_options()
        my_custom_value = options.custom_value
        self.log.debug('Command line custom value is: {}'.format(my_custom_value))
        self.action_dict = {0: 'nothing', 1: 'right', 2: 'left',  3: 'forward', 4: 'backward'}
        self.action_space = ['nothing', 'right', 'left', 'forward', 'backward']
        self.performance_recording = {}
        self.trial_number = 0
        self.agent = None

        return

    def _add_ta2_command_line_options(self, parser: optparse.OptionParser):
        """If you do not want to use this function, you can remove it from TA2.py to clean up
        your code.  This is already defined in the parent class.

        This function allows you to easily add custom arguments to the command line parser.  To
        see what is already defined, please see the _add_command_line_options() function in the
        parent class found in options/TA2_logic.py.

        Parameters
        ----------
        parser : optparse.OptionParser
            This is the command line parser object, you can add custom entries to it here.

        Returns
        -------
        optparse.OptionParser
            The parser object that you have added additional options to.
        """
        parser.add_option("--custom-value",
                          dest="custom_value",
                          help="Example for adding custom options to the command line parser.",
                          default="HelloWorld!")
        return parser

    def experiment_start(self):
        """This function is called when this TA2 has connected to a TA1 and is ready to begin
        the experiment.
        """
        self.log.info('Experiment Start')
        self.agent = Simple()
        return

    def training_start(self):
        """This function is called when we are about to begin training on episodes of data in
        your chosen domain.
        """
        self.log.info('Training Start')
        return

    def training_episode_start(self, episode_number: int):
        """This function is called at the start of each training episode, with the current episode
        number (0-based) that you are about to begin.

        Parameters
        ----------
        episode_number : int
            This identifies the 0-based episode number you are about to begin training on.
        """
        self.log.info('Training Episode Start: #{}'.format(episode_number))
        return

    def training_instance(self, feature_vector: dict, feature_label: dict) -> dict:
        """Process a training

        Parameters
        ----------
        feature_vector : dict
            The dictionary of the feature vector.  Domain specific feature vector formats are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        feature_label : dict
            The dictionary of the label for this feature vector.  Domain specific feature labels
            are defined on the github (https://github.com/holderlb/WSU-SAILON-NG). This will always
            be in the format of {'action': label}.  Some domains that do not need an 'oracle' label
            on training data will receive a valid action chosen at random.

        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """
        self.log.debug('Training Instance: feature_vector={}  feature_label={}'.format(
            feature_vector, feature_label))

        label_prediction = self.agent.predict(feature_vector)

        return label_prediction

    def training_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        self.log.debug('Training Performance: {}'.format(performance))
        return

    def training_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        """Provides the final performance on the training episode and indicates that the training
        episode has ended.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.

        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
        self.log.info('Training Episode End: performance={}'.format(performance))

        novelty_probability = random.random()
        novelty_threshold = 0.8
        novelty = 0
        novelty_characterization = dict()

        return novelty_probability, novelty_threshold, novelty, novelty_characterization

    def training_end(self):
        """This function is called when we have completed the training episodes.
        """
        self.log.info('Training End')
        return

    def train_model(self):
        """Train your model here if needed.  If you don't need to train, just leave the function
        empty.  After this completes, the logic calls save_model() and reset_model() as needed
        throughout the rest of the experiment.
        """
        self.log.info('Train the model here if needed.')

        # Simulate training the model by sleeping.
        self.log.info('Simulating training with a 5 second sleep.')

        time.sleep(5)
        self.agent = Simple()

        return

    def save_model(self, filename: str):
        """Saves the current model in memory to disk so it may be loaded back to memory again.

        Parameters
        ----------
        filename : str
            The filename to save the model to.
        """
        self.log.info('Save model to disk.')
        return

    def reset_model(self, filename: str):
        """Loads the model from disk to memory.

        Parameters
        ----------
        filename : str
            The filename where the model was stored.
        """
        self.log.info('Load model from disk.')
        del self.agent
        self.agent = Simple()
        spec = spec_from_file_location("NAPPING", "NAPPING.pyc")
        NAPPING = module_from_spec(spec)
        spec.loader.exec_module(NAPPING)
        self.AdaptationPrinciple = NAPPING.AdaptationPrinciple
        self.napping = self.AdaptationPrinciple(action_space=self.action_space, model=self.agent,
                                           eval=self.AdaptationPrinciple.eval)

        return

    def trial_start(self, trial_number: int, novelty_description: dict):
        """This is called at the start of a trial with the current 0-based number.

        Parameters
        ----------
        trial_number : int
            This is the 0-based trial number in the novelty group.
        novelty_description : dict
            A dictionary that will have a description of the trial's novelty.
        """
        self.novelty_indicator = False

        self.log.info('Trial Start: #{}  novelty_desc: {}'.format(trial_number,
                                                                  str(novelty_description)))
        self.trial_result = []
        #
        # spec = spec_from_file_location("NAPPING", "NAPPING.pyc")
        # NAPPING = module_from_spec(spec)
        # spec.loader.exec_module(NAPPING)

        self.current_np_states = []
        self.next_np_states = []
        self.actions = []
        self.rewards = []
        self.if_done = []

        return

    def testing_start(self):
        """This is called after a trial has started but before we begin going through the
        episodes.
        """
        self.log.info('Testing Start')
        return

    def testing_episode_start(self, episode_number: int):
        """This is called at the start of each testing episode in a trial, you are provided the
        0-based episode number.

        Parameters
        ----------
        episode_number : int
            This is the 0-based episode number in the current trial.
        """
        self.log.info('Testing Episode Start: #{}'.format(episode_number))
        self.np_state = None
        self.pre_np_state = None
        self.old_action = None
        self.action = None
        self.last_zv = None
        self.current_zv = None
        self.tick = 0
        self.reward = 0
        return

    def testing_instance(self, feature_vector: dict, novelty_indicator: bool = None) -> dict:
        """Evaluate a testing instance.  Returns the predicted label or action, if you believe
        this episode is novel, and what novelty level you beleive it to be.

        Parameters
        ----------
        feature_vector : dict
            The dictionary containing the feature vector.  Domain specific feature vectors are
            defined on the github (https://github.com/holderlb/WSU-SAILON-NG).
        novelty_indicator : bool, optional
            An indicator about the "big red button".
                - True == novelty has been introduced.
                - False == novelty has not been introduced.
                - None == no information about novelty is being provided.

        Returns
        -------
        dict
            A dictionary of your label prediction of the format {'action': label}.  This is
                strictly enforced and the incorrect format will result in an exception being thrown.
        """
        self.log.debug('Testing Instance: feature_vector={}, novelty_indicator={}'.format(
            feature_vector, novelty_indicator))

        self.tick += 1
        self.reward = self.tick/200
        # if self.pre_np_state is not None:
        #     self.current_np_states.append(self.pre_np_state)
        #     self.next_np_states.append(feature_vector)
        #     self.actions.append(self.pre_action)
        #     self.rewards.append(self.reward)
        #     self.if_done.append(False)

        x_dot = feature_vector['pole']['x_velocity']
        y_dot = feature_vector['pole']['y_velocity']
        v = np.array([x_dot, y_dot])
        x = feature_vector['pole']['x_quaternion']
        y = feature_vector['pole']['y_quaternion']
        z = feature_vector['pole']['z_quaternion']
        w = feature_vector['pole']['w_quaternion']
        roll_x, pitch_y, yaw_z = euler_from_quaternion(x,y,z,w)
        zv = v + np.array([roll_x, pitch_y])

        past_result = self.trial_result
        if len(past_result) != 0:
            if np.max(past_result[-10:]) <= 0.7 and len(past_result) >= 15:
                self.novelty_indicator = True

        if self.novelty_indicator and self.last_zv is not None:
            self.napping.update(self.pre_np_state, None, self.old_action, self.action,
                                last_zv=self.last_zv, current_zv=zv)

        if not self.novelty_indicator:
            label_prediction = self.agent.predict(feature_vector)
        else:
            #label_prediction = self.agent.predict(feature_vector)
            self.old_action, self.action = self.napping.predict(feature_vector)
            if 'action' not in self.old_action:
                self.old_action = {'action':self.old_action}
            if 'action' not in self.action:
                self.action = {'action':self.action}
            self.pre_np_state = feature_vector
            self.pre_action = self.action
            self.last_zv = zv
            label_prediction = self.action

        # Use simple agent to predict

        return label_prediction

    def testing_performance(self, performance: float, feedback: dict = None):
        """Provides the current performance on training after each instance.

        Parameters
        ----------
        performance : float
            The normalized performance score.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.
        """
        self.log.debug('Testing Performance: {}'.format(performance))

        return

    def testing_episode_end(self, performance: float, feedback: dict = None) -> \
            (float, float, int, dict):
        """Provides the final performance on the testing episode.

        Parameters
        ----------
        performance : float
            The final normalized performance score of the episode.
        feedback : dict, optional
            A dictionary that may provide additional feedback on your prediction based on the
            budget set in the TA1. If there is no feedback, the object will be None.

        Returns
        -------
        float, float, int, dict
            A float of the probability of there being novelty.
            A float of the probability threshold for this to evaluate as novelty detected.
            Integer representing the predicted novelty level.
            A JSON-valid dict characterizing the novelty.
        """
        self.log.info('Testing Episode End: performance={}'.format(performance))
        self.log.info(f'# of adaptation principles: {len(self.napping.adaptation_principle_mapping)}')

        self.trial_result.append(performance)

        novelty_probability = 1 if self.novelty_indicator else 0

        novelty_threshold = 0.8
        novelty = random.choice(list(range(4)))
        novelty_characterization = dict()
        # if self.pre_np_state is not None:
        #     self.if_done[-1] = True
        #     for i in range(len(self.if_done)):
        #         obs = self.current_np_states[i]
        #         next_obs = self.next_np_states[i]
        #         action = self.actions[i]
        #         reward = self.rewards[i]
        #         done = self.if_done[i]
        #         self.model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
        #     if len(self.trial_result[5:]) % 3 == 0:
        #         tmp_path = "/tmp/sb3_log/"
        #         # set up logger
        #         self.model.train(gradient_steps=self.model.replay_buffer.pos // 5, batch_size=64)
        #         self.napping = self.AdaptationPrinciple(action_space=self.action_space, model=self.model,
        #                                             eval=self.AdaptationPrinciple.eval)
        return novelty_probability, novelty_threshold, novelty, novelty_characterization

    def testing_end(self):
        """This is called after the last episode of a trial has completed, before trial_end().
        """
        self.log.info('Testing End')
        return

    def trial_end(self):
        """This is called at the end of each trial.
        """
        self.log.info('Trial End')
        self.performance_recording[self.trial_number] = self.trial_result
        self.trial_number += 1
        return

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        self.log.info('Experiment End')
        for t in self.performance_recording:
            if len(self.performance_recording[t]) != 0:
                self.log.critical(
                f"trial id: {t} pre-novelty performance: {np.average(self.performance_recording[t][:1])} post-novelty performance: {np.average(self.performance_recording[t][1:])}")
        with open('all_records','w+') as f:
            for key, values in self.performance_recording.items():
                if len(values) != 0:
                    for i, v in enumerate(values):
            	        f.write(f"{key}, {i}, {v}\n")
        return


if __name__ == "__main__":
    agent = TA2Agent()
    agent.run()
