#!/usr/bin/env python3
# ************************************************************************************************ #
# **                                                                                            ** #
# **    AIQ-SAIL-ON TA2 Agent Example                                                           ** #
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

import copy
import optparse
import queue
import random
import threading
import time

from objects.TA2_logic import TA2Logic
from stable_baselines3 import PPO, DQN
import numpy as np
import math

from importlib.util import spec_from_file_location, module_from_spec


class ThreadedProcessingExample(threading.Thread):
    def __init__(self, processing_object: list, response_queue: queue.Queue):
        threading.Thread.__init__(self)
        self.processing_object = processing_object
        self.response_queue = response_queue
        self.is_done = False
        return

    def run(self):
        """All work tasks should happen or be called from within this function.
        """
        is_novel = False
        message = ''

        # Do some fake work here.
        for work in self.processing_object:
            sum = 0
            for i in range(work):
                sum += i
            message += 'Did sum of {}. '.format(work)

        self.response_queue.put((is_novel, message))
        return

    def stop(self):
        self.is_done = True
        return


class TA2Agent(TA2Logic):
    def __init__(self):
        super().__init__()

        self.possible_answers = list()
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
        self.model = PPO.load("./PPO_cartpole3d_allwalls")
        self.action_dict = {0: 'nothing', 1: 'right', 2: 'left',  3: 'forward', 4: 'backward'}
        self.log.debug('Command line custom value is: {}'.format(my_custom_value))
        self.performance_recording = {}
        self.trial_number = 0

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
        if feature_label not in self.possible_answers:
            self.possible_answers.append(copy.deepcopy(feature_label))

        label_prediction = random.choice(self.possible_answers)

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
        
        spec = spec_from_file_location("NAPPING", "NAPPING.pyc")
        NAPPING = module_from_spec(spec)
        spec.loader.exec_module(NAPPING)
        self.AdaptationPrinciple = NAPPING.AdaptationPrinciple
        self.napping = self.AdaptationPrinciple(action_space=[0, 1, 2, 3, 4], model=self.model,
                                           eval=self.AdaptationPrinciple.eval)

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

        # Throw something together to create work.
        # work_list = list([5000])
        # for i in range(3 + episode_number):
        #     work_list.append(50000)
        # response_queue = queue.Queue()
        # response = None
        # # Initialize the example of doing work safely outside the main thread.
        # # Remember, in Python all objects beyond int, float, bool, and str are passed
        # # by reference.
        # threaded_work = ThreadedProcessingExample(processing_object=work_list,
        #                                           response_queue=response_queue)
        # # Start the work in a separate thread.
        # threaded_work.start()
        #
        # while response is None:
        #     try:
        #         # Try to get the response from the queue for 5 seconds before we let the AMQP
        #         # network event loop do any required work (such as sending heartbeats) for
        #         # 0.5 seconds.  By having the get(block=True) we ensure that there is basically
        #         # no wait for the result once it is put in the queue.
        #         response = response_queue.get(block=True, timeout=5)
        #     except queue.Empty:
        #         # Process any amqp events for 0.5 seconds before we try to get the results again.
        #         self.process_amqp_events()
        #
        # # Safely end and clean up the threaded work object.
        # threaded_work.stop()
        # threaded_work.join()
        #
        # self.log.info('message from threaded work: {}'.format(response[1]))
        # self.log.warning('Please remove this sample threaded work object from '
        #                  'testing_episode_start() before running actual experiments.')
        self.np_state = None
        self.pre_np_state = None
        self.old_action = None
        self.action = None
        self.last_zv = None
        self.current_zv = None
        self.novelty_indicator = False
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
        # self.log.debug('Testing Instance: feature_vector={}, novelty_indicator={}'.format(
        #     feature_vector, novelty_indicator))
        # Return dummy random choices, but should be determined by trained model
        # label_prediction = random.choice(self.possible_answers)
        # return only 1 blocks that are nearest to the cart

        np_state = self.AdaptationPrinciple.process_featvec(feature_vector)
        v = np_state[10:12]
        #roll_x, pitch_y, yaw_z = euler_from_quaternion(x,y,z,w)
        zv = v

        past_result = self.trial_result
        if len(past_result) != 0:
            if np.max(past_result[-5:]) <= 0.8 and len(past_result) >= 10:
                self.novelty_indicator = True

        if self.novelty_indicator and self.last_zv is not None:
            self.napping.update(self.pre_np_state, np_state, self.old_action, self.action,
                                last_zv=self.last_zv, current_zv=zv)

        if not self.novelty_indicator:
            self.action = self.model.predict(np_state)[0] 
        else:
            #self.action = self.model.predict(np_state)[0]
            self.old_action, self.action = self.napping.predict(np_state)
            self.pre_np_state = np_state
            self.last_zv = zv
        label_prediction = {'action': self.action_dict[self.action.item() if not isinstance(self.action, int) else self.action]}

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
        self.trial_result.append(performance)
        self.log.info(f'# of adaptation principles: {len(self.napping.adaptation_principle_mapping)}')
        novelty_probability = 1 if self.novelty_indicator else 0
        novelty_threshold = 0.5
        novelty = random.choice(list(range(4)))
        novelty_characterization = dict()

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
        self.trial_number += 1

        return

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        self.log.info('Experiment End')
        return


if __name__ == "__main__":
    agent = TA2Agent()
    agent.run()
