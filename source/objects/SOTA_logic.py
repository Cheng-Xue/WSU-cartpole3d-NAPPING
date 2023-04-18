#!/usr/bin/env python3
# ************************************************************************************************ #
# **                                                                                            ** #
# **    AIQ-SAIL-ON SOTA Core Logic                                                             ** #
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

import configparser
import datetime
import copy
import json
import logging
import logging.handlers
import optparse
import psutil
import pytz
import queue
import random
import sys
import threading
import time
import uuid

from . import rabbitmq
from . import objects
from .TA2_logic import TA2Logic


class SotaLogic(TA2Logic):
    def __init__(self):
        super().__init__()
        options = self._get_command_line_options()
        config_file = options.config
        printout = options.printout
        debug = options.debug
        fulldebug = options.fulldebug
        logfile = options.logfile
        no_testing = options.no_testing
        just_one_trial = options.just_one_trial
        ignore_secret = options.ignore_secret
        loop_sota = options.loop_sota
        self.agent_name = 'SOTA'

        log_level = logging.WARNING
        # Define a global log object in case we need to set options for fulldebug.
        global_log = logging.getLogger()
        if printout:
            log_level = logging.INFO
        if debug:
            log_level = logging.DEBUG
        if fulldebug:
            global_log.setLevel(log_level)
        # Define a logger for this class that outputs with the name.
        self.log = logging.getLogger(__name__).getChild(self.agent_name)
        self.log.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if logfile is not None:
            fh = logging.handlers.TimedRotatingFileHandler(logfile,
                                                           when='midnight',
                                                           backupCount=30)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            # Only set the handler for one of the two, otherwise you will see output doubled for
            # every message you log.
            # fulldebug already set in super __init__().
            if not fulldebug:
                self.log.addHandler(fh)
        if printout:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            # Only set the handler for one of the two, otherwise you will see output doubled for
            # every message you log.
            # fulldebug already set in super __init__().
            if not fulldebug:
                self.log.addHandler(ch)

        self._keyboard_ended = False
        self._loop_sota = loop_sota
        if self._loop_sota:
            self._no_testing = False
            self._just_one_trial = False

        self._model_filename_pat = 'model/model.SOTA.{}.{}.file'.format(self._sail_on_domain, '{}')
        self.end_training_early = False
        self.end_experiment_early = False

        # The rest of the init happens how we want in TA2Logic.
        return

    def _get_command_line_options(self):
        parser = optparse.OptionParser(usage="usage: %prog [options]")
        parser = self._add_command_line_options(parser)
        parser.add_option("--loop-sota",
                          dest="loop_sota",
                          action="store_true",
                          help='Causes the program to loop in Mode #1.',
                          default=False)

        (options, args) = parser.parse_args()
        if options.fulldebug:
            options.debug = True
        return options

    def _run_sail_on(self):
        try:
            if self._amqp is not None:
                self._amqp.stop()
                del self._amqp
            self._amqp = rabbitmq.Connection(agent_name=self._agent_name,
                                             amqp_user=self._amqp_user,
                                             amqp_pass=self._amqp_pass,
                                             amqp_host=self._amqp_host,
                                             amqp_port=self._amqp_port,
                                             amqp_vhost=self._amqp_vhost,
                                             amqp_ssl=self._amqp_ssl)

            # If we are looping, delete the current experiment_secret.
            if self._loop_sota:
                self._experiment_secret = None
            else:
                # We are NOT looping, so make sure the loop does not continue.
                self._keyboard_ended = True

            self._amqp.run()

            # Initialize the SOTA agent.
            self.initialize_sota_agent()

            # Build the model.
            model = objects.Model(model_name=self._model_name,
                                  organization=self._organization,
                                  aiq_username=self._aiq_username,
                                  aiq_secret=self._aiq_secret)

            # Try to register as SOTA for a SAIL-ON experiment.
            if self._experiment_secret is None or self._no_testing:
                my_experiment = self._amqp.register_as_sota(model=model,
                                                            domain=self._sail_on_domain,
                                                            no_testing=self._no_testing,
                                                            seed=self._seed,
                                                            description=self._description)
                # self.log.info(str(my_experiment))
                if not isinstance(my_experiment, objects.AiqObject):
                    if isinstance(my_experiment, objects.CasasObject):
                        self.log.error(str(my_experiment.get_json()))
                    else:
                        self.log.error(my_experiment)
                    raise objects.AiqExperimentException('Something is not right.')

                # We might have to wait until there is an experiment that we need to run.
                while isinstance(my_experiment, objects.SotaIdle):
                    # Wait 30 seconds before asking again if there are any experiments we can run.
                    self._amqp.sleep(duration=30.0)
                    my_experiment = self._amqp.register_as_sota(model=model,
                                                                domain=self._sail_on_domain,
                                                                no_testing=self._no_testing,
                                                                seed=self._seed,
                                                                description=self._description)
                    # self.log.debug(str(my_experiment))

                self.log.info('There is an experiment ready!')
                # self.log.debug(str(my_experiment))

                # Store the experiment_secret locally.
                self._experiment_secret = my_experiment.experiment_secret
                if self._experiment_secret is not None:
                    # Now we can set the model filename.
                    self._set_model_filename()
                    # Set the experiment_secret in the config object.
                    self._config.set('sail-on', 'experiment_secret', self._experiment_secret)
                    # Write out the config with the new experiment_secret value.
                    self._write_config_file()

                    # Run the SAIL-ON experiment!
                    self._run_sail_on_experiment()
            else:
                self._set_model_filename()
                # Here we don't need to start a new experiment, just register to work on 1 or
                # many trials for the given experiment.
                my_experiment = self._amqp.start_work_on_sota_trials(
                    model=model,
                    experiment_secret=self._experiment_secret,
                    just_one_trial=self._just_one_trial,
                    domain=self._sail_on_domain)
                if isinstance(my_experiment, objects.CasasResponse):
                    if my_experiment.status == 'error':
                        for casas_error in my_experiment.error_list:
                            self.log.error(casas_error.message)
                            self.log.error(str(casas_error.error_dict))
                else:
                    # We have our response.
                    # Start working on trials until TA1 tells us the experiment is done, or at least
                    # we are done with what we requested.
                    self._run_jump_to_sail_on_testing()

        except KeyboardInterrupt:
            self._stop()
            self._keyboard_ended = True
        except objects.AiqExperimentException as e:
            self.log.error(e.value)
            self._stop()
        except objects.AiqDataException as e:
            self.log.error(e.value)
            self._stop()
            # If we have a data exception, we don't want to try again.
            self._keyboard_ended = True
        except objects.CasasRabbitMQException as e:
            self.log.error(e.value)
            self._stop()

        # Stop self._amqp now that the experiment is done.
        self._stop()
        return

    def run(self):
        while not self._keyboard_ended:
            self._run_sail_on()
        return

    def initialize_sota_agent(self):
        raise ValueError('initialize_sota_agent() not defined.')

    def experiment_start(self):
        raise ValueError('experiment_start() not defined.')

    def training_start(self):
        raise ValueError('training_start() not defined.')

    def training_episode_start(self, episode_number: int):
        raise ValueError('training_episode_start() not defined.')

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
        raise ValueError('training_instance() not defined.')

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
        raise ValueError('training_performance() not defined.')

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
        raise ValueError('training_episode_end() not defined.')

    def training_end(self):
        """This function is called when we have completed the training episodes.
        """
        raise ValueError('training_end() not defined.')

    def train_model(self):
        """Train your model here if needed.  If you don't need to train, just leave the function
        empty.  After this completes, the logic calls save_model() and reset_model() as needed
        throughout the rest of the experiment.
        """
        raise ValueError('train_model() not defined.')

    def save_model(self, filename: str):
        """Save the current trained model to disk so agent can reset to it at the start of
        each trial.

        Parameters
        ----------
        filename : str
            The filename to save the model to.
        """
        raise ValueError('save_model() not defined.')

    def reset_model(self, filename: str):
        """Reset model to state just after training.

        Parameters
        ----------
        filename : str
            The filename where the model was stored.
        """
        raise ValueError('reset_model() not defined.')

    def trial_start(self, trial_number: int, novelty_description: dict):
        """This is called at the start of a trial with the current 0-based number.

       Parameters
       ----------
       trial_number : int
           This is the 0-based trial number in the novelty group.
       novelty_description : dict
           A dictionary that will have a description of the trial's novelty.
       """
        raise ValueError('trial_start() not defined.')

    def testing_start(self):
        """This is called after a trial has started but before we begin going through the
        episodes.
        """
        raise ValueError('testing_start() not defined.')

    def testing_episode_start(self, episode_number: int):
        """This is called at the start of each testing episode in a trial, you are provided the
        0-based episode number.

        Parameters
        ----------
        episode_number : int
            This is the 0-based episode number in the current trial.
        """
        raise ValueError('testing_episode_start() not defined.')

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
        raise ValueError('testing_instance() not defined.')

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
        raise ValueError('testing_performance() not defined.')

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
        raise ValueError('testing_episode_end() not defined.')

    def testing_end(self):
        """This is called after the last episode of a trial has completed, before trial_end().
        """
        raise ValueError('testing_end() not defined.')

    def trial_end(self):
        """This is called at the end of each trial.
        """
        raise ValueError('trial_end() not defined.')

    def experiment_end(self):
        """This is called when the experiment is done.
        """
        raise ValueError('experiment_end() not defined.')

