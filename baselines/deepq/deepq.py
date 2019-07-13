import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import os.path as osp

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path, sess, scope):
        save_variables(path, sess, scope)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=1e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          multiplayer=False,
          save_interval=None,
          save_path=None,
          callback=None,
          load_path=None,
          load_path_1=None,
          load_path_2=None,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """

    # This was all handled in not the most elegant way
    # Variables have a _1 or _2 appended to them to separate them
    # and a bunch of if statementss to have the _2 variables not do anything in single-player


    # when in multiplayer Space Invaders, need to not reward players for other player dying
    isSpaceInvaders = False
    if "SpaceInvaders" in str(env):
        isSpaceInvaders = True
    interval_count=0


    # put a limit on the amount of memory used, otherwise TensorFlow will consume nearly everything
    # this leaves 1 GB free on my computer, others may need to change it

    # Create all the functions necessary to train the model
    # Create two separate TensorFlow sessions
    sess_1 = tf.Session()
    if multiplayer:
        sess_2 = tf.Session()
    else:
        # set session 2 to None if it's not being used
        sess_2 = None
    set_global_seeds(seed)
    # specify the q functions as separate objects
    q_func_1 = build_q_func(network, **network_kwargs)
    if multiplayer:
        q_func_2 = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    # build everything for the first model
    # pass in the session and the "_1" suffix
    act_1, train_1, update_target_1, debug_1 = deepq.build_train(
        sess = sess_1,
        suffix = "_1",
        make_obs_ph=make_obs_ph,
        q_func=q_func_1,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        scope="deepq_1"
    )
    # a lot of if multiplayer statements duplicating these actions for a second network
    # pass in session 2 and "_2" instead
    if multiplayer:
        act_2, train_2, update_target_2, debug_2 = deepq.build_train(
            sess = sess_2,
            suffix = "_2",
            make_obs_ph=make_obs_ph,
            q_func=q_func_2,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise,
            scope="deepq_2"
        )

    # separate act_params for each wrapper
    act_params_1 = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func_1,
        'num_actions': env.action_space.n,
    }
    if multiplayer:
        act_params_2 = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func_2,
            'num_actions': env.action_space.n,
        }
    # make the act wrappers
    act_1 = ActWrapper(act_1, act_params_1)
    if multiplayer:
        act_2 = ActWrapper(act_2, act_params_2)
    # I need to return something if it's single-player
    else:
        act_2 = None

    # Create the replay buffer
    # separate replay buffers are required for each network
    # this is required for competitive because the replay buffers hold rewards
    # and player 2 has different rewards than player 1
    if prioritized_replay:
        replay_buffer_1 = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if multiplayer:
            replay_buffer_2 = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer_1 = ReplayBuffer(buffer_size)
        if multiplayer:
            replay_buffer_2 = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    # initialize both sessions
    U.initialize(sess_1)
    if multiplayer:
        U.initialize(sess_2)
    # the session was passed into these functions when they were created
    # the separate update functions work within the different sessions
    update_target_1()
    if multiplayer:
        update_target_2()

    # keep track of rewards for both models separately
    episode_rewards_1 = [0.0]
    saved_mean_reward_1 = None
    if multiplayer:
        episode_rewards_2 = [0.0]
        saved_mean_reward_2 = None
    obs = env.reset()
    reset = True

    # storing stuff in a temporary directory while it's working
    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td
        model_file_1 = os.path.join(td, "model_1")
        temp_file_1 = os.path.join(td, "temp_1")
        model_saved_1 = False
        if multiplayer:
            model_file_2 = os.path.join(td, "model_2")
            temp_file_2 = os.path.join(td, "temp_2")
            model_saved_2 = False

        if tf.train.latest_checkpoint(td) is not None:
            if multiplayer:
                # load both models if multiplayer is on
                load_variables(model_file_1, sess_1, "deepq_1")
                logger.log('Loaded model 1 from {}'.format(model_file_1))
                model_saved_1 = True
                load_variables(model_file_2, sess_2, "deepq_2")
                logger.log('Loaded model 2 from {}'.format(model_file_2))
                model_saved_2 = True
            # otherwise just load the first one
            else:
                load_variables(model_file_1, sess_1, "deepq_1")
                logger.log('Loaded model from {}'.format(model_file_1))
                model_saved_1 = True
        # I have separate load variables for single-player and multiplayer
        # this should be None if multiplayer is on
        elif load_path is not None:
            load_variables(load_path, sess_1, "deepq_1")
            logger.log('Loaded model from {}'.format(load_path))
        # load the separate models in for multiplayer
        # should load the variables into the appropriate sessions

        # my format may restrict things to working properly only when a Player 1 model is loaded into session 1, and same for Player 2
        # however, in practice, the models won't work properly otherwise
        elif multiplayer:
            if load_path_1 is not None:
                load_variables(load_path_1, sess_1, "deepq_1")
                logger.log('Loaded model 1 from {}'.format(load_path_1))
            if load_path_2 is not None:
                load_variables(load_path_2, sess_2, "deepq_2")
                logger.log('Loaded model 2 from {}'.format(load_path_2))


        if save_interval is not None and save_path is not None:
                if multiplayer:
                    if model_saved_1 and model_saved_2:
                        save_variables(temp_file_1, sess_1, "deepq_1")
                        save_variables(temp_file_2, sess_2, "deepq_2")
                        load_variables(model_file_1, sess_1, "deepq_1")
                        load_variables(model_file_2, sess_2, "deepq_2")
                    save_path_1 = osp.expanduser(save_path + "/stage" + str(interval_count) + "_player1")
                    save_path_2 = osp.expanduser(save_path + "/stage" + str(interval_count) + "_player2")
                    act_1.save(save_path_1, sess_1, "deepq_1")
                    act_2.save(save_path_2, sess_2, "deepq_2")
                    if model_saved_1 and model_saved_2:
                        load_variables(temp_file_1, sess_1, "deepq_1")
                        load_variables(temp_file_2, sess_2, "deepq_2")
                else:
                    if model_saved_1:
                        save_variables(temp_file_1, sess_1, "deepq_1")
                        load_variables(model_file_1, sess_1, "deepq_1")
                    save_path_solo = osp.expanduser(save_path + "/stage" + str(interval_count))
                    act_1.save(save_path_solo, sess_1, "deepq_1")
                    if model_saved_1:
                        load_variables(temp_file_1, sess_1, "deepq_1")
                interval_count = interval_count + 1
        
        # actual training starts here
        for t in range(total_timesteps):
            # use this for updating purposes
            actual_t = t+1
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            # receive model 1's action based on the model and observation
            action_1 = act_1(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action_1 = action_1
            # do the same for model 2 if in multiplayer
            if multiplayer:
                action_2 = act_2(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action_2 = action_2
            reset = False
            # apply actions to the environment
            if multiplayer:
                new_obs, rew_1, rew_2, done, _ = env.step(env_action_1, env_action_2)
            # apply single action if there isn't a second model
            else:
                new_obs, rew_1, rew_2, done, _ = env.step(env_action_1)

            # manual clipping for Space Invaders multiplayer
            if isSpaceInvaders and multiplayer:
                # don't reward a player when the other player dies
                # change the reward to 0

                # the only time either player will get rewarded 200 is when the other player dies
                if rew_1 == 200:
                    rew_1 = 0.0
                if rew_2 == 200:
                    rew_2 = 0.0
                # manually clip the rewards using the sign function
                rew_1 = np.sign(rew_1)
                rew_2 = np.sign(rew_2)
                combo_factor = 0.25
                rew_1_combo = rew_1 + combo_factor*rew_2
                rew_2_combo = rew_2 + combo_factor*rew_1
                rew_1 = rew_1_combo
                rew_2 = rew_2_combo

            # Store transition in the replay buffers
            replay_buffer_1.add(obs, action_1, rew_1, new_obs, float(done))
            if multiplayer:
                # pass reward_2 to the second player
                # this reward will vary based on the game
                replay_buffer_2.add(obs, action_2, rew_2, new_obs, float(done))
            obs = new_obs
            # separate rewards for each model
            episode_rewards_1[-1] += rew_1
            if multiplayer:
                episode_rewards_2[-1] += rew_2
            if done:
                obs = env.reset()
                episode_rewards_1.append(0.0)
                if multiplayer:
                    episode_rewards_2.append(0.0)
                reset = True
            if actual_t > learning_starts and actual_t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                # sample from the two replay buffers
                if prioritized_replay:
                    experience_1 = replay_buffer_1.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t_1, actions_1, rewards_1, obses_tp1_1, dones_1, weights_1, batch_idxes_1) = experience_1
                    # keep all the variables with separate names
                    if multiplayer:
                        experience_2 = replay_buffer_2.sample(batch_size, beta=beta_schedule.value(t))
                        (obses_t_2, actions_2, rewards_2, obses_tp1_2, dones_2, weights_2, batch_idxes_2) = experience_2
                # do the same if there's no prioritization
                else:
                    obses_t_1, actions_1, rewards_1, obses_tp1_1, dones_1 = replay_buffer_1.sample(batch_size)
                    weights_1, batch_idxes_1 = np.ones_like(rewards_1), None
                    if multiplayer:
                        obses_t_2, actions_2, rewards_2, obses_tp1_2, dones_2 = replay_buffer_2.sample(batch_size)
                        weights_2, batch_idxes_2 = np.ones_like(rewards_2), None
                # actually train the model based on the samples
                td_errors_1 = train_1(obses_t_1, actions_1, rewards_1, obses_tp1_1, dones_1, weights_1)
                if multiplayer:
                    td_errors_2 = train_2(obses_t_2, actions_2, rewards_2, obses_tp1_2, dones_2, weights_2)
                # give new priority weights to the observations
                if prioritized_replay:
                    new_priorities_1 = np.abs(td_errors_1) + prioritized_replay_eps
                    replay_buffer_1.update_priorities(batch_idxes_1, new_priorities_1)
                    if multiplayer:
                        new_priorities_2 = np.abs(td_errors_2) + prioritized_replay_eps
                        replay_buffer_2.update_priorities(batch_idxes_2, new_priorities_2)

            if actual_t > learning_starts and actual_t % target_network_update_freq == 0:
                # Update target networks periodically.
                update_target_1()
                if multiplayer:
                    update_target_2()


            # this section is for the purposes of logging stuff
            # calculate the average reward over the last 100 episodes
            mean_100ep_reward_1 = round(np.mean(episode_rewards_1[-101:-1]), 1)
            if multiplayer:
                mean_100ep_reward_2 = round(np.mean(episode_rewards_2[-101:-1]), 1)
            num_episodes = len(episode_rewards_1)
            # every given number of episodes log and print out the appropriate stuff
            if done and print_freq is not None and len(episode_rewards_1) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                # print out both rewards if multiplayer
                if multiplayer:
                    logger.record_tabular("mean 100 episode reward 1", mean_100ep_reward_1)
                    logger.record_tabular("mean 100 episode reward 2", mean_100ep_reward_2)
                else:
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward_1)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            # save best-performing version of each model
            # I've opted out of this for competitive multiplayer because it's difficult to determine what's "best"
            
            if (checkpoint_freq is not None and actual_t > learning_starts and
                    num_episodes > 100 and actual_t % checkpoint_freq == 0):
                # if there's a best reward, save it as the new best model
                if saved_mean_reward_1 is None or mean_100ep_reward_1 > saved_mean_reward_1:
                    if print_freq is not None:
                        if multiplayer:
                            logger.log("Saving model 1 due to mean reward increase: {} -> {}".format(saved_mean_reward_1, mean_100ep_reward_1))
                        else:
                            logger.log("Saving model due to mean reward increase: {} -> {}".format(saved_mean_reward_1, mean_100ep_reward_1))
                    save_variables(model_file_1, sess_1, "deepq_1")
                    model_saved_1 = True
                    saved_mean_reward_1 = mean_100ep_reward_1

                if multiplayer and (saved_mean_reward_2 is None or mean_100ep_reward_2 > saved_mean_reward_2):
                    if print_freq is not None:
                        logger.log("Saving model 2 due to mean reward increase: {} -> {}".format(saved_mean_reward_2, mean_100ep_reward_2))
                    save_variables(model_file_2, sess_2, "deepq_2")
                    model_saved_2 = True
                    saved_mean_reward_2 = mean_100ep_reward_2

            if save_interval is not None and actual_t % save_interval == 0 and save_path is not None:
                if multiplayer:
                    if model_saved_1 and model_saved_2:
                        save_variables(temp_file_1, sess_1, "deepq_1")
                        save_variables(temp_file_2, sess_2, "deepq_2")
                        load_variables(model_file_1, sess_1, "deepq_1")
                        load_variables(model_file_2, sess_2, "deepq_2")
                    save_path_1 = osp.expanduser(save_path + "/stage" + str(interval_count) + "_player1")
                    save_path_2 = osp.expanduser(save_path + "/stage" + str(interval_count) + "_player2")
                    act_1.save(save_path_1, sess_1, "deepq_1")
                    act_2.save(save_path_2, sess_2, "deepq_2")
                    if model_saved_1 and model_saved_2:
                        load_variables(temp_file_1, sess_1, "deepq_1")
                        load_variables(temp_file_2, sess_2, "deepq_2")
                else:
                    if model_saved_1:
                        save_variables(temp_file_1, sess_1, "deepq_1")
                        load_variables(model_file_1, sess_1, "deepq_1")
                    save_path_solo = osp.expanduser(save_path + "/stage" + str(interval_count))
                    act_1.save(save_path_solo, sess_1, "deepq_1")
                    if model_saved_1:
                        load_variables(temp_file_1, sess_1, "deepq_1")
                interval_count = interval_count + 1

        # restore models at the end to the best performers
        if model_saved_1:
            if print_freq is not None:
                logger.log("Restored model 1 with mean reward: {}".format(saved_mean_reward_1))
            load_variables(model_file_1, sess_1, "deepq_1")
        if multiplayer and model_saved_2:
            if print_freq is not None:
                logger.log("Restored model 2 with mean reward: {}".format(saved_mean_reward_2))
            load_variables(model_file_2, sess_2, "deepq_2")
    return act_1, act_2, sess_1, sess_2
