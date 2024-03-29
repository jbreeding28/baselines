import sys
import re
import multiprocessing
import os
import os.path as osp
import gym
import gc
import cloudpickle
from collections import defaultdict
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
import random
import cv2
cv2.ocl.setUseOpenCL(False)
from baselines.common.atari_wrappers import *
import matplotlib
import csv

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines.common import plot_util as pu
from baselines import logger
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)
    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    # updated learning function, returning up to two models
    # I have to pass in the multiplayer argument
    model_1, model_2, sess_1, sess_2 = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        print_freq=10,
        multiplayer=args.multiplayer,
        **alg_kwargs
    )

    return model_1, model_2, sess_1, sess_2, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed
    play = args.play
    mode = args.mode
    multiplayer = args.multiplayer
    env_type, env_id = get_env_type(args)
    isSpaceInvaders = False
    if "SpaceInvaders" in args.env:
        isSpaceInvaders = True
    if env_type in {'atari', 'retro'}:
        # this should be the only algorithm I'll use
        if alg == 'deepq':
            # BEGIN MY CODE
            # clip reward when training
            # don't clip when playing to see actual score
            # add mode in as an environment parameter
            if play:
                # if I'm playing to see how well the network scores, I want to unclip rewards
                env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True, 'clip_rewards': False}, env_kwargs={'game_mode': mode})
            else:
                # otherwise, keep the basic reward used by the base algorithm
                if multiplayer and isSpaceInvaders:
                    # unclip rewards for space invaders multiplayer, I'll do it manually.
                    env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True, 'clip_rewards': False}, env_kwargs={'game_mode': mode})
                else:
                    env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True, 'clip_rewards': True}, env_kwargs={'game_mode': mode})
            # END MY CODE
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    # return two models, two sessions, and the environment type
    # if there's only a single model being trained, model_2 and sess_2 are None
    model_1, model_2, sess_1, sess_2, env = train(args, extra_args)
    # figure out if it's a multiplayer session
    # multiplayer stuff is left entirely up to the user
    multiplayer = args.multiplayer
    if args.save_path is not None and rank == 0:
        if multiplayer:
            save_path_1 = osp.expanduser(args.save_path + "_player1")
            # I needed the sessions to properly save the models here
            # the variables are specifically linked to the sessions
            model_1.save(save_path_1, sess_1)
            save_path_2 = osp.expanduser(args.save_path + "_player2")
            model_2.save(save_path_2, sess_2)
        else:
            save_path = osp.expanduser(args.save_path)
            model_1.save(save_path, sess_1)
    # play a number of games to evaluate the network
    if args.play:
        logger.log("Running trained model")
        obs = env.reset()


        action_path = osp.expanduser(args.log_path + "/actions.csv")
        actions_1 = list()
        actions_2 = list()
        rewards_1 = list()
        rewards_2 = list()
        done_list = list()
        lives_list = list()
        state_1 = model_1.initial_state if hasattr(model_1, 'initial_state') else None
        # copy what the first model is doing if there's multiple models
        if multiplayer:
            state_2 = model_2.initial_state if hasattr(model_2, 'initial_state') else None
        dones = np.zeros((1,))
        # BEGIN MY CODE
        # create a bunch of variables for holding various types of scores

        # episode reward is left over from the original but isn't really used
        episode_rew_1 = 0
        episode_rew_2 = 0

        # these variables hold the score of the current game and score across all games
        game_score = 0
        game_score_1 = 0
        game_score_2 = 0
        total_score = 0
        games_won = 0

        # keep hold of the highest score, initialize to zero
        max_score = 0
        # keep track of how many games are played
        game_count = 0
        game_steps = 0

        # get the number of games that are specified (default 10)
        # dependent on the user to make sure the number is valid
        num_games = args.num_games
        # boolean variable which tells the program whether or not to render the game
        render = args.render or args.render_fast
        # the default display time for one frame
        # due to the stacking of frames, only every fourth frame is displayed
        # and these games run at 60 fps
        # each fourth frame is rendered over the three missing frames
        frame_time = float(1/60)
        # get the render speed (default 3)
        render_speed = args.render_speed
        # constrain the speed to between 1x and 10x
        if render_speed <= 1:
            render_speed = 1
        elif render_speed >= 10:
            render_speed = 10
        # calculate the appropriate frame speed
        frame_time = frame_time/render_speed

        computer_view = args.computer_view

        # need special code to handle Pong
        # create variable to keep track of whether or not I'm playing Pong
        isPong = False
        if "Pong" in args.env:
            isPong = True
        isSpaceInvaders = False
        if "SpaceInvaders" in args.env:
            isSpaceInvaders = True

        # while loop carried over from base code
        # this will play games until so many have been played
        while True:
            # each loop through, get the current time at the start
            start_time = datetime.datetime.now()
            # get the appropriate action based on the observation of the environment
            if state_1 is not None:
                action_1, _, state_1, _ = model_1.step(obs,S=state_1, M=dones)
            # duplicate for a second model
            if multiplayer and state_2 is not None:
                action_2, _, state_2, _ = model_2.step(obs,S=state_2, M=dones)
                
            else:
                action_1, _, _, _ = model_1.step(obs)
                # have the second model take an action if appropriate
                if multiplayer:
                    action_2, _, _, _ = model_2.step(obs)
            # take a step forward in the environment, return new observation
            # return any reward and if the environment needs to be reset
            # pass in both actions if there are two models
            # reward in this case is the default reward
            # in competitive multiplayer, this is Player 1's reward
            if multiplayer:
                obs, rew_1, rew_2, done, _ = env.step(action_1, action_2)
            # otherwise, ignore the second 
            else:
                obs, rew_1, rew_2, done, _ = env.step(action_1)
            game_steps += 1
            # check to see if either player has died in Space Invaders multiplayer
            # this rewards a player when their opponent dies
            # remove this just to measure the score gained from destroying aliens
            if isSpaceInvaders and multiplayer:
                if rew_1 >= 200:
                    rew_1 = rew_1 - 200
                if rew_2 >= 200:
                    rew_2 = rew_2 - 200
            # get the number of lives remaining, which is relevant in certain games
            # in the multiplayer games I'll look at, the players should share a common life

            #append actions, rewards, and done (converted to 0 or 1) to the lists
            actions_1.append(action_1[0])
            rewards_1.append(rew_1)
            if multiplayer:
                actions_2.append(action_2[0])
                rewards_2.append(rew_2)
            done_list.append(int(done == True))

            lives = env.getLives()
            # append number of lives to the list
            lives_list.append(lives)
            # add reward from previous step to overall score
            episode_rew_1 += rew_1[0] if isinstance(env, VecEnv) else rew_1
            episode_rew_2 += rew_2[0] if isinstance(env, VecEnv) else rew_2
            # render the frame if the user wants it
            if render:
                if computer_view:
                    env.render(frame=obs)
                else:
                    env.render()
            done = done.any() if isinstance(done, np.ndarray) else done
            # done is true whenever a reset is necessary
            # occurs on death or game over
            if done:
                # Pong only uses done on game over, so make the episode reward the game score
                if isPong:
                    game_score = episode_rew_1
                # if it's not Pong, just do what I did before
                else:
                    game_score += episode_rew_1 + episode_rew_2
                game_score_1 += episode_rew_1
                game_score_2 += episode_rew_2
                if isPong:
                    total_score += episode_rew_1
                    games_won += (episode_rew_1 > 0)
                else:
                    total_score += episode_rew_1 + episode_rew_2
                # reset for next go around
                episode_rew_1 = 0
                episode_rew_2 = 0
                # reset the environment

                # on game over, this starts a new game
                # otherwise, continues the game but returns player to initial position
                obs = env.reset()
            # can make the games run at a given framerate
            if render and not args.render_fast:
                # just wait until it's time to push a new frame
                while (datetime.datetime.now() - start_time).total_seconds() < frame_time:
                    # pass means just wait and do nothing
                    pass
            # if there are no lives left, the game is over

            # use number of lives to differentiate between losing a life and game over
            # Pong doesn't use lives, and doesn't return "done" until game over
            # use the isPong variable to keep track of this
            if (lives == 0 and not isPong) or (done and isPong):
                # update highest score
                if game_score > max_score:
                    max_score = game_score
                # increment game counter
                game_count += 1
                # after the game is over, log the game number and score
                # game number is just so the person running this understands where they are

                # this method is based off of what I saw in the Deep Q code
                # record the data to the logger
                logger.record_tabular("game", game_count)
                if multiplayer:
                    logger.record_tabular("total score", game_score)
                    logger.record_tabular("player 1 score", game_score_1)
                    logger.record_tabular("player 2 score", game_score_2)
                else:
                    logger.record_tabular("score", game_score)
                logger.record_tabular("steps", game_steps)
                # then dump it to the log file and the terminal
                logger.dump_tabular()
                # game is over, reset the score
                game_score = 0
                game_score_1 = 0
                game_score_2 = 0
                game_steps = 0
            # print out average and max score when number of games is finished
            if game_count == num_games:
                print(" ")
                print('average score={}'.format(float(total_score/num_games)))
                if isPong:
                    print('win percentage={}'.format(float(games_won * 100/num_games)))
                else:
                    print('win percentage={}'.format(float(100)))
                # break out of this true loop
                break
        # END MY CODE
        
        # create file to save actions to
        # open it for writing
        action_file = open(action_path,'w+')
        with action_file as csv_scores:
            # filewriter object
            filewriter = csv.writer(csv_scores, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # if it's multiplayer, write values for both players
            if multiplayer:
                # column headers
                filewriter.writerow(['model1_actions', 'model1_rew', 'model2_actions', 'model2_rew', 'lives', 'done'])
                for j in range(0,len(actions_1)):
                    # iterate through arrays and write row by row
                    filewriter.writerow([actions_1[j],rewards_1[j],actions_2[j],rewards_2[j],lives_list[j],done_list[j]])
            else:
                # column headers for single-player
                filewriter.writerow(['model1_actions', 'model1_rew', 'lives', 'done'])
                for j in range(0,len(actions_1)):
                    # write row by row
                    filewriter.writerow([actions_1[j],rewards_1[j],lives_list[j],done_list[j]])

    env.close()

    if args.build_state_library:
        # based off of the library path I specify, specify file locations for the library and list of actions
        library_path = osp.expanduser(args.library_path + "/state_library")
        action_path = osp.expanduser(args.library_path + "/actions.csv")
        # empty list
        state_library = list()
        logger.log("Building state library")
        # initialize environment
        obs = env.reset()
        state_1 = model_1.initial_state if hasattr(model_1, 'initial_state') else None
        # copy what the first model is doing if there's multiple models
        if multiplayer:
            state_2 = model_2.initial_state if hasattr(model_2, 'initial_state') else None
        dones = np.zeros((1,))
        isPong = False
        # Pong needs to be handled differently
        if "Pong" in args.env:
            isPong = True
        model_1_actions = list()
        if multiplayer:
            model_2_actions = list()
        while True:
            state_library.append(StateWrapper(obs))
            if state_1 is not None:
                action_1, _, state_1, _ = model_1.step(obs,S=state_1, M=dones)
            # duplicate for a second model
            if multiplayer and state_2 is not None:
                action_2, _, state_2, _ = model_2.step(obs,S=state_2, M=dones)
                
            else:
                action_1, _, _, _ = model_1.step(obs)
                # have the second model take an action if appropriate
                if multiplayer:
                    action_2, _, _, _ = model_2.step(obs)
            # take a step forward in the environment, return new observation
            # return any reward and if the environment needs to be reset
            model_1_actions.append(action_1[0])
            if multiplayer:
                model_2_actions.append(action_2[0])
            # pass in both actions if there are two models
            # reward in this case is the default reward
            # in competitive multiplayer, this is Player 1's reward
            if multiplayer:
                obs, _, _, done, _ = env.step(action_1, action_2)
            # otherwise, ignore the second 
            else:
                obs, _, _, done, _ = env.step(action_1)
            lives = env.getLives()
            done = done.any() if isinstance(done, np.ndarray) else done
            # done is true whenever a reset is necessary
            # occurs on death or game over
            if done:
                # Pong only uses done on game over, so make the episode reward the game score
                if isPong:
                    break
                # if it's not Pong, just do what I did before
                elif lives == 0:
                    break
                else:
                    obs = env.reset()
        env.close()
        dirname = os.path.dirname(library_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        library_file = open(library_path,'w+b')
        cloudpickle.dump(state_library, library_file)
        library_file.close()

        action_file = open(action_path,'w+')
        with action_file as csv_scores:
            filewriter = csv.writer(csv_scores, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if multiplayer:
                filewriter.writerow(['model1_actions_groundtruth', 'model2_actions_groundtruth'])
                for j in range(0,len(model_1_actions)):
                    filewriter.writerow([model_1_actions[j], model_2_actions[j]])
            else:
                filewriter.writerow(['model1_actions_groundtruth'])
                for j in range(0,len(model_1_actions)):
                    filewriter.writerow([model_1_actions[j]])

        image_path = osp.expanduser(args.library_path + "/state_images/")
        dirname = os.path.dirname(image_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        # convert each state to an image
        for i in range(1,len(state_library) + 1):
            image_path = osp.expanduser(args.library_path + "/state_images/state" + str(i) + ".jpg")
            img = state_library[i-1].state
            frame1 = img[0:84,0:84,0]
            frame2 = img[0:84,0:84,1]
            frame3 = img[0:84,0:84,2]
            frame1 = np.reshape(frame1, (84, 84, 1))
            frame2 = np.reshape(frame2, (84, 84, 1))
            frame3 = np.reshape(frame3, (84, 84, 1))
            if "SpaceInvaders" in args.env:
                frames = [frame1, frame2, frame3]
            else:
                frame4 = img[0:84,0:84,3]
                frame4 = np.reshape(frame4, (84, 84, 1))
                frames = [frame1, frame2, frame3, frame4]
            frame = LazyFrames(frames)
            if "SpaceInvaders" in args.env:
                img=np.round(0.25*frame._frames[0])+np.round(0.5*frame._frames[1])+np.round(frame._frames[2])
            else:
                img=np.round(0.125*frame._frames[0])+np.round(0.25*frame._frames[1])+np.round(0.5*frame._frames[2])+np.round(frame._frames[3])
            img = img.astype(np.dtype('u1'))
            img=np.concatenate((img, img, img),axis=2)
            height = np.shape(img)[0]
            width = np.shape(img)[1]
            size = 4
            # resize the screen and return it as the image
            img = cv2.resize(img, (width*size, height*size), interpolation=cv2.INTER_AREA)
            matplotlib.image.imsave(image_path, img)

    if args.evaluate_states:
        library_path = osp.expanduser(args.library_path + "/state_library")
        action_load_path = osp.expanduser(args.library_path + "/actions.csv")
        action_test_path = osp.expanduser(args.eval_path + "/actions.csv")
        loaded_models_path = osp.expanduser(args.eval_path + "/loaded_models.txt")
        dirname = os.path.dirname(loaded_models_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        loaded_models_file = open(loaded_models_path,'w+')
        loaded_models_file.write(str(extra_args))
        loaded_models_file.close()


        library_file = open(library_path, 'rb')
        state_library = cloudpickle.load(library_file)
        library_file.close()
        model_1_actions = list()
        if multiplayer:
            model_2_actions = list()
        for i in range(1,len(state_library) + 1):
            # reconstruct the state

            # get whole matrix
            img = state_library[i-1].state
            # extract each layer
            frame1 = img[0:84,0:84,0]
            frame2 = img[0:84,0:84,1]
            frame3 = img[0:84,0:84,2]
            # reshape each layer into the proper format
            frame1 = np.reshape(frame1, (84, 84, 1))
            frame2 = np.reshape(frame2, (84, 84, 1))
            frame3 = np.reshape(frame3, (84, 84, 1))
            if "SpaceInvaders" in args.env:
                # concatenate
                frames = [frame1, frame2, frame3]
            else:
                # fourth frame only used for Pong, doesn't work for Space Invaders
                frame4 = img[0:84,0:84,3]
                frame4 = np.reshape(frame4, (84, 84, 1))
                # concatenate
                frames = [frame1, frame2, frame3, frame4]
            # create the state by passing in the concatenated frames to the LazyFrames class
            obs = LazyFrames(frames)
            # the observation can now be used with the models
            # get actions and append them to my saved lists
            action_1, _, _, _ = model_1.step(obs)
            model_1_actions.append(action_1[0])
            if multiplayer:
                action_2, _, _, _ = model_2.step(obs)
                model_2_actions.append(action_2[0])

        # load the ground truth actions from the library
        action_load_file = open(action_load_path, 'r')
        reader = csv.reader(action_load_file)
        # open new file for writing
        model_action_file = open(action_test_path,'w+')
        with model_action_file as csv_scores:
            filewriter = csv.writer(csv_scores, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            row_num = 0
            # loop through rows in ground truth file
            for row in reader:
                if row_num == 0:
                    # write headers if it's row 0
                    if multiplayer:
                        write_data = ['model1_actions', 'model2_actions']
                    else:
                        write_data = ['model1_actions']
                else:
                    # otherwise, get actions to write
                    if multiplayer:
                        write_data = [model_1_actions[row_num - 1], model_2_actions[row_num - 1]]
                    else:
                        write_data = [model_1_actions[row_num - 1]]
                # write the row of the ground truth actions plus the taken actions from whatever models I'm testing and write the row
                if len(row) == 1:
                    filewriter.writerow([row[0]] + write_data)
                else:
                    filewriter.writerow([row[0], row[1]] + write_data)
                row_num += 1


        

        # save the states as images
        image_path = osp.expanduser(args.eval_path + "/state_images/")
        dirname = os.path.dirname(image_path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        # for each state
        for i in range(1,len(state_library) + 1):
            # reconstruct the state properly
            image_path = osp.expanduser(args.eval_path + "/state_images/state" + str(i) + ".jpg")
            img = state_library[i-1].state
            frame1 = img[0:84,0:84,0]
            frame2 = img[0:84,0:84,1]
            frame3 = img[0:84,0:84,2]
            frame1 = np.reshape(frame1, (84, 84, 1))
            frame2 = np.reshape(frame2, (84, 84, 1))
            frame3 = np.reshape(frame3, (84, 84, 1))
            if "SpaceInvaders" in args.env:
                frames = [frame1, frame2, frame3]
            else:
                frame4 = img[0:84,0:84,3]
                frame4 = np.reshape(frame4, (84, 84, 1))
                frames = [frame1, frame2, frame3, frame4]
            frame = LazyFrames(frames)
            # multiply different layers of the frame to visualize motion
            if "SpaceInvaders" in args.env:
                img=np.round(0.25*frame._frames[0])+np.round(0.5*frame._frames[1])+np.round(frame._frames[2])
            else:
                img=np.round(0.125*frame._frames[0])+np.round(0.25*frame._frames[1])+np.round(0.5*frame._frames[2])+np.round(frame._frames[3])
            # convert to 8 bit unsigned integers
            img = img.astype(np.dtype('u1'))
            # concatenate to get an "RGB" image
            img=np.concatenate((img, img, img),axis=2)
            # upscale by a factor of 4
            height = np.shape(img)[0]
            width = np.shape(img)[1]
            size = 4
            # resize the screen and return it as the image
            img = cv2.resize(img, (width*size, height*size), interpolation=cv2.INTER_AREA)
            # save the image
            matplotlib.image.imsave(image_path, img)

            

    sess_1.close()
    if multiplayer:
        sess_2.close()
    return model_1, model_2

# small class to easily access state observations in the saved lists I create
class StateWrapper(object):
    def __init__(self, obj):
        self.state = obj


if __name__ == '__main__':
    main(sys.argv)
