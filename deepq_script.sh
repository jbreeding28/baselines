#!/bin/sh

# this needs to be in the main folder to work

# variables user can change
ENV_NAME=SpaceInvadersNoFrameskip-v4
MULTIPLAYER_TRAIN=false
TRAIN_MODE=1
SAVE_PATH=~/models/SpaceInvaders_solo_scratch
# leave these blank if you don't want to load in a starting point
LOAD_PATH_TRAIN_1=
LOAD_PATH_TRAIN_2=


# number of full full steps
NUM_STEPS=1000000
SAVE_INTERVAL=200000

# number of times to loop
NUM_TIMES=$((NUM_STEPS / SAVE_INTERVAL))

# keep track of how many timesteps I've completed
COMPLETED_TIMESTEPS=0

# specify extensions for logged files
SETTINGS_NAME=/settings.txt
# create full paths
SCORE_TEXT_PATH=$SAVE_PATH$SCORE_TEXT_NAME
SCORE_CSV_PATH=$SAVE_PATH$SCORE_CSV_NAME
SETTINGS_PATH=$SAVE_PATH$SETTINGS_NAME
GAMES_PATH=$SAVE_PATH$GAME_LOG
# overwrite the folder if it exists
if [ -d $SAVE_PATH ]
then
    rm -rf $SAVE_PATH
fi
# create a new log folder
mkdir $SAVE_PATH

# log settings for reproduction purposes
touch $SETTINGS_PATH
echo "Environment: $ENV_NAME" >> "$SETTINGS_PATH"
echo "Training multiplayer: $MULTIPLAYER_TRAIN" >> "$SETTINGS_PATH"
echo "Playing multiplayer: $MULTIPLAYER_PLAY" >> "$SETTINGS_PATH"
echo "Training mode: $TRAIN_MODE" >> "$SETTINGS_PATH"
echo "Playing mode: $PLAY_MODE" >> "$SETTINGS_PATH"
echo "Full training steps: $NUM_STEPS" >> "$SETTINGS_PATH"
echo "Pause training interval: $SAVE_INTERVAL" >> "$SETTINGS_PATH"

# create arrays to hold average scores and steps completed
declare -a SCORES
declare -a STEPS
declare -a WINS

# activate virtual environment and change directory
source openai/bin/activate
cd baselines


# train

# loop through the code playing games with each saved model
for (( i=0; i<=$NUM_TIMES; i++ ))
do
    

    # string extension
    STAGE=/stage
    # load the files according to the specific stage name
    LOAD_PATH=$SAVE_PATH$STAGE$i
    # suffixes to differentiate files
    PLAYER_1=_player1
    PLAYER_2=_player2
    # specify full load paths for multiplayer
    LOAD_PATH_1=$LOAD_PATH$PLAYER_1
    LOAD_PATH_2=$LOAD_PATH$PLAYER_2


# check if I need to load an initial model in
    if [ -z "$LOAD_PATH_TRAIN_1" ] && [ -z "$LOAD_PATH_TRAIN_2" ]
    then
        if ($MULTIPLAYER_TRAIN)
        then
            python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$COMPLETED_TIMESTEPS --multiplayer --mode=$TRAIN_MODE --save_path=$LOAD_PATH
        else
            python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$COMPLETED_TIMESTEPS --mode=$TRAIN_MODE --save_path=$LOAD_PATH
        fi
    else
        # if I'm here, I need to load a model
        if ($MULTIPLAYER_TRAIN)
        then
            # check if a load model for player 1 exists. If not, execute the first statement and load only player 2.
            # verified that one of the paths exists, so if player 1's doesn't exist, then only player 2's exists
            if [ -z "$LOAD_PATH_TRAIN_1" ]
            then
                python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$COMPLETED_TIMESTEPS --multiplayer --mode=$TRAIN_MODE --save_path=$LOAD_PATH --load_path_2=$LOAD_PATH_TRAIN_2
            # if no load path exists for loading in player 2 but a player 1
            elif [ -z "$LOAD_PATH_TRAIN_2" ]
            then
                python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$COMPLETED_TIMESTEPS --multiplayer --mode=$TRAIN_MODE --save_path=$LOAD_PATH --load_path_1=$LOAD_PATH_TRAIN_1
            # here, both players should be loaded
            else
                python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$COMPLETED_TIMESTEPS --multiplayer --mode=$TRAIN_MODE --save_path=$LOAD_PATH --load_path_1=$LOAD_PATH_TRAIN_1 --load_path_2=$LOAD_PATH_TRAIN_2
            fi
        else
        # in single-player, if load path doesn't exist
            if [ -z "$LOAD_PATH_TRAIN_1" ]
            then
                python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$COMPLETED_TIMESTEPS --mode=$TRAIN_MODE --save_path=$LOAD_PATH
            else
                python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$COMPLETED_TIMESTEPS --mode=$TRAIN_MODE --save_path=$LOAD_PATH --load_path=$LOAD_PATH_TRAIN_1
            fi
        fi
    fi
    COMPLETED_TIMESTEPS=$((COMPLETED_TIMESTEPS + SAVE_INTERVAL))


done
# end of main loop

echo ""
echo ""
echo "Script complete"
echo ""
echo ""
# close virtual environment and end script
deactivate
