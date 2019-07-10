#!/bin/sh

# this needs to be in the main folder to work

# variables user can change
ENV_NAME=SpaceInvadersNoFrameskip-v4
MULTIPLAYER=true
MODE=33
NUM_GAMES=10
SAVE_PATH=~/models/Pong_computer_performance_1m


# number of full full steps
NUM_STEPS=60000
SAVE_INTERVAL=20000



# number of times to loop
NUM_TIMES=$((NUM_STEPS / SAVE_INTERVAL))

# keep track of how many timesteps I've completed
COMPLETED_TIMESTEPS=0

# specify extensions for logged files
GAME_LOG=/games.txt
SCORE_TEXT_NAME=/scores.txt
SCORE_CSV_NAME=/scores.csv
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
echo "Multiplayer: $MULTIPLAYER" >> "$SETTINGS_PATH"
echo "Mode: $MODE" >> "$SETTINGS_PATH"
echo "Full training steps: $NUM_STEPS" >> "$SETTINGS_PATH"
echo "Pause training interval: $SAVE_INTERVAL" >> "$SETTINGS_PATH"
echo "Number of games every pause: $NUM_GAMES" >> "$SETTINGS_PATH"

# create arrays to hold average scores and steps completed
declare -a SCORES
declare -a STEPS

# activate virtual environment and change directory
source openai/bin/activate
cd baselines

if ($MULTIPLAYER)
then
    python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$NUM_STEPS --save_interval=$SAVE_INTERVAL --multiplayer --mode=$MODE --save_path=$SAVE_PATH
else
    python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=$NUM_STEPS --save_interval=$SAVE_INTERVAL --mode=$MODE --save_path=$SAVE_PATH
fi

# main for loop
# each loop consists of training for a given number of steps, stopping, then playing X amount of games
for (( i=1; i<=$NUM_TIMES; i++ ))
do
    

    # string extension
    STAGE=/stage
    # save the files according to the specific stage name
    LOAD_PATH=$SAVE_PATH$STAGE$i

    # multiplayer stuff, loads the code with two players
    if ($MULTIPLAYER)
    then
        # suffixes to differentiate files
        PLAYER_1=_player1
        PLAYER_2=_player2
        # initial start, no loading (for now)
        # specify new load paths
        LOAD_PATH_1=$LOAD_PATH$PLAYER_1
        LOAD_PATH_2=$LOAD_PATH$PLAYER_2
        # load the models and play the number of games, grab the terminal output and post it in a temporary file
        (python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=0 --multiplayer --mode=$MODE --load_path_1=$LOAD_PATH_1 --load_path_2=$LOAD_PATH_2 --play --num_games=$NUM_GAMES) | tee "$GAMES_PATH"
        # (python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=0 --mode=1 --load_path=$LOAD_PATH_1 --play --num_games=$NUM_GAMES) | tee "$GAMES_PATH"
        # grab the temporary file and load it into a variable
        OUTPUT=`cat $GAMES_PATH`
        #specific keyword at the end usually in the terminal output to show average score
        AVERAGE_KEYWORD="average score="
        # placement is that everything after the above string is the average score
        # remove everything else so only the score is left
        SCORE=$(echo $OUTPUT | grep -oP "$AVERAGE_KEYWORD\K.*")
        # add the score to the array
        SCORES[i-1]=$SCORE
        # delete the temporary file
        rm $GAMES_PATH
    # single-player code
    else
        # play games and get output
        (python3 -m baselines.run --alg=deepq --env=$ENV_NAME --num_timesteps=0 --mode=$MODE --load_path=$LOAD_PATH --play --num_games=$NUM_GAMES) | tee "$GAMES_PATH"
        # get output and extract average score
        OUTPUT=`cat $GAMES_PATH`
        AVERAGE_KEYWORD="average score="
        SCORE=$(echo $OUTPUT | grep -oP "$AVERAGE_KEYWORD\K.*")
        SCORES[i-1]=$SCORE
        # delete temp file
        rm $GAMES_PATH
    fi
    # increment number of timesteps
    COMPLETED_TIMESTEPS=$((COMPLETED_TIMESTEPS + SAVE_INTERVAL))
    # add to array
    STEPS[i-1]=$COMPLETED_TIMESTEPS


done
# end of main loop

# create txt and csv files
touch $SCORE_TEXT_PATH
touch $SCORE_CSV_PATH

# log the scores and number of steps to a text file, separated by a semicolon
for (( j=0; j<"${#SCORES[@]}"; j++ )) 
do
    echo -n "${STEPS[j]}" >> "$SCORE_TEXT_PATH"
    echo -n ";" >> "$SCORE_TEXT_PATH"
    echo "${SCORES[j]}" >> "$SCORE_TEXT_PATH"
done
# convert the text file to a csv file
(echo "Steps;AvgScore" ; cat "$SCORE_TEXT_PATH") | sed 's/;/,/g' > "$SCORE_CSV_PATH"
# delete text file since it's no longer needed
rm $SCORE_TEXT_PATH
# give completion message
echo ""
echo ""
echo "Script complete"
echo ""
echo ""
# close virtual environment and end script

deactivate
