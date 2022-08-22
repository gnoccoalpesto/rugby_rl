#!/usr/bin/env python3

# import numpy as np
from statistics import mean
from rugby_game import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.nn import log_softmax
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import argparse
try:
    # import tkinter
    # import pandas as pd
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    print('\033[93m'+
            'matplotlib or tkinter modules not found!\n plot unavailable'+
          '\033[0m' )




class policyNet:
    """
        hyperparams not changable at runtime
        INPUT SHAPE: flattened field array, default=1,13*9==117
        OUTPUT SHAPE: 4 if all attacking players learn  else 5
    """
    def __init__(self,input_shape,output_len,single_actor=True,play_randomly=False):

        self.LEARNING_RATE=2
        self.REWARD_DISCOUNT_FACTOR=.92

        #if single_actor, probabilities only for ball bearer
        self.OUTPUT_LEN=output_len*(1 if single_actor else 3)
        self.OUTPUT_SHAPE=(1,1 if single_actor else 3,output_len)

        #OUTPUT: action probabilities
        self.model= keras.Sequential([
                keras.layers.Flatten(input_shape=input_shape),
                keras.layers.Dense(32, activation="softmax"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(activation='softmax',units=self.OUTPUT_LEN),
                keras.layers.Reshape((self.OUTPUT_SHAPE))
        ])

        self.optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)

        self.PLAY_RANDOMLY=play_randomly

    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def sample(self,state:np.ndarray):

        # with tf.device('/device:GPU:0'):
        #TODO action masking for non valid ones
        if self.PLAY_RANDOMLY:
            actions_probabilities=1/output_len*np.ones(self.OUTPUT_SHAPE,dtype=float)
            return actions_probabilities

        #TODO normalization; NOTE easier to do in game       
        return self.model.predict(state[None,:])[0]


    def train(self,states:list,actions:list,rewards:list,next_states:list):

        # with tf.device('/device:GPU:0'):
        game_loss=[]
        #TODO more efficient if G_t=gamma * G_t++ + R_t++?
        expected_return=0
        for time_step,(state,action,next_state) in enumerate(zip(states,actions,next_states)):
            #DISCOUNTED REWARDS SUM
            expected_return=sum([(self.REWARD_DISCOUNT_FACTOR**future_time)*reward
                for future_time,reward in enumerate(rewards[time_step:])])

            #GRADIENT ASCENT
            with tf.GradientTape() as tape:
                actions_probabilities=self.model(state[None,:])[0]
                log_probs=tf.nn.log_softmax(actions_probabilities)
                # log_probs=np.log(softmaxInActionPreferencesPolicy(self.propabilities,features_mode='unitary'))
                #TODO masking
                # log_prob=tf.reduce_sum(tf.one_hot(net_output,num_outputs)*log_probs,axis=1)

                loss=-expected_return*log_probs
                game_loss.append(loss.numpy()[0,0])

                vars=self.model.trainable_variables
            #COMPUTE GRADIENT
            grads = tape.gradient(loss, vars)
            #GRADIENT ASCENT
            self.optimizer.apply_gradients(zip(grads,vars))
            #AVERAGE FOR SSTATISTICS
            return np.mean(game_loss)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
def startGame(game:RugbyGame):

    loss_history=[]
    state_history=[]
    # next_state_history=[]
    actions_history=[]
    rewards_history=[]

    while game.game_counter<game.GAMES_TO_PLAY:
        #INITIALIZE AND OBSERVE STATE
        #NOTE field==field's state
        field=game.reset()

        #EPISODE GENERATION
        while True:
            state_history.append(field)

            #ACTIONS SELECTION
            attack_probabilities=net.sample(field)

            #SYSTEM EVOLUTION
            #NOTE actions: the ones actually performed
            field,rewards,actions=game.step(attack_probabilities)

            #TERMINATION
            if field is None:
                #EPISODE REWARD for win or loss
                #NOTE since episode termination computed at the beginning of step
                #TODO could change
                rewards_history[-1]+=rewards

                #EVOLVED STATE HISTORY
                next_state_history=state_history[1:].copy()
                state_history.pop()
                break

            actions_history.append(actions)
            rewards_history.append(rewards)

        #GAME STATISTICS
        if game.VERBOSE:print('GAME OVER. winner: {}\n'.format(game.winner))
        print('PARTIAL SCORE\tATT {} -- {} DEF'.format(game.ATTACKERS_WON,game.DEFENDERS_WON))

        #UPDATE NET
        if not args.RNG:
            loss_history.append(net.train(  state_history, \
                                            actions_history, \
                                            rewards_history, \
                                            next_state_history))

    #LEARNING RESULT WRT LOSS
    if not args.RNG:
        try:
            plt.plot(loss_history,color='red',marker='o')
            plt.xticks(range(0,len(loss_history)+1,1))
            plt.ylabel('average loss')
            plt.xlabel('games')
            games_score=" final score: ATT {} -- {} DEF".format(game.ATTACKERS_WON,game.DEFENDERS_WON)
            plt.title('LEARNING RESULTS;'+games_score)
            plt.show()
        except NameError: pass


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
def softmaxInActionPreferencesPolicy(weights:np.ndarray,features=[],
                                    features_mode='linear',stabilized=False):
    """
    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
    stable in case output reaches the limit of float bytes
    """
    if features_mode=='unitary':
        preferences=np.sum(weights)
    elif features_mode=='linear':
        preferences=np.dot(weights.T,features)

    pref_exponential = np.exp(preferences if not stabilized else preferences-np.max(preferences))
    return np.exp(pref_exponential)/np.sum(pref_exponential)


#############################################################################
#############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--single_actor',action='store_true',
                    help='decide wether all attacking team or only ball-bearer learns')
    parser.add_argument('-w', '--width',type=int, default=9, required=False,
                    help='field width')
    parser.add_argument('-l', '--lenght',type=int, default=13, required=False,
                    help='field lenght')
    parser.add_argument('-d', '--duration',type=int, default=20, required=False,
                    help='game duration (steps)')
    parser.add_argument('-g', '--games',type=int, default=500, required=False,
                    help='games to play')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='prints actions and partial scores in cli')
    parser.add_argument('-t', '--tuiose', required=False, action='store_true',
                        help='prints field(game) state in cli')
    parser.add_argument('-r', '--renderose', action='store_true',
                        help='shows visual game field (requires openCV)')
    parser.add_argument('-f', '--frame_duration',type=int, default=10, required=False,
                        help='frame permanence duration in ms')
    #TODO add mutual exclusivity between --RNG and -s
    parser.add_argument('--RNG',action='store_true',
                        help='starts a random game; all params valid except --single_actor')
    #RETURNS DICT
    # return vars(parser.parse_args())
    #RETURNS STRUC
    return parser.parse_args()


if __name__ == '__main__':

    args=parse_arguments()

    game = RugbyGame(field_width=args.width,
                field_lenght=args.lenght,
                game_duration=args.duration,
                games_to_play=args.games,
                single_actor=args.single_actor,
                verbose=args.verbose,
                tuiose=args.tuiose,
                renderose=args.renderose,
                render_duration_ms=args.frame_duration)

    #BEHAVIOUR OVERRIDE; NOTE UNUSED
    # game.attacker_behaviour=np.random.choice
    # game.arg_attacker_behaviour=game.ATTACKER_ACTIONS
    # game.defender_behaviour=np.random.choice
    # game.arg_defender_behaviour=game.DEFENDER_ACTIONS

    output_len=len(game.attacker_action_pool if not args.single_actor
                    else game.attacker_ball_action_pool)

    net=policyNet(  input_shape=(args.lenght,args.width),\
                    output_len=output_len,
                    single_actor=args.single_actor,
                    play_randomly=args.RNG)

    print('-=`WELCOME TO THE GAME OF RUGBY=-')
    startGame(game)


    try:
        cv2.destroyAllWindows()
    except NameError: pass
    print('GAME OVER\nSCORE ATT {} - {} DEF'
        .format(game.ATTACKERS_WON,game.DEFENDERS_WON))
