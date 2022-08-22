#!/usr/bin/env python3
try:
    import cv2
except ModuleNotFoundError:
    print('\033[93m'+
            'cv2 module (openCV) not found!\n render unavailable'+
          '\033[0m' )
import numpy as np
import math
import time
from rugby_players import *




class RugbyGame:
    def __init__(self,field_width,field_lenght,game_duration, \
                games_to_play=1, single_actor=False, \
                verbose=False,tuiose=False,renderose=False,render_duration_ms=50):

        #GAME FIELD
        self.FIELD_SIZE_X=field_width
        self.FIELD_SIZE_Y=field_lenght
        self.TILE_SIZE=29
        self.DEFENSE_TEAM_LINE=math.ceil(1.5*field_lenght//5)
        self.ATTACK_TEAM_LINE=math.ceil(field_lenght*(1-1.5/5))

        #TEAMS
        DEFENDERS_LINE=2
        ATTACKERS_LINE=field_lenght-DEFENDERS_LINE
        MID_COLUMN=math.floor(field_width//2)
        LEFT_COLUMN=MID_COLUMN-3
        RIGHT_COLUMS=MID_COLUMN+3
        #NOTE [x,y]
        self.ATTACKERS_INITIAL_POSES=[  [LEFT_COLUMN,ATTACKERS_LINE],
                                        [MID_COLUMN,ATTACKERS_LINE],
                                        [RIGHT_COLUMS,ATTACKERS_LINE]]
        self.DEFENDERS_INITIAL_POSES=[  [LEFT_COLUMN,DEFENDERS_LINE],
                                        [MID_COLUMN,DEFENDERS_LINE],
                                        [RIGHT_COLUMS,DEFENDERS_LINE]]
        self.attackers=[Player( number=number, role="attacker",
                                initial_pose_x=self.ATTACKERS_INITIAL_POSES[number][0],
                                initial_pose_y=self.ATTACKERS_INITIAL_POSES[number][1])
                        for number in range(0,len(self.ATTACKERS_INITIAL_POSES))]

        self.defenders=[Player( number=number, role="defenders",
                                initial_pose_x=self.DEFENDERS_INITIAL_POSES[number][0],
                                initial_pose_y=self.DEFENDERS_INITIAL_POSES[number][1])
                        for number in range(0,len(self.DEFENDERS_INITIAL_POSES))]

        #BEHAVIOURS
        self.SINGLE_ACTOR=single_actor
        self.attacker_behaviour=np.random.choice
        self.ATTACKER_ACTIONS=[ballPass,stepBack,dodge,tackle,advance]
        self.ATTACKER_BALL_ACTIONS=[ballPass,stepBack,dodge,advance]
        self.ATTACKER_NO_BALL_ACTIONS=[advance,stepBack,dodge,tackle]
        #this assignation for external behaviour override
        self.attacker_action_pool=self.ATTACKER_ACTIONS
        self.attacker_ball_action_pool=self.ATTACKER_BALL_ACTIONS
        self.attacker_no_ball_action_pool=self.ATTACKER_NO_BALL_ACTIONS
        #to mantain a compact formation, able to pass ball
        self.average_attackers_distance=0

        self.defender_behaviour=np.random.choice
        self.DEFENDER_BALL_ACTIONS=[advance,dodge,stepBack,ballPass]
        self.DEFENDER_NO_BALL_ACTIONS=[advance,dodge,stepBack,tackle]
        self.defender_ball_action_pool=self.DEFENDER_BALL_ACTIONS
        self.defender_no_ball_action_pool=self.DEFENDER_NO_BALL_ACTIONS
        
        #TERMINATION CONDITIONS
        self.GAME_DURATION=int(game_duration)
        self.game_over=False

        #STATS
        self.GAMES_TO_PLAY=int(games_to_play)
        self.game_counter=0
        self.winner=""
        self.DEFENDERS_WON=0
        self.ATTACKERS_WON=0

        #prints actions in terminal
        self.VERBOSE=verbose
        #prints game state in terminal
        self.TUIOSE=tuiose
        #shows visual representation of game
        self.RENDEROSE=renderose
        self.RENDER_DURATION_ms=render_duration_ms
        self.last_passage=None

        #hesitation
        self.MAX_DECISION_TIME=.01

        #FIELD INITIALIZATION
        self.update()


    def reset(self):
        #RESET GAME
        self.game_time=0
        #RESET TEAMS
        for player in [*self.attackers,*self.defenders]:
            player.reset()
        self.attackers[1].has_ball=True
        self.previous_ball_x=self.ball_x=self.attackers[1].x
        self.previous_ball_y=self.ball_y=self.attackers[1].y
        self.had_ball=self.has_ball='attackers'

        field=self.update()

        #UPDATE STATS
        self.winner=""
        self.game_counter+=1
        if self.VERBOSE:print("====\t===\t===\t====")

        if self.RENDEROSE:
            try:
                self.render()
            except NameError: pass
            
        return field


#---------------------------------------------------------------------
    def step(self,team_actions_probabilities):
        """
        this method checks if the game terminated, otherwise plays a game's turn: attack and defense phases
        self.SINGLE_ACTOR: only the ball bearer acts intentionally, otherwise randomly

        :param team_actions_probabilities has correct shape for selection of ball/not ball reserved actions
            otherwise actions to avoid are learned 
            TODO action masking for more precise learning

        :return field state,performed actions, turn reward
        """
        rewards=[]
        actions=[]

        if self.checkTerminalConditions(): 
            return None,reward_values['win'] if self.ATTACKERS_WON else reward_values['loss'],None

        self.game_time+=1

        if self.VERBOSE:print("GAME {} ({})\t TIME {} ({})"
                            .format(self.game_counter,self.GAMES_TO_PLAY,
                                    self.game_time,self.GAME_DURATION))
        if self.RENDEROSE:
            try:
                self.render()
            except NameError: pass

        #VALUES EXTRACTION
        team_actions_probabilities= \
            team_actions_probabilities[0][0] if self.SINGLE_ACTOR \
            else team_actions_probabilities[0]
        
        #NOTE defense phase performed agter reward assignation
        attack_rewards,attack_actions=self.attackPhase(team_actions_probabilities)
        
        actions=attack_actions
        if not self.SINGLE_ACTOR:
            rewards.extend(attack_rewards)
        else:
            rewards=[attack_rewards]
        rewards.append(reward_values['time'])

        rewards.extend(self.defensePhase())

        #SUM OF THE COOPERATIVE (attack) AND COMPETITIVE (defense) REWARDS
        rewards=sum(rewards)
        return self.field,rewards,actions


    def update(self):
        """
        updates and returns field state
        """
        self.field=np.zeros((self.FIELD_SIZE_Y,self.FIELD_SIZE_X),dtype=int)

        for player in [*self.attackers,*self.defenders]:
            if player.role=='attacker':
                self.field[player.y,player.x]=\
                  player.number+1 if not player.has_ball else 11+player.number
            else: self.field[player.y,player.x]=-(player.number+1)

        if self.TUIOSE:
            print("GAME {} ({})\t TIME {} ({})"
                            .format(self.game_counter,self.GAMES_TO_PLAY,
                                    self.game_time,self.GAME_DURATION))
            print(self.field)

        return self.field


    def checkTerminalConditions(self):
        """
        game terminates in 2 cases:
        - attacking team passes try line
        - game's time expires
        - TODO defense team scores
        """
        #GAME TIME ENDED
        if self.game_time>=self.GAME_DURATION:
            if self.VERBOSE:print("______time's up______")
            self.DEFENDERS_WON+=1
            self.winner="DEFENDERS"
            return True

        #ATTACKING TEAM SURPASSED TRY LINE
        for player in self.attackers:
            if player.has_ball and player.y<=self.DEFENSE_TEAM_LINE:
                if self.VERBOSE:print("______point scored by attackers______")
                self.ATTACKERS_WON+=1
                self.winner="ATTACKERS"
                return True


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def attackPhase(self,team_actions_probabilities):
        rewards=[]
        actions=[]
        self.average_attackers_distance=0

        for player in self.attackers:
            action_string= \
                "{}".format("*" if player.has_ball else "")+player.name 

            #counts time available to perform an action
            start_time=time.time()
            phase_initial_ball_y=self.ball_y
            while True:

                reward=0
                
                #only ball bearer learns to act
                #   hence team_actions_probabilities single row
                if self.SINGLE_ACTOR :
                    if player.has_ball:
                        action_pool= self.attacker_ball_action_pool
                        action_probability=team_actions_probabilities
                        #NOTE NORMALIZATION
                        action_probability/=np.sum(action_probability)
                    else:
                        #others randomly act
                        action_pool= self.attacker_no_ball_action_pool
                        action_probability=None
                else:
                #everyone learns to act
                    action_pool= self.attacker_action_pool
                    action_probability=team_actions_probabilities[player.number]
                    #NOTE NORMALIZATION 
                    action_probability/=np.sum(action_probability)

                selected_action=self.attacker_behaviour(action_pool, \
                                                        p=action_probability).__name__

                #ACTIONS EFFECTS
                if selected_action=='advance':
                    motion=np.random.randint(1,player.MAX_SPEED)
                    if advance(player,self,motion):
                        if player.has_ball:
                            reward=reward_values['ball_possesion']
                        action_string+=' advances'
                        if not self.SINGLE_ACTOR:
                            action=[1,0,0,0,0]
                        else:
                            action=[1,0,0,0]
                        break

                elif  selected_action=='stepBack':
                    if stepBack(player,self):
                        action_string+=' steps back'
                        if player.has_ball:
                            reward=reward_values['ball_possesion']
                        if not self.SINGLE_ACTOR:
                            action=[0,1,0,0,0]
                        else:
                            action=[0,1,0,0]
                        break

                elif selected_action=='dodge':
                    motion=1
                    motion=np.random.choice([-1,1])*motion
                    if dodge(player,self,motion):
                        if player.has_ball:
                            reward=reward_values['ball_possesion']
                        action_string+=' dodges'
                        if not self.SINGLE_ACTOR:
                            action=[0,0,1,0,0]
                        else:
                            action=[0,0,1,0]
                        break

                elif  selected_action=='tackle':
                    tackled=np.random.choice(self.defenders)
                    if tackled.has_ball:reward=reward_values['ball_gain']
                    if tackle(player,tackled,self):
                        action_string+=' tackles'
                        if not self.SINGLE_ACTOR:
                            action=[0,0,0,1,0]
                        else:
                            action=[0,0,0,1]
                        break

                elif  selected_action=='ballPass':
                    receiver=player
                    while receiver.name==player.name:
                        receiver=np.random.choice(self.attackers)
                    if ballPass(player,receiver,self,MAX_PASS_DISTANCE=7,force_backpass=False):
                        reward=reward_values['ball_possesion']
                        action_string+=' passes'
                        self.last_passage=BallPassage(player,receiver)
                        if not self.SINGLE_ACTOR:
                            action=[0,0,0,0,1]
                        else:
                            action=[0,0,0,1]
                        break

                #PLAYER HESITATED
                if time.time()-start_time>self.MAX_DECISION_TIME:
                    action_string+=' hesitates'
                    if not self.SINGLE_ACTOR:
                        action=[0,0,0,0,0]
                    else:
                        action=[0,0,0,0]
                    break
            

            # #POSITIVE REWARD TO MOVE THE BALL TOWARD THE TRY LINE
            if (self.SINGLE_ACTOR and (not rewards or reward>rewards)):
                ball_advancement=phase_initial_ball_y-phase_current_ball_y
                rewards=reward+reward_values['toward_try']* \
                    (ball_advancement if ball_advancement>0 else 0)
                actions=action
            elif not self.SINGLE_ACTOR:
                rewards.append(reward)
                ball_advancement=phase_initial_ball_y-phase_current_ball_y
                if ball_advancement>0:
                    rewards.append(reward_values['toward_try']*ball_advancement)
                actions.append(action)
                
            
            if self.VERBOSE:print(action_string)
            self.update()
            if self.RENDEROSE:
                try:
                    self.render()
                except NameError: pass


        #PUNISHING FOR KEEPING A FORMATION TOO BROAD
            self.average_attackers_distance=\
                averageWingMidDistance(self.attackers,self.average_attackers_distance,self.game_time)
            phase_current_ball_y=self.ball_y
                
        if self.average_attackers_distance>self.attackers[1].MAX_PASS_DISTANCE:
            if self.SINGLE_ACTOR:
                rewards+=self.average_attackers_distance*reward_values['broad_formation']
            else:
                rewards.append(self.average_attackers_distance*reward_values['broad_formation'])


        return rewards,actions


    def defensePhase(self):
        rewards=[]
        
        for player in self.defenders:
            action_string= \
                "{}".format("*" if player.has_ball else "")+player.name 

            start_time=time.time()

            while True:
                reward=0

                action_pool=self.defender_no_ball_action_pool if not player.has_ball \
                            else self.defender_ball_action_pool

                selected_action= self.defender_behaviour(action_pool).__name__

                if selected_action=='advance':
                    motion=np.random.randint(1,player.MAX_SPEED)
                    if advance(player,self):
                        action_string+=' advances'
                        break

                elif  selected_action=='tackle':
                    tackled=np.random.choice(self.attackers)
                    if tackled.has_ball:
                        reward=reward_values['ball_loss']
                    if tackle(player,tackled,self):
                        action_string+=' tackles'
                        break

                elif  selected_action=='stepBack':
                    if stepBack(player,self):
                        action_string+=' steps back'
                        break

                elif selected_action=='dodge':
                    motion=1
                    motion=np.random.choice([-1,1])*motion
                    if dodge(player,self,motion):
                        action_string+=' dodges'
                        break

                elif  selected_action=='ballPass':
                    receiver=player
                    while receiver.name==player.name:
                        receiver=np.random.choice(self.defenders)
                    if ballPass(player,receiver,self):
                        action_string+=' passes'
                        self.last_passage=BallPassage(player,receiver)
                        break
                
                if time.time()-start_time>self.MAX_DECISION_TIME:
                    action_string+=' hesitates'
                    reward=reward_values['hesitation']
                    break
            
            #NEGATIVE ATTACK REWARDS
            #NOTE REWARD_COEFF weights rewards 1/3 if only one attacker acts intentionally
            REWARD_COEFF=.333 if self.SINGLE_ACTOR else 1
            rewards.append(REWARD_COEFF*reward)

            if self.VERBOSE:print(action_string)
            self.update()
            if self.RENDEROSE:
                try:
                    self.render()
                except NameError: pass
                
        return rewards


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    def render(self):
        #FIELD
        visual_field=np.zeros((self.FIELD_SIZE_Y*self.TILE_SIZE,
                                    self.FIELD_SIZE_X*self.TILE_SIZE,3
                                    ),dtype="uint8")
                                    
        visual_field[self.TILE_SIZE*(self.DEFENSE_TEAM_LINE+1)-1,:,:]=self.attackers[0].color
        visual_field[self.TILE_SIZE*(self.DEFENSE_TEAM_LINE+1)-2,:,:]=150
        #NOTE line is barely visible since defenders scoring by try is not implemented yet
        visual_field[self.TILE_SIZE*(self.ATTACK_TEAM_LINE-1)+1,:,:]= np.array(self.defenders[0].color)//2.8
        visual_field[self.TILE_SIZE*(self.ATTACK_TEAM_LINE-1)+2,:,:]=150//2.8

        #PLAYERS
        for player in [*self.attackers,*self.defenders]:
            visual_field[self.TILE_SIZE*player.y:(player.y+1)*self.TILE_SIZE,\
                        self.TILE_SIZE*player.x:self.TILE_SIZE*(player.x+1)]=player.color
    
            visual_field[self.TILE_SIZE*player.y:self.TILE_SIZE*(player.y+1)
            ,int(self.TILE_SIZE*(player.number/3+ player.x)):int((player.x+(.7+player.number)/3)*self.TILE_SIZE)
                        ]=(80,255,255) 
        
        #BALL
            if player.has_ball:
                visual_field[
                  int(self.TILE_SIZE*(player.y+.2)):int(self.TILE_SIZE*(player.y+.8)),\
                  int(self.TILE_SIZE*(player.x+.2)):int(self.TILE_SIZE*(player.x+.8)),:]=player.ball_color 
        
        #LAST PASSAGE TRAJECTORY
        if not self.last_passage is None:   
            cv2.arrowedLine(visual_field,
                            (int(self.TILE_SIZE*(self.last_passage.sender.x+.5)),
                             int(self.TILE_SIZE*(self.last_passage.sender.y+.5))),
                            (int(self.TILE_SIZE*(self.last_passage.receiver.x+.5)),
                             int(self.TILE_SIZE*(self.last_passage.receiver.y+.5))),
                            color=self.last_passage.sender.passage_color,thickness=4)
            self.last_passage=self.last_passage.reset()

        try:
            cv2.imshow('GAME:',visual_field)
            cv2.waitKey(self.RENDER_DURATION_ms)
        except NameError: pass


    def tileIsFree(self,tile_x,tile_y):
        for player in [*self.attackers,*self.defenders]:
            if player.x==tile_x and player.y==tile_y:
                return False
        return True


#################################################################################
#REWARDS      ###################################################################
#NOTE reward assignation done in attack/defense phases
reward_values={ 'loss': -200,
                'time': -10,
                'ball_loss': -5,
                'win': 200,
                'toward_try': +8,#+20,
                'away_try': -3,#unused
                'broad_formation': -40,
                'ball_gain': 0,
                'ball_possesion': 0,
                'hesitation':-40
            }

        
###################################################################
#ACTIONS     ######################################################
#NOTE actions return action_performed:bool
def advance(player:Player,game:RugbyGame,motion=1):
    if player.can_advance:
        if player.role=='attacker':
            destination_tile_y=player.y-motion
            if destination_tile_y>=0 and destination_tile_y<game.FIELD_SIZE_Y:
                if all([game.tileIsFree(player.x,path_tile) 
                  for path_tile in range(destination_tile_y,player.y)]):
                    game.previous_ball_y=player.y
                    game.ball_y=destination_tile_y
                    player.y=destination_tile_y
                    return True
        else:
            destination_tile_y=player.y+motion
            if destination_tile_y>=0 and destination_tile_y<game.FIELD_SIZE_Y:
                if all([game.tileIsFree(player.x,path_tile) 
                  for path_tile in range(player.y+1,destination_tile_y+1)]):
                    game.previous_ball_y=player.y
                    game.ball_y=destination_tile_y
                    player.y=destination_tile_y
                    return True
    return False


def stepBack(player:Player,game:RugbyGame,motion=1):
    if player.role=='attacker':
        destination_tile_y=player.y+motion
        if destination_tile_y>=0 and destination_tile_y<game.FIELD_SIZE_Y:
            if all([game.tileIsFree(player.x,path_tile) 
              for path_tile in range(player.y+1,destination_tile_y+1)]):
                player.can_advance=True
                game.previous_ball_y=player.y
                game.ball_y=destination_tile_y
                player.y=destination_tile_y
                return True
    else:
        destination_tile_y=player.y-motion
        if destination_tile_y>=0 and destination_tile_y<game.FIELD_SIZE_Y:
            if all([game.tileIsFree(player.x,path_tile) 
              for path_tile in range(destination_tile_y,player.y)]):
                game.previous_ball_y=player.y
                game.ball_y=destination_tile_y
                player.y=destination_tile_y
                player.can_advance=True
                return True
    return False


def dodge(player:Player,game:RugbyGame,motion=0):
    """
    distance is signed
    """
    destination_tile_x=player.x+motion
    if destination_tile_x>=0 and destination_tile_x<game.FIELD_SIZE_X:
        if motion>0:
            if all([game.tileIsFree(path_tile,player.y) 
              for path_tile in range(player.x+1,destination_tile_x+1)]):
                game.previous_ball_x=player.x
                game.ball_x=destination_tile_x
                player.x=destination_tile_x
                return True
        else:
            if all([game.tileIsFree(path_tile,player.y) 
                for path_tile in range(destination_tile_x,player.x)]):
                game.previous_ball_x=player.x
                game.ball_x=destination_tile_x
                player.x=destination_tile_x
                return True
    return False


def ballPass(thrower:Player,receiver:Player,game:RugbyGame,MAX_PASS_DISTANCE=None,force_backpass=False):
    """
    MAX_PASS_DISTANCE can be set to be senders
    force_backpass does not harm that much attackers
    """
    if thrower.has_ball:
        if distance(thrower,receiver)<= MAX_PASS_DISTANCE \
          if MAX_PASS_DISTANCE is not None else thrower.MAX_PASS_DISTANCE:
            if not force_backpass or \
              (thrower.y<=receiver.y and thrower.role=='attacker') or \
              (thrower.y>=receiver.y and thrower.role=='defender'):
                thrower.has_ball=False
                receiver.has_ball=True
                game.previous_ball_x=thrower.x
                game.previous_ball_y=thrower.y
                game.ball_x=receiver.x
                game.ball_y=receiver.y
                return True
    return False


def tackle(charger:Player,tackled:Player,game:RugbyGame ,MAX_TACKLE_DISTANCE=1.5):
    """
    removing the ball OR stopping the tackled harms a bit
    """
    if charger.can_advance and distance(charger,tackled)<=MAX_TACKLE_DISTANCE:
        if tackled.has_ball:
            tackled.has_ball=False
            game.had_ball=tackled.role
            charger.has_ball=True
            game.has_ball=charger.role
            game.previous_ball_x=tackled.x
            game.previous_ball_y=tackled.y
            game.ball_x=charger.x
            game.ball_y=charger.y
        else:
            tackled.can_advance=False
        return True
    return False
