#!/usr/bin/env python3
import numpy as np


class Player:
    def __init__(self,number,role,initial_pose_y,initial_pose_x):
        if role=='attacker':
            self.name="A"+str(number)
            self.color= (0,0,255)
            self.ball_color=30
            self.passage_color=(255,0,255)
        else:
            self.name="D"+str(number)
            self.color= (255,0,0)
            self.ball_color=(255,0,255)
            self.passage_color=(100,255,0)
        self.number=number
        self.role=role

        #TODO use this instead of whole tile for visual
        # self.SHAPE_SIZE=5

        self.previous_x=self.initial_x=initial_pose_x
        self.previous_y=self.initial_y=initial_pose_y

        #speed of 4 massively improves attackers winrate
        self.MAX_SPEED=3

        self.MAX_PASS_DISTANCE=4

        self.reset()


    def reset(self):
        self.previous_x=self.x=self.initial_x
        self.previous_y=self.y=self.initial_y

        self.had_ball=self.has_ball=False

        self.could_advance=self.can_advance=True

        # self.team_distance=0


def distance(player1:Player,player2:Player):
    """
    euclidean distance between 2 players
    """
    return np.linalg.norm(\
     np.array([player1.x,player1.y])-np.array([player2.x,player2.y]))
  

def averagePlayerTeamDistance(player:Player,teammates:Player,time=1):
    """
    rationale: if player stays closer to MAX_BALL_PASS_DISTANCE
    """
    #TODO typing teammates players array
    player.team_distance=(time-1)*player.team_distance
    for teammate in teammates:
        #TODO how to arg type for arrays such methods visible?
        if teammate.name!=player.name:
            player.team_distance+=distance(player,teammate)
    player.team_distance/=time


def averageWingMidDistance(team:list,average_wing_mid_distance:float,time_step=0):
    """
    running average for distance between mid(central) and mings
    """
    wing_mid_distance=(distance(team[0],team[1])+distance(team[2],team[1]))/2
    average_wing_mid_distance=(average_wing_mid_distance*time_step + wing_mid_distance)/(time_step+1)
    return average_wing_mid_distance

    
#TODO may substitute with a named touple
#from collection import namedtouples
class BallPassage:
    def __init__(self,sender:Player,receiver:Player):
        self.sender=sender
        self.receiver=receiver

    def reset(self):
        return None
