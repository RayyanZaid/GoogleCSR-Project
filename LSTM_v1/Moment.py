# A moment looks like this : [x1,y1,x2,y2, .... x5,y5, bx,by,bz, shot_clock, game_clock]  :  LABEL

# Each moment should have :
    # 1) 25-D array with all the necessary spatio-temporal data
    # 2) A label

# Line from CSV : "[{'x': 81.67902, 'y': 18.37563}, {'x': 89.80558, 'y': 31.86329}, {'x': 80.31479, 'y': 31.49464}, {'x': 69.50902, 'y': 2.11084}, {'x': 79.07408, 'y': 26.42447}, {'x': 74.55269, 'y': 13.83294}, {'x': 83.27226, 'y': 44.45603}, {'x': 79.66844, 'y': 33.28058}, {'x': 70.63343, 'y': 0.09149}, {'x': 59.74416, 'y': 33.44953}]","{'x': 89.12535, 'y': 25.66681, 'z': 8.61488}",10.95,706.98,1

from Ball import Ball
from Player import Player
import numpy as np
import sys
from typing import List

from rectangularToPolar import RectangularToPolar


class Moment:

    def __init__(self, jsonMomentArray):    
        self.jsonMomentArray = jsonMomentArray
        self.momentArray = []
        self.momentLabel = 0  # Initialize the label to 0

        players = jsonMomentArray[5][1:]  # Hardcoded position for players in json
        self.players = [Player(player) for player in players]

        ball = jsonMomentArray[5][0]  # Hardcoded position for ball in json
        self.ball = Ball(ball)
        self.shot_clock = jsonMomentArray[3]
        self.game_clock = jsonMomentArray[2]

        self.quarterNum = jsonMomentArray[0]

        self.offensiveSide : str =  None
        self.possessingTeamID : int = None
        self.leftHoop = np.array([5,25])
        self.rightHoop = np.array([89,25])


        


    def fillMomentFromJSON(self, currentTeamPossessionID  : int):

        
        # storing all 25 values into the moment Array (no label yet)
        
        defensiveTeamArray : List[float] = []

        isLeft = False
        for eachPlayer in self.players:
            polarObject : RectangularToPolar

            if self.offensiveSide == "Left":
                polarObject = RectangularToPolar(float(eachPlayer.x), float(eachPlayer.y) , self.leftHoop[0] , self.leftHoop[1])
                isLeft = True
            else:
                polarObject = RectangularToPolar(float(eachPlayer.x), float(eachPlayer.y) , self.rightHoop[0] , self.rightHoop[1])
                isLeft = False

            r , direction = polarObject.returnPolarCoordinates()

            if eachPlayer.team.id == currentTeamPossessionID:
                self.momentArray.append(r)
                self.momentArray.append(direction)
            else:
                defensiveTeamArray.append(r)
                defensiveTeamArray.append(direction)

        self.momentArray = self.momentArray + defensiveTeamArray

        # add ball location
        
    
        if isLeft:
            polarObject = RectangularToPolar(float(self.ball.x), float(self.ball.y) , self.leftHoop[0] , self.leftHoop[1])
            r , direction = polarObject.returnPolarCoordinates()
        else:
            polarObject = RectangularToPolar(float(self.ball.x), float(self.ball.y) , self.leftHoop[0] , self.leftHoop[1])
            r , direction = polarObject.returnPolarCoordinates()
        
        self.momentArray.append(r)
        self.momentArray.append(direction)
        self.momentArray.append(float(self.ball.radius))

        # add shot and game clock

        if self.shot_clock == None: 
            self.momentArray.append(float(24.0))
        else:
            self.momentArray.append(float(self.shot_clock))


        if self.game_clock == None: 
            self.momentArray.append(float(720.00))
        else:
            self.momentArray.append(float(self.game_clock))

    def whichSideIsOffensive(self) -> str:

        # halfcourt is at x = 47.0

        # count how many are less than 47.0
        
        counter = 0

        for eachPlayer in self.players:
            x = eachPlayer.x

            if x <= 47.0:
                counter +=1

        if counter >= 5:
            self.offensiveSide = "Left"
        else:
            self.offensiveSide = "Right"

    def whichTeamHasPossession(self):
        shortestDistance = sys.maxsize

        ball_x = self.ball.x
        ball_y = self.ball.y

        ball_point = np.array([ball_x,ball_y])
        self.possessingTeamID = None
        for eachPlayer in self.players:
            player_point = np.array([eachPlayer.x , eachPlayer.y])

            currDistance = np.linalg.norm(player_point-ball_point)

            if currDistance < shortestDistance:
                shortestDistance = currDistance
                self.possessingTeamID = eachPlayer.team.id


