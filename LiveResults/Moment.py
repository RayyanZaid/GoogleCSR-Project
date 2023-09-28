from Ball import Ball
from Player import Player

class Moment:
    """A class for keeping info about the jsonMomentArrays"""
    def __init__(self, jsonMomentArray):
        self.quarter = jsonMomentArray[0]  # Hardcoded position for quarter in json
        self.game_clock = jsonMomentArray[2]  # Hardcoded position for game_clock in json
        print(self.game_clock)
        self.shot_clock = jsonMomentArray[3]  # Hardcoded position for shot_clock in json
        ball = jsonMomentArray[5][0]  # Hardcoded position for ball in json
        self.ball = Ball(ball)
        players = jsonMomentArray[5][1:]  # Hardcoded position for players in json
        self.players = [Player(player) for player in players]
        self.momentArray = []

    def fillMomentFromJSON(self):

        
        # storing all 25 values into the moment Array (no label yet)

        for eachPlayer in self.players:
            self.momentArray.append(float(eachPlayer.x))
            self.momentArray.append(float(eachPlayer.y))

        # add ball location
        
        self.momentArray.append(float(self.ball.x))
        self.momentArray.append(float(self.ball.y))
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
            return "Left"
        else:
            return "Right"