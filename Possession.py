from typing import List
from Moment import Moment
from TemporalWindow import TemporalWindow
# Starting Temporal Window Code

# numToName = {
#     0.0: "null",
#     1.0: "FG Try",
#     2.0: "Shoot F.",
#     3.0: "Nonshoot F.",
#     4.0: "Turnover"
# }

# nameToAvgPoints = {
#     "null" : 0,
#      "FG Try" : 1.25,
#      "Shoot F." : 0.7, 
#      "Nonshoot F." : 0.15,
#      "Turnover" : -1
# }

class Possession:

    def __init__(self):

        self.moments : List[Moment] = []

        self.temporalWindows = []


        self.terminalActionIndex = -1

        self.possessingTeamID : int

        self.homePossessionCounter = 0

        self.visitorPossessionCounter = 0

        self.pointsScored = 0



    def addMoment(self, moment):

        self.moments.append(moment)

    # def calculatePointsScored(self):

    #     if self.terminalActionIndex != -1:
    #         momentWithTerminalAction : Moment = self.moments[self.terminalActionIndex]
    #         labelNumber = momentWithTerminalAction.momentLabel
    #         labelName = numToName[labelNumber]
    #         labelPoints = nameToAvgPoints[labelName]
    #         self.pointsScored = labelPoints



