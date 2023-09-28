from typing import List
from Moment import Moment
from TemporalWindow import TemporalWindow
# Starting Temporal Window Code
class Possession:

    def __init__(self):

        self.moments : List[Moment] = []

        self.temporalWindows = []


        self.terminalActionIndex = -1

        self.possessingTeamID : int

        self.homePossessionCounter = 0

        self.visitorPossessionCounter = 0



    def addMoment(self, moment):

        self.moments.append(moment)