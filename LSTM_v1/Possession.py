from typing import List
from Moment import Moment
from TemporalWindow import TemporalWindow
# Starting Temporal Window Code
class Possession:

    def __init__(self):

        self.moments : List[Moment] = []

        self.temporalWindows = []


        self.terminalActionIndex = -1



    def addMoment(self, moment):

        self.moments.append(moment)