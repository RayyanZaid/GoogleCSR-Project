from typing import List
from Moment import Moment

class Possession:

    def __init__(self):

        self.moments : List[Moment] = []

        self.temporalWindows = []



    def addMoment(self, moment):

        self.moments.append(moment)