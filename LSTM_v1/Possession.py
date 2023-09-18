from typing import List
from Moment import Moment

# Starting Temporal Window Code
class Possession:

    def __init__(self):

        self.moments : List[Moment] = []

        self.temporalWindows = []



    def addMoment(self, moment):

        self.moments.append(moment)