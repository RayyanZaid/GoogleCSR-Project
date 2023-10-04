from Moment import Moment

from typing import List
# Spatio-temporal input to LSTM



class TemporalWindow:

    def __init__(self) -> None:
        
        self.moments : List[Moment] = []
        self.label : int
        