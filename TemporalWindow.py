from Moment import Moment

from typing import List
# Spatio-temporal input to LSTM

T = 128 # 128 moments in a Temporal Window

class TemporalWindow:

    def __init__(self) -> None:
        
        self.moments : List[Moment] = []
        self.label : int
        