from nba_api.stats.endpoints import playbyplay
# already have the game_id 

from enum import Enum

class EventMsgType(Enum):
    FIELD_GOAL_MADE = 1
    FIELD_GOAL_MISSED = 2
    FREE_THROWfree_throw_attempt = 3
    REBOUND = 4
    TURNOVER = 5
    FOUL = 6
    VIOLATION = 7
    SUBSTITUTION = 8
    TIMEOUT = 9
    JUMP_BALL = 10
    EJECTION = 11
    PERIOD_BEGIN = 12
    PERIOD_END = 13


class NBA_API_TerminalActions:

    def __init__(self, game_id):
        
        self.terminalActionsArray = []


game_id = "0021500524"

df = playbyplay.PlayByPlay(game_id).get_data_frames()[0]
head = df.head()

values = head.values
print(df.loc[df['EVENTMSGTYPE'] == 6]) #hint: use the EVENTMSGTYPE values above to see different data
print(values)
