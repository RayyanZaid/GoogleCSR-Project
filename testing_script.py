# PURPOSE : Display live probabilities

# Goals

    # 1) Read through JSON
    # 2) Display Player and Ball Trajectories
    # 3) Display the probability of each terminal action


from Game import Game
import argparse
import sys
import json


# 20 
# 82

if sys.gettrace() is not None:  # Check if debugger is active
    args = argparse.Namespace(path=r"0021500492.json" , event = 25, model = "1D_Conv_LSTM_v8")  # Provide default values for debugging
else:
    parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')
    parser.add_argument('--path', type=str,
                        help='a path to json file to read the events from',
                        required=True)
    parser.add_argument('--event', type=int, default=0,
                        help="""an index of the event to create the animation to
                                (the indexing start with zero, if you index goes beyond out
                                the total number of events (plays), it will show you the last
                                one of the game)""")
    
    parser.add_argument('--model', type=str, default=0,
                        help="""the name of the model you want to test""")
    
    args = parser.parse_args()

game = Game(path_to_json=args.path, event_index=args.event, modelName = args.model)
game.read_json()

game.start()
