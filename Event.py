import matplotlib.pyplot as plt
import random
from Constant import Constant
from Moment import Moment
from Team import Team
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D
import numpy as np
from typing import List

from globals import WINDOW_SIZE, MOMENT_SIZE

from keras.models import load_model

model1 = load_model(r"1D_Conv_LSTM_v7")

mapping = {
    0.0: "null",
    1.0: "FG Try",
    2.0: "Shoot F.",
    3.0: "Nonshoot F.",
    4.0: "Turnover"
}

basePoints = 1.55
nullWeight = 0
FGTryWeight = 0.45
shootingFoulWeight = 0.6
nonShootingFoulWeight = 0.15
turnOverWeight = -1 * basePoints

def calculateExpectedPoints(nullProb, FGTryProb, shootFoulProb, nonshootFoulProb, turnoverProb):
    return basePoints + (nullProb * nullWeight) + (FGTryProb * FGTryWeight) + (shootFoulProb * shootingFoulWeight) + (nonshootFoulProb * nonShootingFoulWeight) + (turnoverProb * turnOverWeight)

# Goal: Read through each moment
def convertMomentstoModelInput(listOfMoments: List[List[float]], currentMomentArray: List[float]) -> List[List[float]]:
    if len(currentMomentArray) != MOMENT_SIZE:
        return listOfMoments

    while len(listOfMoments) >= WINDOW_SIZE:
        listOfMoments.pop(0)

    listOfMoments.append(currentMomentArray)
    np_list = np.array(listOfMoments)
    return listOfMoments

# only works if the length of the list of moments is equal to WINDOW_SIZE
def predict(listOfMoments: List[List[float]]):
    predictions = []

    # Check if the input data has the correct shape
    if len(listOfMoments) == WINDOW_SIZE:
        # Convert the list of moments to a numpy array and add a batch dimension

        # Predict with the model
        predictions = model1.predict([listOfMoments])
        predictions = predictions[0]
    else:
        predictions = [0.0, 0.0, 0.0, 0.0, 0.0]

    return predictions

class Event:
    def __init__(self, event):
        self.event = event
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        self.listOfMoments: List[List[float]] = []
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'], player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        values = list(zip(player_names, player_jerseys))
        self.player_ids_dict = dict(zip(player_ids, values))
        self.prev_i = -1
        self.prev_player_circles = None
        self.prev_ball_circle = None
        self.prev_bar_plot = None
        self.homeTeamID = event['home']['teamid']
        self.awayTeamID = event['visitor']['teamid']
        self.homeTeamPossessionCounter = 0
        self.awayTeamPossessionCounter = 0
        self.currentPossessionTeamID: int

        self.expectedPointsText = "Expected Points: "

        self.expectedPointsList = []  # Initialize an empty list to store expected points
        self.xValues = []
        

        

    def update_both(self, i, player_circles, ball_circle, annotations, clock_info, bar_plot, expectedPointsText,line_plot: Line2D,ax3):
        momentObject: Moment = self.moments[i]
        if i == self.prev_i:
            return self.prev_player_circles, self.prev_ball_circle, self.prev_bar_plot
        momentObject.whichTeamHasPossession()
        if momentObject.possessingTeamID == self.homeTeamID:
            self.homeTeamPossessionCounter += 1
        else:
            self.awayTeamPossessionCounter += 1
        if self.homeTeamPossessionCounter >= self.awayTeamPossessionCounter:
            self.currentPossessionTeamID = self.homeTeamID
        else:
            self.currentPossessionTeamID = self.awayTeamID
        momentObject.whichSideIsOffensive()
        momentObject.fillMomentFromJSON(self.currentPossessionTeamID)
        momentArray: List[float] = momentObject.momentArray
        self.listOfMoments: List[List[float]] = convertMomentstoModelInput(self.listOfMoments, momentArray)

        # Update player positions on the court
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)
        if moment.game_clock is not None and moment.shot_clock is not None:
            clock_text = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                moment.quarterNum,
                int(moment.game_clock) % 3600 // 60,
                int(moment.game_clock) % 60,
                moment.shot_clock
            )
        else:
            clock_text = "No time"
        clock_info.set_text(clock_text)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF

        # Update the bar graph
        predictions = predict(self.listOfMoments)


        expectedPoints = calculateExpectedPoints(predictions[0], predictions[1], predictions[2], predictions[3], predictions[4])
        
        
        self.expectedPointsList.append(expectedPoints)
        self.xValues.append(i)
       



        
        
        
        x_values = self.xValues

        lower_x_limit = max(0, i - WINDOW_SIZE)  # Calculate lower limit for x-values
        upper_x_limit = i  # Calculate upper limit for x-values

        x_data = x_values[lower_x_limit:upper_x_limit]  # Get x-data within the calculated limits
        y_data = self.expectedPointsList[lower_x_limit:upper_x_limit]  # Get corresponding y-data

        # Update the line plot
        line_plot[0].set_data(x_data, y_data)
        
        ax3.set_xlim(lower_x_limit, upper_x_limit)  # Set x-axis limits dynamically
        expectedPointsText.set_text(self.expectedPointsText + "{:.2f}".format(expectedPoints))

        

       
        for rect, new_height in zip(bar_plot.patches, predictions):
            rect.set_height(new_height)

        self.prev_i = i
        self.prev_player_circles = player_circles
        self.prev_ball_circle = ball_circle
        self.prev_bar_plot = bar_plot
        return player_circles, ball_circle, bar_plot, line_plot

    def show(self):
        fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1, 2]})
        plt.subplots_adjust(hspace=0.3)
        court = plt.imread(r"court.png")
        ax1.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        ax1.axis('off')

        y_values = [random.uniform(0.0, 1.0) for _ in range(5)]
        bar_plot = ax2.bar(range(5), y_values, tick_label=[mapping[0], mapping[1], mapping[2], mapping[3], mapping[4]])
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, axis='y')
        ax2.set_aspect('equal')
        # ax2.set_xlabel('Event Type')
        ax2.set_ylabel('Probability')
        ax2.set_title('Predicted Probabilities from LSTM')

        start_moment = self.moments[0]
        player_dict = self.player_ids_dict
        clock_info = ax1.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER], color='black', horizontalalignment='center',
                                 verticalalignment='center')
        annotations = [ax1.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                    horizontalalignment='center',
                                    verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]
        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax1.add_patch(circle)
        ax1.add_patch(ball_circle)



        y_values= self.expectedPointsList

        x_values = self.xValues

        line_plot = ax3.plot(x_values, y_values, linestyle='-')

        
        ax3.set_ylim(1.0, 2.0)
        ax3.grid(True, axis='y')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Expected points')
        ax3.set_title('Predicted Points during a Possession')
        
        # expected_points_text = ax3.annotate('', xy=(0.8, 1), color='black', fontsize=12, ha='center')
        expected_points_text = fig.text(0.5, 0.05, "", ha='center', fontsize=12, color='black')
        # Animation
        anim = animation.FuncAnimation(
    fig, self.update_both,
    fargs=(player_circles, ball_circle, annotations, clock_info, bar_plot, expected_points_text,line_plot,ax3),
    frames=len(self.moments), interval=Constant.INTERVAL)

        


        # Show the plot
        plt.show()
