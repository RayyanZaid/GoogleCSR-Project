import matplotlib.pyplot as plt
import random
from Constant import Constant
from Moment import Moment
from Team import Team
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc

class Event:
    """A class for handling and showing events"""

    def __init__(self, event):
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                        player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        values = list(zip(player_names, player_jerseys))
        # Example: 101108: ['Chris Paul', '3']
        self.player_ids_dict = dict(zip(player_ids, values))

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)

            if moment.game_clock is not None and moment.shot_clock is not None:
                clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                            moment.quarter,
                            int(moment.game_clock) % 3600 // 60,
                            int(moment.game_clock) % 60,
                            moment.shot_clock)
            else:
                clock_test = "No time"
                
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle

    def update_bar_graph(self, i, bar_plot):
        # Update bar graph values randomly
        new_y_values = [random.uniform(0.0, 1.0) for _ in range(5)]
        for rect, new_height in zip(bar_plot.patches, new_y_values):
            rect.set_height(new_height)
        return bar_plot

    def show(self):
        # Set up the main subplot for the court and bar graph
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 2]})

        # Plot the basketball court
        court = plt.imread(r"D:\coding\GoogleCSR-Project\LiveResults\court.png")
        ax1.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        ax1.axis('off')

        # Plot the player circles and annotations
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict

        clock_info = ax1.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
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

        # Create the initial bar graph with random values
        y_values = [random.uniform(0.0, 1.0) for _ in range(5)]
        bar_plot = ax2.bar(range(5), y_values, tick_label=['0', '1', '2', '3', '4'])
        ax2.set_xlabel('X Labels')
        ax2.set_ylabel('Y Values')
        ax2.set_title('Randomized Bar Graph')

        # Animation
        anim = animation.FuncAnimation(
            fig, self.update_radius,
            fargs=(player_circles, ball_circle, annotations, clock_info),
            frames=len(self.moments), interval=Constant.INTERVAL)
        
        # Animation for updating the bar graph
        anim_bar_graph = animation.FuncAnimation(
            fig, self.update_bar_graph,
            fargs=(bar_plot,),
            frames=len(self.moments), interval=Constant.INTERVAL)

        plt.show()

# Example usage:
# event_data = # your event data here
# event = Event(event_data)
# event.show()
