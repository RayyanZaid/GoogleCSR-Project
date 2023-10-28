import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Create the figure and subplots with a layout of 2 rows and 2 columns
fig, (ax1, ax2, ax3) = plt.subplots(2, 2, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})

# Plot the basketball court on ax1
court = plt.imread(r"court.png")
ax1.imshow(court, zorder=0, extent=[0,1,0,1,0])
ax1.axis('off')

# Create the initial bar graph with random values on ax2
y_values = [random.uniform(0.0, 1.0) for _ in range(5)]
bar_plot = ax2.bar(range(5), y_values, tick_label=["1","2","3","4","5"])

# Set the y-axis limits for the bar graph from 0.0 to 1.0
ax2.set_ylim(0.0, 1.0)

# Add grid lines to the bar graph
ax2.grid(True, axis='y')
ax2.set_aspect('equal')
ax2.set_xlabel('Event Type')
ax2.set_ylabel('Probability')
ax2.set_title('Predicted Probabilities from LSTM')

# Rotate the x-axis labels for better readability
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# Create a line graph with grid on ax3
x_values = []
y_values_line = []

max_length = 12
def update_line_graph(i):
    x_values.append(i*0.25)
    y_values_line.append(random.uniform(0.0, 1.0))

    if len(x_values) > max_length:
        # Shift to the left to maintain the maximum length
        x_values.pop(0)
        y_values_line.pop(0)
    ax3.clear()
    ax3.plot(x_values, y_values_line)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Value')
    ax3.set_title('Random Line Graph with Grid')
    ax3.set_aspect('equal')  # Set the aspect ratio to 'equal' for both axes

# Set aspect ratio of ax3 to be equal, making it square
line_animation = animation.FuncAnimation(fig, update_line_graph, blit=False, interval=250)

# Show the plot (or save it to a file if needed)
plt.show()