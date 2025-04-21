import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.animation import FuncAnimation

# Data from SQL response
data = [
    ('Canada', 'Toronto', 2930000),
    ('China', 'Shanghai', 30000000),
    ('Japan', 'Tokyo', 13960000),
    ('South Korea', 'Seoul', 9776000),
    ('United States', 'Chicago', 2679000)
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Country', 'City', 'Population'])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Function to update the pie chart
def update(frame):
    ax.clear()
    ax.set_title(f'Population Distribution (Frame {frame})')
    ax.pie(df['Population'], labels=df['City'], autopct='%1.1f%%', startangle=90 + frame * 10)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Create an animation
ani = FuncAnimation(fig, update, frames=range(36), interval=100)

# Save the animation as a GIF
ani.save('population_distribution.gif', writer='pillow')

# Show the plot (optional)
plt.show()