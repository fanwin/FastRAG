import plotly.express as px
import pandas as pd

# Data from SQL response
data = {
    'Country': ['Canada', 'China', 'Japan', 'South Korea', 'United States'],
    'City': ['Toronto', 'Shanghai', 'Tokyo', 'Seoul', 'Chicago'],
    'Population': [2930000, 30000000, 13960000, 9776000, 2679000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a pie chart
fig = px.pie(df, values='Population', names='City',
             title='Population Distribution by City',
             hover_data=['City', 'Population'],
             labels={'City': 'City Name', 'Population': 'Population'},
             hole=0.3)  # Add a hole for a donut-like appearance

# Add animation for the pie chart sections
fig.update_traces(pull=[0.1, 0, 0, 0, 0])  # Pull the first section (Toronto) slightly

# Show the plot
fig.show()