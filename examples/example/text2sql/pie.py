import matplotlib.pyplot as plt

# Data from SQL response
data = [
    ('Canada', 'Toronto', 2930000),
    ('China', 'Shanghai', 30000000),
    ('Japan', 'Tokyo', 13960000),
    ('South Korea', 'Seoul', 9776000),
    ('United States', 'Chicago', 2679000)
]

# Extract city names and populations
cities = [city for _, city, _ in data]
populations = [pop for _, _, pop in data]

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(populations, labels=cities, autopct='%1.1f%%', startangle=140)
plt.title('Population Distribution of Cities')
plt.show()