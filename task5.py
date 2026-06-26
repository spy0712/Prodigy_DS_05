import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Accident Dataset
data = {
    'Weather': ['Clear', 'Rainy', 'Foggy', 'Clear', 'Rainy',
                'Foggy', 'Clear', 'Rainy'],
    'Road_Condition': ['Dry', 'Wet', 'Wet', 'Dry',
                       'Wet', 'Wet', 'Dry', 'Wet'],
    'Time': ['Morning', 'Night', 'Evening', 'Afternoon',
             'Night', 'Morning', 'Evening', 'Night'],
    'Accidents': [15, 30, 25, 10, 35, 20, 12, 40]
}

df = pd.DataFrame(data)

# Style
sns.set_style("whitegrid")

# 1. Weather Impact
plt.figure(figsize=(8, 5))
sns.barplot(x='Weather', y='Accidents', data=df)
plt.title("Accidents by Weather Condition")
plt.tight_layout()
plt.savefig("weather_impact.png")
plt.close()

# 2. Road Condition Impact
plt.figure(figsize=(8, 5))
sns.barplot(x='Road_Condition', y='Accidents', data=df)
plt.title("Accidents by Road Condition")
plt.tight_layout()
plt.savefig("road_condition_impact.png")
plt.close()

# 3. Time of Day Impact
plt.figure(figsize=(8, 5))
sns.barplot(x='Time', y='Accidents', data=df)
plt.title("Accidents by Time of Day")
plt.tight_layout()
plt.savefig("time_impact.png")
plt.close()

# 4. Heatmap
pivot_table = pd.pivot_table(
    df,
    values='Accidents',
    index='Weather',
    columns='Road_Condition',
    aggfunc='mean'
)

# Replace NaN values
pivot_table = pivot_table.fillna(0)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt=".1f")
plt.title("Accident Hotspots")
plt.tight_layout()
plt.savefig("accident_hotspots.png")
plt.close()

print("PNG files created successfully!")