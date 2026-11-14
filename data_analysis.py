
import pandas as pd
import matplotlib.pyplot as plt

# Road damage data
data = {
    "Damage Type": ["Pothole", "Crack"],
    "Count": [120, 80]
}

df = pd.DataFrame(data)

# Total damage detections
total_damage = df["Count"].sum()

print("AI ROAD DAMAGE ANALYSIS")
print("-----------------------")

print("Total Damage Detections:", total_damage)

# Damage statistics
print("\nDamage Statistics:")
print(df)

# Calculate percentage
df["Percentage"] = (
    df["Count"] / total_damage
) * 100

print("\nDamage Percentage:")
print(df)

# Most common damage
most_common = df.loc[
    df["Count"].idxmax(), "Damage Type"
]

print("\nMost Common Damage:", most_common)

# Bar Chart
plt.figure(figsize=(8, 5))

plt.bar(
    df["Damage Type"],
    df["Count"]
)

plt.title("Road Damage Detection Analysis")
plt.xlabel("Damage Type")
plt.ylabel("Number of Detections")

plt.tight_layout()
plt.savefig("road_damage_chart.png")
plt.show()

# Pie Chart
plt.figure(figsize=(7, 7))

plt.pie(
    df["Count"],
    labels=df["Damage Type"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Road Damage Distribution")

plt.savefig("damage_distribution.png")
plt.show()

# Save CSV report
df.to_csv(
    "road_damage_report.csv",
    index=False
)

print("\nAnalysis Report Generated Successfully!")

