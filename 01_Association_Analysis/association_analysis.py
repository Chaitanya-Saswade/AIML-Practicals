import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 45, 50, 55, 62, 68, 75, 82, 90]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Calculate correlation
correlation = df["Study_Hours"].corr(df["Marks"])

print("\nCorrelation between Study Hours and Marks:")
print(round(correlation, 2))

# Interpret correlation
if correlation > 0:
    print("There is a positive association between Study Hours and Marks.")
elif correlation < 0:
    print("There is a negative association between Study Hours and Marks.")
else:
    print("There is no linear association.")

# Scatter plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x="Study_Hours",
    y="Marks",
    data=df
)

plt.title("Association Between Study Hours and Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()