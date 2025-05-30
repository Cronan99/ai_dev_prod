# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv("ai_dev_productivity.csv")
# %%
df.head(10)
# %%
df.describe()
# %%
df.info()
# %%
df.isnull().sum()
# %%
plt.figure(figsize=(10, 6))
sns.lmplot(data=df, x='hours_coding', y='ai_usage_hours',
           scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('AI Usage vs. Hours Coding')
plt.xlabel('Hours Coding')
plt.ylabel('AI Usage Hours')
plt.grid()
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# %%
# Create bins for coffee intake
bin_width = 100  # Change as needed
df['coffee_bin'] = pd.cut(df['coffee_intake_mg'], bins=range(0, int(df['coffee_intake_mg'].max()) + bin_width, bin_width))

# Group by bins to get success rate and count
summary = df.groupby('coffee_bin').agg(
    success_rate=('task_success', 'mean'),
    count=('task_success', 'count')
).reset_index()

# Convert bins to strings for plotting
summary['coffee_bin_str'] = summary['coffee_bin'].astype(str)

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for count
sns.barplot(data=summary, x='coffee_bin_str', y='count', ax=ax1, alpha=0.3, color='gray')
ax1.set_ylabel('Count of Observations', color='gray')
ax1.tick_params(axis='y', labelcolor='gray')
ax1.set_xlabel('Coffee Intake (mg, binned)')

# Line plot for success rate
ax2 = ax1.twinx()
sns.lineplot(data=summary, x='coffee_bin_str', y='success_rate', ax=ax2, color='blue', marker='o')
ax2.set_ylabel('Task Success Rate', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Task Success Rate vs. Coffee Intake (with Observation Count)')
plt.xticks(rotation=45)

plt.grid()
plt.tight_layout()
plt.show()
# %%