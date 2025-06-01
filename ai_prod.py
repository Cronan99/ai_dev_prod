# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
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
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
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
# Create bins for coffee intake
bin_width = 70  # 40mg is one cup of coffee
df['coffee_bin'] = pd.cut(df['coffee_intake_mg'], bins=range(0, int(df['coffee_intake_mg'].max()) + bin_width, bin_width))

# Group by bins to get success rate and count
summary = df.groupby('coffee_bin').agg(
    success_rate=('task_success', 'mean'),
    count=('task_success', 'count')
).reset_index()
# Calculate number of cups for labeling
def bin_to_cups(bin_interval):
    start_cups = bin_interval.left // 70
    end_cups = bin_interval.right // 70
    if end_cups - start_cups == 1:
        return f"{start_cups} cup"
    else:
        return f"{start_cups}-{end_cups - 1} cups"

summary['coffee_bin_str'] = summary['coffee_bin'].apply(bin_to_cups)

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
X = df.drop(columns=['task_success', "coffee_bin"])
y = df['task_success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# %%
model.compile(optimizer="adam", 
              loss="binary_crossentropy", 
              metrics=["accuracy"])
# %%
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)
model.save('my_model.keras')
# %%
# Get probabilities
y_pred_proba = model.predict(X_test_scaled).ravel()

# Threshold at 0.5 for classification
y_pred_classes = (y_pred_proba >= 0.5).astype(int)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred_classes))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# %%
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
# %%
class KerasModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])  # <-- Add this line

    def fit(self, X, y):
        pass  # Already trained

    def predict(self, X):
        return (self.model.predict(X) >= 0.5).astype(int).ravel()

    def predict_proba(self, X):
        preds = self.model.predict(X)
        return np.hstack([(1 - preds), preds]) if preds.ndim == 2 and preds.shape[1] == 1 else preds

# %%
# Create a wrapper for the Keras model
keras_model = KerasModelWrapper(model)
# %%
# Calculate permutation importance
result = permutation_importance(
    keras_model,
    X_test_scaled,
    y_test,
    scoring='f1',
    n_repeats=30,
    random_state=42
)

importance_means = result.importances_mean
sorted_idx = importance_means.argsort()

print("Importance means:", importance_means)
print("Sorted indices:", sorted_idx)

plt.barh(range(len(sorted_idx)), importance_means[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Permutation Importance")
plt.show()
# %%