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
# %%
# Grid Search and Hyperparameter Optimization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l1, l2, l1_l2
import time
from sklearn.model_selection import learning_curve

print("Starting Grid Search for hyperparameter optimization...")

# Define the hyperparameters to test
param_grid = {
    'hidden_layers': [
        [64, 32],          # Original architecture
        [128, 64],         # Wider network
        [64, 32, 16],      # Deeper network
        [128, 64, 32]      # Wider and deeper
    ],
    'regularization': [
        None,              # No regularization
        l2(0.001),         # L2 regularization (Ridge)
        l1(0.001),         # L1 regularization (Lasso)
        l1_l2(l1=0.001, l2=0.001)  # Combined L1 and L2 (Elastic Net)
    ],
    'dropout_rate': [0.0, 0.2, 0.3],
    'learning_rate': [0.01, 0.001],
    'batch_size': [16, 32, 64]
}

# Define metrics to track
results = []
best_val_auc = 0
best_model = None
best_params = None

# Set up k-fold cross-validation
n_folds = 3
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=5,
    restore_best_weights=True,
    mode='max'
)

# Tracking progress
total_combinations = (len(param_grid['hidden_layers']) * 
                     len(param_grid['regularization']) * 
                     len(param_grid['dropout_rate']) * 
                     len(param_grid['learning_rate']) * 
                     len(param_grid['batch_size']))
                     
print(f"Total combinations to test: {total_combinations}")
progress = 0
start_time = time.time()

# Loop through all hyperparameter combinations
for hidden_layers in param_grid['hidden_layers']:
    for reg in param_grid['regularization']:
        for dropout in param_grid['dropout_rate']:
            for lr in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    progress += 1
                    print(f"\nTesting combination {progress}/{total_combinations}")
                    print(f"Hidden layers: {hidden_layers}, Regularization: {type(reg).__name__ if reg else 'None'}, " +
                          f"Dropout: {dropout}, Learning rate: {lr}, Batch size: {batch_size}")
                    
                    # For cross-validation
                    fold_scores = []
                    
                    # Single fold for faster testing (comment out for full CV)
                    # Simplified for time - using validation split instead of full CV
                    # Build the model
                    model = tf.keras.Sequential()
                    model.add(tf.keras.Input(shape=(X_train_scaled.shape[1],)))
                    
                    # Add hidden layers with chosen architecture
                    for units in hidden_layers:
                        model.add(tf.keras.layers.Dense(
                            units, 
                            activation='relu',
                            kernel_regularizer=reg
                        ))
                        if dropout > 0:
                            model.add(tf.keras.layers.Dropout(dropout))
                            
                    # Output layer
                    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                    
                    # Compile model with chosen learning rate
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    model.compile(
                        optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
                    )
                    
                    # Train the model
                    history = model.fit(
                        X_train_scaled, 
                        y_train,
                        epochs=30,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Evaluate on test set
                    test_results = model.evaluate(X_test_scaled, y_test, verbose=0)
                    test_auc = test_results[2]  # AUC is the third metric
                    val_auc = max(history.history['val_auc'])
                    
                    # Record results
                    params = {
                        'hidden_layers': hidden_layers,
                        'regularization': type(reg).__name__ if reg else 'None',
                        'dropout_rate': dropout,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'test_loss': test_results[0],
                        'test_accuracy': test_results[1],
                        'test_auc': test_auc,
                        'val_auc': val_auc,
                        'epochs_trained': len(history.history['loss'])
                    }
                    results.append(params)
                    
                    print(f"  Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}")
                    
                    # Save if best model
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_model = model
                        best_params = params
                        print(f"  New best model found!")

elapsed_time = time.time() - start_time
print(f"\nGrid search completed in {elapsed_time/60:.2f} minutes")

# Print best results
print("\nBest model parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

print("\nTop 5 models by validation AUC:")
top_models = sorted(results, key=lambda x: x['val_auc'], reverse=True)[:5]
for i, model_params in enumerate(top_models):
    print(f"\n{i+1}. Val AUC: {model_params['val_auc']:.4f}, Test AUC: {model_params['test_auc']:.4f}")
    print(f"   Hidden layers: {model_params['hidden_layers']}")
    print(f"   Regularization: {model_params['regularization']}")
    print(f"   Dropout: {model_params['dropout_rate']}")
    print(f"   Learning rate: {model_params['learning_rate']}")
    print(f"   Batch size: {model_params['batch_size']}")

# %%
# Save the best model
best_model.save('best_model.keras')

# %%
# Visualize the performance of different hyperparameters

# Plot validation AUC vs test AUC
plt.figure(figsize=(10, 6))
plt.scatter([r['val_auc'] for r in results], [r['test_auc'] for r in results], alpha=0.7)
plt.xlabel('Validation AUC')
plt.ylabel('Test AUC')
plt.title('Validation AUC vs Test AUC for Different Hyperparameters')
plt.grid(True, linestyle='--', alpha=0.7)

# Add diagonal line for reference
min_auc = min(min([r['val_auc'] for r in results]), min([r['test_auc'] for r in results]))
max_auc = max(max([r['val_auc'] for r in results]), max([r['test_auc'] for r in results]))
plt.plot([min_auc, max_auc], [min_auc, max_auc], 'k--')
plt.tight_layout()
plt.show()




# %%
# Evaluate the best model
best_model_wrapper = KerasModelWrapper(best_model)
y_pred_proba_best = best_model.predict(X_test_scaled).ravel()
y_pred_classes_best = (y_pred_proba_best >= 0.5).astype(int)

# Classification report
print("Best Model Classification Report:\n", classification_report(y_test, y_pred_classes_best))

# ROC
fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_proba_best)
roc_auc_best = auc(fpr_best, tpr_best)

# Plot ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr_best, tpr_best, color='blue', lw=2, label=f'Best model ROC (AUC = {roc_auc_best:.2f})')
plt.plot(fpr, tpr, color='green', lw=2, linestyle=':', label=f'Original model ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Original vs Best Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()
# %% 
# Learning Curve for Best Model

print("Generating learning curve to check for overfitting...")

def create_model():
    """Recreate the best model architecture for learning curve evaluation"""
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X_train_scaled.shape[1],)))
    
    # Use the best model architecture from grid search results
    for units in best_params['hidden_layers']:
        reg = None
        if best_params['regularization'] == 'L1':
            reg = l1(0.001)
        elif best_params['regularization'] == 'L2':
            reg = l2(0.001)
        elif best_params['regularization'] == 'L1L2':
            reg = l1_l2(l1=0.001, l2=0.001)
            
        model.add(tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=reg))
        
        if best_params['dropout_rate'] > 0:
            model.add(tf.keras.layers.Dropout(best_params['dropout_rate']))
            
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Define training sizes
train_sizes = np.linspace(0.1, 1.0, 5)

# Training and validation scores for different training set sizes
train_scores = []
val_scores = []

X_train_val, X_val, y_train_val, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

for train_size in train_sizes:
    n_samples = int(len(X_train_val) * train_size)
    
    # Subset the training data
    X_subset = X_train_val[:n_samples]
    y_subset = y_train_val[:n_samples]
    
    # Train model on subset
    model = create_model()
    history = model.fit(
        X_subset, y_subset,
        epochs=30,
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val),
        verbose=0
    )
    
    # Get the best accuracy scores
    train_acc = max(history.history['accuracy'])
    val_acc = max(history.history['val_accuracy'])
    
    train_scores.append(train_acc)
    val_scores.append(val_acc)
    
    print(f"Training size: {train_size*100:.0f}% - Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', color='r', label='Training accuracy')
plt.plot(train_sizes, val_scores, 'o-', color='g', label='Validation accuracy')

plt.title('Learning Curve for Best Model')
plt.xlabel('Training Set Size (Proportion)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.ylim([min(min(train_scores), min(val_scores))-0.05, 1.05])

# Calculate the gap between training and validation scores
train_mean = np.mean(train_scores)
val_mean = np.mean(val_scores)
gap = train_mean - val_mean

# Add text annotation explaining the overfitting assessment
if gap > 0.1:
    assessment = "Significant overfitting (>10% gap)"
elif gap > 0.05:
    assessment = "Moderate overfitting (5-10% gap)"
else:
    assessment = "Minimal overfitting (<5% gap)"

plt.text(0.5, 0.2, f"Gap: {gap:.4f}\n{assessment}", 
         bbox=dict(facecolor='white', alpha=0.5),
         transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
# %%
