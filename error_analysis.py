import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
# Replace with your actual file paths if they are in a different directory
gold_df = pd.read_csv('data/test_labels_subtask1.csv')
pred_df = pd.read_csv('pred_subtask1.csv')

# 2. Merge the datasets on user_id and text_id
df = pd.merge(gold_df, pred_df, on=['user_id', 'text_id'])

# 3. Calculate Absolute Error (AE) for Valence and Arousal
df['val_error'] = np.abs(df['valence'] - df['pred_valence'])
df['aro_error'] = np.abs(df['arousal'] - df['pred_arousal'])

# Calculate text length (word count) to see if length affects performance
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

# ==========================================
# VISUALIZATION 1: Predicted vs. True Scatter
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Valence Plot
sns.scatterplot(data=df, x='valence', y='pred_valence', alpha=0.5, ax=axes[0])
axes[0].plot([df['valence'].min(), df['valence'].max()],
             [df['valence'].min(), df['valence'].max()],
             color='red', linestyle='--')  # Identity line (Perfect prediction)
axes[0].set_title('Valence: True vs. Predicted')
axes[0].set_xlabel('True Valence')
axes[0].set_ylabel('Predicted Valence')

# Arousal Plot
sns.scatterplot(data=df, x='arousal', y='pred_arousal', alpha=0.5, ax=axes[1])
axes[1].plot([df['arousal'].min(), df['arousal'].max()],
             [df['arousal'].min(), df['arousal'].max()],
             color='red', linestyle='--')
axes[1].set_title('Arousal: True vs. Predicted')
axes[1].set_xlabel('True Arousal')
axes[1].set_ylabel('Predicted Arousal')

plt.tight_layout()
plt.savefig('predicted_vs_true_scatter.png')
print("Saved scatter plot as 'predicted_vs_true_scatter.png'")

# ==========================================
# EXTRACTING THE WORST OFFENDERS
# ==========================================
# Get the top 20 worst predictions for qualitative analysis
top_val_errors = df.nlargest(20, 'val_error')[
    ['text', 'valence', 'pred_valence', 'val_error', 'text_length']]
top_aro_errors = df.nlargest(20, 'aro_error')[
    ['text', 'arousal', 'pred_arousal', 'aro_error', 'text_length']]

# Save to CSV so you can read them easily
top_val_errors.to_csv('worst_valence_predictions.csv', index=False)
top_aro_errors.to_csv('worst_arousal_predictions.csv', index=False)
print("Saved worst predictions to CSV for manual review.")

# ==========================================
# CORRELATION: Does text length matter?
# ==========================================
val_len_corr = df['val_error'].corr(df['text_length'])
aro_len_corr = df['aro_error'].corr(df['text_length'])

print(
    f"\nCorrelation between Text Length and Valence Error: {val_len_corr:.3f}")
print(f"Correlation between Text Length and Arousal Error: {aro_len_corr:.3f}")
