import pandas as pd

# Load datasets
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add label columns
fake_df['label'] = 0
true_df['label'] = 1

# Only keep necessary columns
fake_df = fake_df[['title', 'text', 'label']]
true_df = true_df[['title', 'text', 'label']]

# Combine and shuffle
combined_df = pd.concat([fake_df, true_df], ignore_index=True)
combined_df.dropna(subset=['text', 'label'], inplace=True)  # Remove any rows with missing text or label
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Save
combined_df.to_csv('data/fake_or_real_news.csv', index=False)

print("✅ Dataset is cleaned and saved as 'data/fake_or_real_news.csv'")
