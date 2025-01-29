import pandas as pd


df = pd.read_csv('New-Online_Retail.csv')

# sample_df = df.sample(n=2000, random_state=42)

# sample_df.to_csv('New-Online_Retail.csv', index=False)

print(df.columns)
