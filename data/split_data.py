import pandas as pd


df = pd.read_excel('Online-Retail.xlsx')

sample_df = df.sample(n=2000, random_state=42)

sample_df.to_csv('New-Online_Retail.csv', index=False)

print(sample_df)