import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('speeddating.csv', na_values=['?'])
print(f'Shape of the dataset: {df.shape}')
print()

print(df.info())
print()

print(df['match'].value_counts())
print()

print('Missing values statistics')
print(df.isna().sum().sort_values(ascending=False).head(30))