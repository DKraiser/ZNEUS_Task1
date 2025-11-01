import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import re

def map_field(field):
    if pd.isna(field):
        return 'Unknown'
    field = field.lower()
    if any(x in field for x in ['math', 'engineering', 'physics', 'biomed', 'computational']):
        return 'STEM'
    elif any(x in field for x in ['economics', 'finance', 'business', 'mba']):
        return 'Economics/Business'
    elif any(x in field for x in ['political', 'sociology', 'psychology', 'social work', 'international affairs', 'anthropology']):
        return 'Social Sciences'
    elif any(x in field for x in ['law', 'public policy', 'administration']):
        return 'Law/Public Policy'
    elif any(x in field for x in ['education', 'teaching', 'speech', 'tesol']):
        return 'Education'
    elif any(x in field for x in ['art', 'music', 'literature', 'creative', 'writing', 'philosophy']):
        return 'Arts/Humanities'
    elif any(x in field for x in ['medicine', 'health', 'biolog', 'epidemiology', 'nutrition', 'neuroscience']):
        return 'Health/Medicine'
    elif any(x in field for x in ['language', 'international', 'cultural']):
        return 'Language/International'
    else:
        return 'Other'

def parse_mean(s):
        if pd.isna(s):
            return None
        match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)', str(s))
        if match:
            return (float(match.group(1)) + float(match.group(2))) / 2
        else:
            return None
        
file_path = os.path.join(os.path.dirname(__file__), 'speeddating.csv')
df = pd.read_csv(file_path, na_values=['?'])
print(f'Shape of the dataset: {df.shape}')
print()

print(df.info())
print()

print(df['match'].value_counts())
print()

print('Missing values statistics')
print(df.isna().sum().sort_values(ascending=False).head(30))


# Data preprocessing
# Ignores
df = df.drop(columns=['decision', 'decision_o'])
df['field'] = df['field'].fillna('Unknown').apply(map_field)

# Default num vals
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# Ranges [a-b]
range_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.contains(r'\[.*-.*\]', regex=True, na=False).any()]
# Categorical values - all that are not in others
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and not col in range_cols]

print(f'Numeric cols: {len(num_cols)}')
print(f'Range cols: {len(range_cols)}')
print(f'Categorical cols: {len(categorical_cols)}')
print(f'Total cols num: {len(num_cols) + len(range_cols) + len(categorical_cols)}')

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Correlations
for col in range_cols:
    df[col] = df[col].apply(parse_mean)
    df[col] = df[col].fillna(df[col].median())

corr = df.corr()
match_corr = corr['match'].sort_values(key=abs, ascending=False)
print("Top correlations with 'match':\n", match_corr[1:].head(50))

similarity_threshold = 0.8
high_corr = []

for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > similarity_threshold:
            col1 = corr.columns[i]
            col2 = corr.columns[j]
            high_corr.append((col1, col2, corr.iloc[i, j]))

sorted_data = sorted(high_corr, key=lambda tup: tup[2], reverse=True)
for col1, col2, corr_value in sorted_data:
    print(f"{col1} and {col2} are highly correlated: {corr_value:.2f}")
    
#Drop useless data
dropped = set()
for c1, c2, p in high_corr:
    dropped.add(c2)

match_corr_threshold = 0.1
low_corr_cols = match_corr[abs(match_corr) < match_corr_threshold].index.tolist()
for col in low_corr_cols:
    if col != 'match':
        dropped.add(col)

df = df.drop(columns=list(dropped), errors='ignore')