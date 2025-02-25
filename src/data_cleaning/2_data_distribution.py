from collections import Counter
import pandas as pd

# Read the CSV file
df = pd.read_csv('data/data_suman/annotations.csv')

# Get the distribution
distribution = df['annotation'].value_counts()

print("Distribution of annotations:")
for annotation, count in distribution.items():
    print(f"{annotation}: {count} ({(count/len(df)*100):.1f}%)")