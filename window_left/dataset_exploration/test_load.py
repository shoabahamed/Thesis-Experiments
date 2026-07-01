import pandas as pd
from pathlib import Path

csv_path = Path("c:/Shoab/Thesis/Experiments/dataset/user1/leap_data/P1_S10_R1.csv")
df = pd.read_csv(csv_path, nrows=5)
cols = list(df.columns)
print("Confidence columns:", [c for c in cols if 'confidence' in c])
print("Palm columns (subset):", [c for c in cols if 'palm' in c][:15])
print("Wrist columns:", [c for c in cols if 'wrist' in c])
print("Thumb columns (subset):", [c for c in cols if 'thumb' in c][:10])
