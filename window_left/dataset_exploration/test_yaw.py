import pandas as pd
import numpy as np
from pathlib import Path

df_ori = pd.read_csv("c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/palm_orientations.csv")

for user in df_ori["User"].unique():
    user_data = df_ori[df_ori["User"] == user]["Left_Yaw"].dropna()
    print(f"\nUser: {user} Left Hand Yaw (deg)")
    print(f"  Count:  {len(user_data)}")
    print(f"  Mean:   {user_data.mean():.2f}")
    print(f"  Std:    {user_data.std():.2f}")
    print(f"  Min:    {user_data.min():.2f}")
    print(f"  25%:    {user_data.quantile(0.25):.2f}")
    print(f"  50%:    {user_data.median():.2f}")
    print(f"  75%:    {user_data.quantile(0.75):.2f}")
    print(f"  Max:    {user_data.max():.2f}")
