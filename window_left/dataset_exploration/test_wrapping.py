import pandas as pd
import numpy as np

df_ori = pd.read_csv("c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/palm_orientations.csv")

print("Left Hand Yaw statistics with different wrapping ranges:")
for user in df_ori["User"].unique():
    yaws = df_ori[df_ori["User"] == user]["Left_Yaw"].dropna().to_numpy()
    
    # Standard [-180, 180]
    mean_std = yaws.mean()
    std_std = yaws.std()
    
    # Wrapped to [0, 360]
    yaws_360 = np.mod(yaws, 360)
    mean_360 = yaws_360.mean()
    std_360 = yaws_360.std()
    
    # Wrapped to [-90, 270] (this shifts the wrapping point away from 180 to -90)
    yaws_shifted = np.mod(yaws + 90, 360) - 90
    mean_shifted = yaws_shifted.mean()
    std_shifted = yaws_shifted.std()
    
    print(f"\n{user}:")
    print(f"  Standard [-180, 180]: Mean = {mean_std:8.2f}, Std = {std_std:8.2f}")
    print(f"  Shifted  [-90, 270] : Mean = {mean_shifted:8.2f}, Std = {std_shifted:8.2f}")
    print(f"  Wrapped  [0, 360]   : Mean = {mean_360:8.2f}, Std = {std_360:8.2f}")
