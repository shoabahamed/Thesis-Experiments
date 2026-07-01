import pandas as pd
import numpy as np
from pathlib import Path

df_ori = pd.read_csv("c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/palm_orientations.csv")
u1_ori = df_ori[df_ori["User"] == "User1 (user1)"].dropna()

print("User1 Left Hand Yaw per Sequence Sample (first 15 sequences):")
for idx, row in u1_ori.head(15).iterrows():
    print(f"  Sequence {row['Sequence']}: Left Yaw Mean = {row['Left_Roll']:.2f} Roll, {row['Left_Pitch']:.2f} Pitch, {row['Left_Yaw']:.2f} Yaw")
    
# Let's inspect frame-by-frame yaw for a couple of sequences to see if there is frame-level flipping
dataset_root = Path("c:/Shoab/Thesis/Experiments/dataset")
u1_csvs = sorted(list((dataset_root / "user1" / "leap_data").glob("*.csv")))

for f_path in u1_csvs[:3]:
    df = pd.read_csv(f_path)
    left_visible = (df["left_confidence"] > 0) & (df["left_palm_x"] != 0)
    ldx, ldz = df.loc[left_visible, "left_palm_dx"], df.loc[left_visible, "left_palm_dz"]
    yaws = np.degrees(np.arctan2(ldx, ldz))
    
    print(f"\nSequence: {f_path.stem}")
    print(f"  Total valid frames: {len(yaws)}")
    print(f"  Yaw Mean:           {np.mean(yaws):.2f}")
    print(f"  Yaw Std:            {np.std(yaws):.2f}")
    print(f"  Yaw Min:            {np.min(yaws):.2f}")
    print(f"  Yaw Max:            {np.max(yaws):.2f}")
