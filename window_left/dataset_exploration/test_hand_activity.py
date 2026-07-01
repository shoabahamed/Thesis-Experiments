import pandas as pd
from pathlib import Path
import numpy as np

users = ["user1", "user2", "user3", "user5"]
dataset_root = Path("c:/Shoab/Thesis/Experiments/dataset")

for user in users:
    user_dir = dataset_root / user / "leap_data"
    csv_files = list(user_dir.glob("*.csv"))
    
    total_frames = 0
    left_active = 0
    right_active = 0
    both_active = 0
    neither_active = 0
    
    for f in csv_files:
        df = pd.read_csv(f)
        total_frames += len(df)
        
        left_val = (df['left_confidence'] > 0) & (df['left_palm_x'] != 0)
        right_val = (df['right_confidence'] > 0) & (df['right_palm_x'] != 0)
        
        left_active += left_val.sum()
        right_active += right_val.sum()
        both_active += (left_val & right_val).sum()
        neither_active += (~left_val & ~right_val).sum()
        
    print(f"User: {user}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Left Hand Active:  {left_active} ({left_active/total_frames*100:.2f}%)")
    print(f"  Right Hand Active: {right_active} ({right_active/total_frames*100:.2f}%)")
    print(f"  Both Active:       {both_active} ({both_active/total_frames*100:.2f}%)")
    print(f"  Neither Active:    {neither_active} ({neither_active/total_frames*100:.2f}%)")
