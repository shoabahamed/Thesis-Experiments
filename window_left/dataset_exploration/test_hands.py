import pandas as pd
from pathlib import Path

users = ["user1", "user2", "user3", "user5"]
dataset_root = Path("c:/Shoab/Thesis/Experiments/dataset")

for user in users:
    user_dir = dataset_root / user / "leap_data"
    csv_files = list(user_dir.glob("*.csv"))
    
    total_left_valid = 0
    total_right_valid = 0
    total_frames = 0
    
    for f in csv_files[:10]: # check first 10 files
        df = pd.read_csv(f, usecols=['left_confidence', 'right_confidence', 'left_palm_x', 'right_palm_x'])
        total_frames += len(df)
        total_left_valid += ((df['left_confidence'] > 0) & (df['left_palm_x'] != 0)).sum()
        total_right_valid += ((df['right_confidence'] > 0) & (df['right_palm_x'] != 0)).sum()
        
    print(f"User: {user}")
    print(f"  Total frames checked: {total_frames}")
    print(f"  Left hand valid: {total_left_valid} ({total_left_valid/total_frames:.1%})")
    print(f"  Right hand valid: {total_right_valid} ({total_right_valid/total_frames:.1%})")
