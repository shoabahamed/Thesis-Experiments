import pandas as pd
from pathlib import Path
import numpy as np

users = ["user1", "user2", "user3", "user5"]
dataset_root = Path("c:/Shoab/Thesis/Experiments/dataset")

for user in users:
    user_dir = dataset_root / user / "leap_data"
    csv_files = list(user_dir.glob("*.csv"))
    
    total_gaps = 0
    total_no_hands = 0
    total_rows = 0
    
    for f in csv_files[:5]:
        df = pd.read_csv(f)
        total_rows += len(df)
        
        # Check gaps in frame_number
        if 'frame_number' in df.columns:
            fn = df['frame_number'].to_numpy()
            diffs = np.diff(fn)
            # any diff > 1 means a gap
            gaps = (diffs - 1).sum()
            total_gaps += gaps
            
        # Check frames where both hands are missing/invalid
        # We can check left_confidence == 0 and right_confidence == 0
        left_val = (df['left_confidence'] > 0) & (df['left_palm_x'] != 0)
        right_val = (df['right_confidence'] > 0) & (df['right_palm_x'] != 0)
        no_hands = (~left_val) & (~right_val)
        total_no_hands += no_hands.sum()
        
    print(f"User: {user}")
    print(f"  Total rows: {total_rows}")
    print(f"  Gaps in frame_number: {total_gaps}")
    print(f"  Frames with no hands active: {total_no_hands}")
