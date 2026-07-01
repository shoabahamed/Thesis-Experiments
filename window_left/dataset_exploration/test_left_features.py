import sys
from pathlib import Path
from unittest.mock import MagicMock

# Custom dummy torch module
class DummyDevice:
    def __init__(self, *args, **kwargs): pass
    def __str__(self): return "cpu"
    def __repr__(self): return "device(type='cpu')"

class DummyTensor: pass

class DummyTorch:
    Tensor = DummyTensor
    device = DummyDevice
    @staticmethod
    def manual_seed(seed): pass
    class cuda:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def manual_seed_all(seed): pass
    class backends:
        class cudnn:
            deterministic = True
            benchmark = False

sys.modules['torch'] = DummyTorch
sys.path.append(str(Path(__file__).resolve().parents[1] / "tchct_net"))

import numpy as np
import pandas as pd

from config import FEATURE_KEYS, FEATURE_INDEX, PALM_TRIPLETS, HANDS
from features import palm_reference_normalize_sequence

DATASET_ROOT = Path("c:/Shoab/Thesis/Experiments/dataset")
USERS = ["user1", "user2", "user3", "user5"]

left_samples = {u: [] for u in USERS}

for user in USERS:
    user_dir = DATASET_ROOT / user / "leap_data"
    csv_files = sorted(list(user_dir.glob("*.csv")))
    
    for f_path in csv_files:
        df = pd.read_csv(f_path)
        if len(df) == 0:
            continue
            
        left_visible = (df["left_confidence"] > 0) & (df["left_palm_x"] != 0)
        val_indices = np.where(left_visible)[0]
        if len(val_indices) == 0:
            continue
            
        raw_features = df.reindex(columns=FEATURE_KEYS).fillna(0.0).to_numpy(dtype=np.float32)
        norm_features = palm_reference_normalize_sequence(raw_features)
        
        # Subsample to at most 150 frames
        if len(val_indices) > 150:
            sampled_idx = np.random.choice(val_indices, size=150, replace=False)
        else:
            sampled_idx = val_indices
            
        for idx in sampled_idx:
            frame_data = norm_features[idx]
            feat_sample = {
                "left_Index Tip_x": frame_data[FEATURE_INDEX["left_index_distal_sx"]],
                "left_Index Tip_y": frame_data[FEATURE_INDEX["left_index_distal_sy"]],
                "left_Index Tip_z": frame_data[FEATURE_INDEX["left_index_distal_sz"]],
                "left_Wrist_x": frame_data[FEATURE_INDEX["left_wrist_x"]],
                "left_Wrist_y": frame_data[FEATURE_INDEX["left_wrist_y"]],
                "left_Wrist_z": frame_data[FEATURE_INDEX["left_wrist_z"]],
            }
            left_samples[user].append(feat_sample)

print("Left Hand Normalized Feature Averages:")
for user in USERS:
    temp_df = pd.DataFrame(left_samples[user])
    print(f"\nUser: {user} (N = {len(temp_df)})")
    if len(temp_df) > 0:
        print(temp_df.mean().to_string())
