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

import pandas as pd
import numpy as np
from data_loading import find_user_recordings, load_segments

DATASET_ROOT = Path("c:/Shoab/Thesis/Experiments/dataset")
user_recordings = find_user_recordings(DATASET_ROOT)

for user, recs in user_recordings.items():
    left_sign_confs = []
    right_sign_confs = []
    left_bg_confs = []
    right_bg_confs = []
    
    for rec in recs:
        df = pd.read_csv(rec["csv_path"])
        segs = load_segments(rec["seg_path"])
        
        # Create a mask for sign frames
        sign_mask = np.zeros(len(df), dtype=bool)
        for s in segs:
            start = max(0, int(s["start"]))
            end = min(len(df) - 1, int(s["end"]))
            sign_mask[start:end+1] = True
            
        l_conf = df["left_confidence"].to_numpy()
        r_conf = df["right_confidence"].to_numpy()
        
        left_sign_confs.extend(l_conf[sign_mask])
        right_sign_confs.extend(r_conf[sign_mask])
        
        left_bg_confs.extend(l_conf[~sign_mask])
        right_bg_confs.extend(r_conf[~sign_mask])
        
    print(f"\nUser: {user}")
    print(f"  Sign Frames: {len(left_sign_confs)}")
    print(f"    Left Hand mean confidence: {np.mean(left_sign_confs):.4f}")
    print(f"    Right Hand mean confidence: {np.mean(right_sign_confs):.4f}")
    print(f"  Background Frames: {len(left_bg_confs)}")
    print(f"    Left Hand mean confidence: {np.mean(left_bg_confs):.4f}")
    print(f"    Right Hand mean confidence: {np.mean(right_bg_confs):.4f}")
