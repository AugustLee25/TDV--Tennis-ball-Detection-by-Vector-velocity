# -*- coding: utf-8 -*-
# example_usage.py — vi du ngan nhat, chay duoc ngay: python example_usage.py
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from tennis_shot_api import ShotDetector

# ---------------------------------------------------------------------------
# 1. KHOI TAO
# ---------------------------------------------------------------------------
det = ShotDetector(
    model_path=os.path.join("models", "lgbm_v10_savgol.joblib"),
    conf=0.6,               # nguong xac suat de ghi nhan su kien
    fps=30.0,               # de quy doi frame_idx -> giay
    cooldown_s=0.0,         # vi du nay replay tung quy dao roi rac nen tat cooldown;
                            # chay video that thi de 0.20
    frame_size=None,        # None = toa do chuan hoa 0..1; (1920,1080) = pixel
    max_window_span=8,      # bo cua so trai qua 8 khung hinh (phan lon la noi suy)
    court_mask=None,        # hoac "court_mask.json" neu muon loc khong gian
)

# ---------------------------------------------------------------------------
# 2. DUA TOA DO BONG VAO, TUNG KHUNG HINH MOT
# ---------------------------------------------------------------------------
# O day lay quy dao that tu data_test.csv lam vi du. Trong pipeline cua ban,
# (x, y) den tu detector/tracker cua ban.
import model_benchmark_V10_features as V10
te = pd.read_csv(os.path.join("reference", "data_test.csv"))
WX, WY = V10.rebuild_window(te)

NAMES = {0: "Normal", 1: "Racket", 2: "Ground"}
print()
print("Chay thu tren 12 quy dao dau cua data_test.csv")
print("%-6s%-34s%-12s%-10s%s" % ("mau", "cac lan push(frame, x, y)", "ket qua", "p", "nhan that"))
print("-" * 86)
n_ok = 0
for i in range(12):
    det.reset()                       # moi quy dao la mot doan doc lap -> reset
    ev = None
    for k in range(5):
        r = det.push(frame_idx=k, x=float(WX[i, k]), y=float(WY[i, k]))
        if r: ev = r
    truth = NAMES[int(te.label.iloc[i])]
    got = ev["label_name"] if ev else "(khong ban)"
    p = "%.3f" % ev["confidence"] if ev else "-"
    hit = (got == truth) or (ev is None and truth == "Normal")
    n_ok += hit
    print("%-6d%-34s%-12s%-10s%s%s" % (
        i, "5 diem quanh khung hinh %d" % i, got, p, truth, "" if hit else "   <-- lech"))
print("-" * 86)
print("Dung %d/12" % n_ok)
print()
print("Mau 4 khong phai model doan SAI: no doan DUNG Racket nhung p=0.5021, duoi")
print("nguong conf=0.6 nen bi chan. Do la danh doi co chu y — ha conf xuong 0.5 thi")
print("bat duoc no, doi lai them su kien gia. Cau hinh production dang de 0.6.")

# ---------------------------------------------------------------------------
# 3. THONG KE
# ---------------------------------------------------------------------------
print()
for k, v in det.stats().items():
    print("   %-30s %s" % (k, v))

print()
print("GHI NHO:")
print("  * push() tra ve su kien cua khung hinh i khi ban day vao diem i+2 (tre 2 khung).")
print("  * Goi det.reset() moi khi CAT CANH, neu khong se sinh su kien gia.")
print("  * Bo qua khung hinh khong bat duoc bong — dung tu noi suy roi day vao.")
print("  * Truyen toa do o float64, dung ep xuong float32.")
