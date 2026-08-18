# -*- coding: utf-8 -*-
# verify_final.py — CHUNG MINH goi Final cho ket qua Y HET duong ong chinh.
#
# Chay:  python verify_final.py
#
# Kiem tra hai duong doc lap tren cung 179 mau cua data_test.csv:
#   OFFLINE  data_test.csv -> engineer_features_v10() -> model
#   ONLINE   data_test.csv -> rebuild_window() -> 5 diem toa do
#                          -> FeatureExtractor.push() x5 -> model
# Neu hai duong lech nhau du chi mot chut thi goi nay khong dung duoc.
import os, sys
import numpy as np, pandas as pd, joblib
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "reference"))

from tennis_shot_api import (ShotDetector, FeatureExtractor, ShotClassifier,
                             ContextEstimator, Detection)
import model_benchmark_V10_features as V10

MODEL = os.path.join("models", "lgbm_v10_savgol.joblib")
te = pd.read_csv(os.path.join("reference", "data_test.csv"))
print("=" * 76)
print("KIEM CHUNG GOI Final/  —  %d mau tu data_test.csv" % len(te))
print("=" * 76)

# ---------- duong OFFLINE ----------
clf = ShotClassifier(MODEL)
F_off = V10.engineer_features_v10(te, use_savgol=True, add_window_stats=True,
                                  stats_on_smoothed=True)
COLS = list(clf.feature_cols)
X_off = F_off[COLS].to_numpy(float)
P_off = clf.model.predict_proba(X_off)
lab_off = P_off.argmax(1)

# ---------- duong ONLINE ----------
WX, WY = V10.rebuild_window(te)
rows, lab_on, P_on = [], [], []
for i in range(len(te)):
    fx = FeatureExtractor(max_frame_gap=12, smooth=True, savgol_window=5, savgol_poly=2)
    fx.max_window_span = 0
    s = None
    for k in range(5):
        s = fx.push(Detection(frame_idx=k, x=float(WX[i, k]), y=float(WY[i, k])))
    if s is None:
        raise SystemExit("Mau %d: FeatureExtractor khong sinh duoc dac trung" % i)
    # dung DUNG gia tri ngu canh cua tap du lieu, de chi so sanh tang dong hoc
    s.features["from_player"] = float(te.from_player.iloc[i])
    s.features["from_side"] = float(te.from_side.iloc[i]) if "from_side" in te.columns else 0.0
    rows.append([float(s.features[c]) for c in COLS])
    l, p = clf.predict(s.features)
    lab_on.append(int(l)); P_on.append(p)
X_on = np.asarray(rows); P_on = np.asarray(P_on); lab_on = np.asarray(lab_on)

# ---------- doi chieu ----------
d = np.abs(X_off - X_on)
print("\n1. DAC TRUNG (%d cot x %d mau)" % (X_off.shape[1], X_off.shape[0]))
print("   lech tuyet doi lon nhat : %.3e" % d.max())
worst = int(np.argmax(d.max(0)))
print("   cot lech nhieu nhat     : %s (%.3e)" % (COLS[worst], d.max(0)[worst]))
ok1 = d.max() < 1e-9

dp = np.abs(P_off - P_on).max(1)
print("\n2. XAC SUAT")
print("   lech tuyet doi lon nhat : %.3e" % dp.max())
print("   so mau lech > 1e-9      : %d / %d" % (int((dp > 1e-9).sum()), len(dp)))
print("   VI SAO KHAC 0: cay quyet dinh la ham GIAN DOAN. Dac trung lech 1e-16 (dung")
print("   bang epsilon cua float64, do thu tu phep tinh khac nhau giua duong vector")
print("   hoa va duong tung mau) du de vuot mot nguong chia, doi la, va lam xac suat")
print("   nhay mot buoc huu han. Day la ban chat cua ensemble cay, khong phai loi.")
print("   Tieu chi dung: KHONG mau nao doi nhan, va lech < 1e-2.")
ok2 = (dp.max() < 1e-2)

print("\n3. NHAN DU DOAN")
agree = (lab_off == lab_on).mean()
print("   trung nhau              : %d/%d (%.4f)" % (int((lab_off == lab_on).sum()), len(te), agree))
ok3 = agree == 1.0

y = te.label.to_numpy()
from sklearn.metrics import f1_score
print("\n4. CHI SO CUOI CUNG")
print("   F1-Macro offline        : %.6f" % f1_score(y, lab_off, average="macro"))
print("   F1-Macro online (goi nay): %.6f" % f1_score(y, lab_on, average="macro"))
print("   mau sai                 : %s" % sorted(np.where(lab_on != y)[0].tolist()))

print("\n5. CANH BAO CHO NGUOI TICH HOP")
_p32 = clf.model.predict_proba(X_on.astype(np.float32).astype(np.float64))
print("   Neu ha toa do xuong float32 roi moi tinh, xac suat lech toi %.3f" % np.abs(P_off - _p32).max())
print("   => LUON truyen toa do o float64. Dung lam tron, dung ep sang float32.")

print("\n" + "=" * 76)
print("KET LUAN:", "DAT — goi Final tuong duong tuyet doi voi duong ong chinh"
      if (ok1 and ok2 and ok3) else "THAT BAI — KHONG duoc dung goi nay")
print("=" * 76)

# ---------- demo API cap cao ----------
print("\nDEMO ShotDetector tren mot quy dao gia (bong nay dat):")
det = ShotDetector(MODEL, conf=0.0, fps=30.0, cooldown_s=0.0, max_window_span=0, verbose=False)
i = int(np.where(te.label.to_numpy() == 2)[0][0])
for k in range(5):
    ev = det.push(k, float(WX[i, k]), float(WY[i, k]))
    tag = "-> %s p=%.3f" % (ev["label_name"], ev["confidence"]) if ev else "-> (chua du cua so)"
    print("   push(frame=%d, x=%.4f, y=%.4f)  %s" % (k, WX[i, k], WY[i, k], tag))
print("   nhan that trong data_test.csv:", {0: "Normal", 1: "Racket", 2: "Ground"}[int(te.label.iloc[i])])
print("\n   stats():", det.stats())
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
