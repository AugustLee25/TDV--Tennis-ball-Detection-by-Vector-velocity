# -*- coding: utf-8 -*-
"""
tennis_shot_api.py - PHAN LOAI CHAM BONG: Racket (vot danh) hay Ground (nay dat).

DAY LA TOAN BO PHAN VIEC CUA MODULE NAY. No KHONG phat hien bong, KHONG bam vet,
KHONG doc video. Dau vao la toa do qua bong theo tung khung hinh do pipeline cua
ban cung cap; dau ra la su kien Racket / Ground kem xac suat.

    pipeline cua ban                          module nay
    ----------------                          ----------
    video -> detector -> tracker -> (frame_idx, x, y) -> ShotDetector.push()
                                                              |
                                          None  hoac  {label_name, confidence, ...}

CACH DUNG NGAN NHAT
-------------------
    from tennis_shot_api import ShotDetector

    det = ShotDetector('models/lgbm_v10_savgol.joblib', fps=30.0)
    for frame_idx, (x, y) in enumerate(toa_do_bong):    # x, y chuan hoa 0..1
        ev = det.push(frame_idx, x, y)
        if ev:
            print(ev['time_s'], ev['label_name'], ev['confidence'])

BON DIEU PHAI BIET TRUOC KHI TICH HOP
-------------------------------------
1. TOA DO CHUAN HOA 0..1, khong phai pixel. Muon truyen pixel thi khoi tao voi
   frame_size=(1920, 1080) - module se tu chia.

2. CO DO TRE 2 KHUNG HINH. Cua so truot can 5 diem (i-2 ... i+2) va phan loai
   DIEM O GIUA. Nen push() tra ve su kien cua khung hinh i khi ban day vao diem
   i+2. Day la ban chat cua bai toan chu khong phai loi: muon biet qua bong co
   doi huong khong thi phai thay ca truoc lan sau.

3. PHAI GOI reset() MOI KHI CAT CANH. Chuoi quy dao gia dinh cac khung hinh lien
   tiep thuoc cung mot cu quay. May quay nhay cho -> van toc bieu kien dao dau ->
   trung khit voi chu ky vat ly cua mot cu nay dat. Khong reset thi moi lan cat
   canh sinh ra mot su kien gia.

4. BO QUA KHUNG HINH KHONG BAT DUOC BONG. Dung noi suy san roi day vao. Module
   tu noi suy len luoi deu khi can, va tu tu choi cua so nao trai qua rong.

MODEL: LightGBM V10 (Savitzky-Golay + thong ke cua so), 26 dac trung.
       Test F1-Macro 0.9672 | Acc 0.9777 | 4/179 mau sai.

SINH RA TU realtime_inference.py bang ast - khong chep tay.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

try:
    import joblib
except ImportError:
    joblib = None

warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', message='.*valid feature names.*')


# =============================================================================
# HANG SO - trich tu realtime_inference.py
# =============================================================================

DEFAULT_FEATURE_COLS: List[str] = [
    "x_curr", "y_curr",
    "vx_prev", "vy_prev", "speed_prev", "ax_prev", "ay_prev",
    "vx_next", "vy_next", "speed_next", "ax_next", "ay_next",
    "angle_radian", "from_player", "from_side",
]

DEFAULT_CLASS_NAMES: Dict[int, str] = {0: "Normal", 1: "Racket", 2: "Ground"}

WINDOW_SIZE = 5

CENTER_IDX = 2


# =============================================================================
# CAC LOP - trich NGUYEN VAN tu realtime_inference.py
# =============================================================================

@dataclass
class Detection:
    frame_idx: int
    x: float          # đã chuẩn hoá 0..1 theo chiều rộng khung hình
    y: float

@dataclass
class FeatureSample:
    frame_idx: int
    features: Dict[str, float]
    x: float
    y: float

class FeatureExtractor:
    """Sinh vector đặc trưng cho mô hình V4/V9 (thô) HOẶC V10/V10.1 (Savitzky-Golay).

    ─────────────────────────────────────────────────────────────────────────
    HAI CHẾ ĐỘ, CHỌN TỰ ĐỘNG THEO BUNDLE
    ─────────────────────────────────────────────────────────────────────────
    smooth=False : động học lấy sai phân trực tiếp trên toạ độ THÔ  -> V4, V9
    smooth=True  : làm mượt Savitzky-Golay (window=5, polyorder=2) TRƯỚC khi
                   lấy đạo hàm, cộng thêm 14 đặc trưng thống kê cửa sổ -> V10+

    Chế độ do khối `feature_engineering` trong file .joblib quyết định, nên
    đổi model là đổi luôn cách sinh đặc trưng — không phải sửa code.

    ─────────────────────────────────────────────────────────────────────────
    VẤN ĐỀ KHUNG HÌNH KHÔNG LIÊN TIẾP
    ─────────────────────────────────────────────────────────────────────────
    Savitzky-Golay giả định các mẫu CÁCH ĐỀU nhau. Nhưng realtime thì YOLO
    thường xuyên trượt vài khung hình, nên 5 lần phát hiện trong buffer có thể
    ứng với các frame 100, 101, 103, 104, 106.

    Áp savgol thẳng lên chuỗi đó là SAI: bộ lọc sẽ coi 5 điểm là cách đều và
    tính ra đạo hàm lệch tỉ lệ với độ thưa.

    Cách xử lý: nếu 5 khung hình không liên tiếp, NỘI SUY TUYẾN TÍNH về lưới
    đều [f_c-2 .. f_c+2] trước khi lọc. Đây đúng là thứ mà pipeline sinh dữ
    liệu train đã làm (yolo_ball_tracker.py nội suy khoảng trống <= 7 frame),
    nên realtime và train khớp nhau về mặt xử lý.
    Bộ đếm `n_interpolated` cho biết việc này xảy ra bao nhiêu lần.
    """

    #: Đầy đủ mọi đặc trưng mà lớp này biết sinh. ShotClassifier sẽ lọc lại
    #: theo đúng bundle['feature_cols'], nên sinh thừa không gây hại.
    PRODUCES = [
        # 13 đặc trưng động học (V4/V9 base, trừ from_side/from_player do ContextEstimator lo)
        "x_curr", "y_curr", "vx_prev", "vy_prev", "speed_prev", "ax_prev", "ay_prev",
        "vx_next", "vy_next", "speed_next", "ax_next", "ay_next", "angle_radian",
        # 12 thống kê cửa sổ của V10
        "sg_resid_mean", "sg_resid_max", "speed_mean_w", "speed_std_w", "speed_ratio",
        "accel_mag_mean_w", "accel_mag_std_w", "angle_mean_w", "angle_max_w",
        "vy_sign_flip", "vx_sign_flip", "jerk_mag",
        # bổ sung của V10.5 / V11 — sinh sẵn để tương thích ngược lẫn xuôi
        "angle_diff_max", "angle_std_w3",
    ]

    def __init__(self, max_frame_gap: int = 12, smooth: bool = False,
                 savgol_window: int = 5, savgol_poly: int = 2) -> None:
        self.max_frame_gap = max_frame_gap
        self.smooth = bool(smooth) and savgol_filter is not None
        self.savgol_window = int(savgol_window)
        self.savgol_poly = int(savgol_poly)
        self.window: Deque[Detection] = deque(maxlen=WINDOW_SIZE)
        self.n_interpolated = 0
        self.n_emitted = 0
        self.max_window_span = 0        # 0 = tắt
        self.n_rejected_span = 0
        if bool(smooth) and savgol_filter is None:
            print("[FeatureExtractor] CẢNH BÁO: thiếu scipy -> không làm mượt được. "
                  "Cài bằng: pip install scipy")

    def reset(self) -> None:
        self.window.clear()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _angle(v1, v2) -> float:
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        return math.acos(max(-1.0, min(1.0, cos)))

    def _uniform_grid(self, dets):
        """Trả về (WX, WY) đã đặt trên lưới khung hình ĐỀU quanh điểm giữa."""
        f = np.array([d.frame_idx for d in dets], dtype=np.float64)
        x = np.array([d.x for d in dets], dtype=np.float64)
        y = np.array([d.y for d in dets], dtype=np.float64)
        if np.all(np.diff(f) == 1.0):
            return x, y, False
        fc = f[CENTER_IDX]
        grid = np.arange(fc - CENTER_IDX, fc - CENTER_IDX + WINDOW_SIZE, dtype=np.float64)
        return np.interp(grid, f, x), np.interp(grid, f, y), True

    # ------------------------------------------------------------------ #
    def push(self, det: Detection):
        """Thêm 1 lần phát hiện. Trả về FeatureSample của khung hình GIỮA nếu đủ dữ liệu."""
        if self.window and (det.frame_idx - self.window[-1].frame_idx) > self.max_frame_gap:
            self.window.clear()
        self.window.append(det)
        if len(self.window) < WINDOW_SIZE:
            return None

        dets = list(self.window)

        # 5 điểm liên tiếp lý tưởng trải ĐÚNG 4 frame. Trải rộng hơn nghĩa là
        # phần lớn cửa sổ do nội suy bịa ra -> thà bỏ còn hơn đoán bừa.
        if self.max_window_span:
            span = dets[-1].frame_idx - dets[0].frame_idx
            if span > self.max_window_span:
                self.n_rejected_span += 1
                return None

        WX, WY, was_interp = self._uniform_grid(dets)
        if was_interp:
            self.n_interpolated += 1

        # ---- Savitzky-Golay: làm mượt + phần dư -------------------------
        if self.smooth:
            SX = savgol_filter(WX, self.savgol_window, self.savgol_poly)
            SY = savgol_filter(WY, self.savgol_window, self.savgol_poly)
        else:
            SX, SY = WX, WY
        resid = np.hypot(WX - SX, WY - SY)          # 0 nếu không làm mượt

        # ---- động học trên toạ độ ĐÃ CHỌN (mượt hoặc thô) ---------------
        vx = np.diff(SX); vy = np.diff(SY)          # (4,)
        ax = np.diff(vx); ay = np.diff(vy)          # (3,)  — hằng số khi poly=2
        sp = np.hypot(vx, vy)

        v_prev = (float(vx[1]), float(vy[1]))
        v_next = (float(vx[2]), float(vy[2]))

        dot = vx[:-1] * vx[1:] + vy[:-1] * vy[1:]
        nrm = sp[:-1] * sp[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            cosv = np.where(nrm > 1e-12, dot / np.where(nrm > 1e-12, nrm, 1.0), 1.0)
        ang = np.arccos(np.clip(cosv, -1.0, 1.0))   # (3,) góc bẻ tại 3 điểm trong

        jx = np.diff(ax); jy = np.diff(ay)
        amag = np.hypot(ax, ay)

        feats = {
            # ---- 13 đặc trưng động học (tương thích ngược V4/V9) ----
            "x_curr":       float(SX[CENTER_IDX]),
            "y_curr":       float(SY[CENTER_IDX]),
            "vx_prev":      v_prev[0],
            "vy_prev":      v_prev[1],
            "speed_prev":   float(sp[1]),
            "ax_prev":      float(ax[0]),
            "ay_prev":      float(ay[0]),
            "vx_next":      v_next[0],
            "vy_next":      v_next[1],
            "speed_next":   float(sp[2]),
            "ax_next":      float(ax[1]),
            "ay_next":      float(ay[1]),
            "angle_radian": self._angle(v_prev, v_next),
            # ---- 12 thống kê cửa sổ của V10 ----
            "sg_resid_mean":    float(resid.mean()),
            "sg_resid_max":     float(resid.max()),
            "speed_mean_w":     float(sp.mean()),
            "speed_std_w":      float(sp.std()),
            "speed_ratio":      float(sp[2] / (sp[1] + 1e-12)),
            "accel_mag_mean_w": float(amag.mean()),
            "accel_mag_std_w":  float(amag.std()),
            "angle_mean_w":     float(ang.mean()),
            "angle_max_w":      float(ang.max()),
            "vy_sign_flip":     float((np.diff(np.sign(vy)) != 0).sum()),
            "vx_sign_flip":     float((np.diff(np.sign(vx)) != 0).sum()),
            "jerk_mag":         float(np.hypot(jx, jy).mean()),
            # ---- bổ sung V10.5 / V11 ----
            "angle_diff_max":   float(np.abs(np.diff(ang)).max()),
            "angle_std_w3":     float(ang.std()),
        }
        self.n_emitted += 1
        p2 = dets[CENTER_IDX]
        return FeatureSample(frame_idx=p2.frame_idx, features=feats,
                             x=float(SX[CENTER_IDX]), y=float(SY[CENTER_IDX]))

    def describe(self) -> str:
        mode = (f"Savitzky-Golay w={self.savgol_window} p={self.savgol_poly}"
                if self.smooth else "toạ độ THÔ (V4/V9)")
        return (f"[FeatureExtractor] {mode} | sinh {len(self.PRODUCES)} đặc trưng | "
                f"max_frame_gap={self.max_frame_gap}")

class ContextEstimator:
    """Ước lượng 2 feature ngữ cảnh `from_side` và `from_player`.

    ┌──────────────────────────────────────────────────────────────────────┐
    │ !!! ĐIỂM CẦN CHỈNH CHO ĐÚNG PIPELINE CỦA BẠN !!!                     │
    │                                                                      │
    │ Hai cột này trong data_train.csv đến từ bước gán nhãn của bạn, không  │
    │ suy ra được từ mỗi toạ độ bóng. Ở đây dùng heuristic hợp lý nhất:     │
    │                                                                      │
    │   from_side   = 0 nếu bóng ở NỬA TRÊN sân (y < net_y), 1 nếu NỬA DƯỚI │
    │   from_player = người vừa đánh quả gần nhất; khởi tạo theo nửa sân,   │
    │                 và ĐẢO mỗi khi model bắt được một sự kiện `Racket`.   │
    │                                                                      │
    │ Nếu code sinh dataset của bạn định nghĩa khác (ví dụ from_player theo │
    │ hướng bóng bay, hay theo player gần bóng nhất), hãy sửa đúng 2 hàm    │
    │ `side_of()` và `current_player()` bên dưới — phần còn lại giữ nguyên. │
    └──────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, net_y: float = 0.5, invert_side: bool = False) -> None:
        self.net_y = net_y
        self.invert_side = invert_side
        self._player: Optional[int] = None

    def reset(self) -> None:
        self._player = None

    def side_of(self, y_norm: float) -> int:
        side = 1 if y_norm >= self.net_y else 0
        return 1 - side if self.invert_side else side

    def current_player(self, y_norm: float) -> int:
        if self._player is None:                 # chưa có sự kiện Racket nào -> đoán theo nửa sân
            self._player = self.side_of(y_norm)
        return self._player

    def on_racket_event(self, y_norm: float) -> None:
        """Gọi khi phát hiện cú chạm vợt -> đổi lượt người đánh."""
        self._player = self.side_of(y_norm)

class ShotClassifier:
    """Nạp bundle joblib do `model_benchmark_V4.ipynb` lưu ra và dự đoán."""

    def __init__(self, model_path: str) -> None:
        if joblib is None:
            raise RuntimeError("Thiếu joblib. Cài bằng: pip install joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Không tìm thấy '{model_path}'.\n"
                "Hãy chạy hết notebook model_benchmark_V4.ipynb (phần 10) để sinh file này."
            )

        obj = joblib.load(model_path)

        if isinstance(obj, dict) and "model" in obj:            # bundle chuẩn của V4
            self.model         = obj["model"]
            self.scaler        = obj.get("scaler")
            self.needs_scaling = bool(obj.get("needs_scaling", False))
            self.feature_cols  = list(obj.get("feature_cols", DEFAULT_FEATURE_COLS))
            raw_names          = obj.get("class_names", DEFAULT_CLASS_NAMES)
            self.class_names   = {int(k): str(v) for k, v in raw_names.items()}
            self.model_name    = obj.get("model_name", type(self.model).__name__)
            self.metrics       = obj.get("metrics", {})
            self.fe_cfg        = dict(obj.get("feature_engineering", {}) or {})
        else:                                                   # estimator trần
            self.model         = obj
            self.scaler        = None
            self.needs_scaling = False
            self.feature_cols  = list(DEFAULT_FEATURE_COLS)
            self.class_names   = dict(DEFAULT_CLASS_NAMES)
            self.model_name    = type(obj).__name__
            self.metrics       = {}
            self.fe_cfg        = {}

        self.classes_ = [int(c) for c in getattr(self.model, "classes_", sorted(self.class_names))]

        print(f"[Model] {self.model_name}")
        print(f"[Model] {len(self.feature_cols)} features | scaling ngoài: {self.needs_scaling}")
        if self.fe_cfg:
            print(f"[Model] feature_engineering = {self.fe_cfg}")
        else:
            print("[Model] bundle không khai báo feature_engineering -> dùng toạ độ THÔ (V4/V9)")
        if self.metrics:
            f1 = self.metrics.get("Test F1-Macro")
            acc = self.metrics.get("Test Acc")
            if f1 is not None:
                print(f"[Model] Test F1-Macro = {float(f1):.4f} | Test Acc = {float(acc):.4f}")

    # ------------------------------------------------------------------ #
    def _to_matrix(self, feats: Dict[str, float]) -> np.ndarray:
        """Xếp dict feature thành ma trận (1, n_features) ĐÚNG thứ tự cột lúc train."""
        try:
            row = [float(feats[c]) for c in self.feature_cols]
        except KeyError as exc:
            raise KeyError(f"Thiếu feature {exc} — kiểm tra lại FeatureExtractor/ContextEstimator") from exc
        X = np.asarray([row], dtype=np.float64)

        if self.needs_scaling and self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def predict(self, feats: Dict[str, float]) -> Tuple[int, np.ndarray]:
        """Trả về (nhãn dự đoán, mảng xác suất theo thứ tự self.classes_)."""
        X = self._to_matrix(feats)
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X))[0]
            label = int(self.classes_[int(np.argmax(proba))])
        else:
            label = int(self.model.predict(X)[0])
            proba = np.zeros(len(self.classes_), dtype=float)
            proba[self.classes_.index(label)] = 1.0
        return label, proba

    def label_name(self, label: int) -> str:
        return self.class_names.get(int(label), str(label))


# =============================================================================
# API CAP CAO — cai duy nhat pipeline khac can dung
# =============================================================================
class CourtMask:
    """Loc khong gian: bo moi phat hien co tam nam NGOAI da giac mat san.

    Tuy chon. Do tren video nghiem thu, no keo ti le su kien Racket sinh ra tu
    khan dai tu 14.6% xuong 3.6% ma gan nhu khong dung toi lop Ground (297 -> 296).
    Chi phi 2.85 micro giay moi diem.

    File JSON do reference/select_court.py sinh ra.
    """

    def __init__(self, path: str) -> None:
        import cv2
        self._cv2 = cv2
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        pts = d.get("points") if isinstance(d, dict) else d
        if not pts or len(pts) < 3:
            raise ValueError("court mask can it nhat 3 dinh: " + path)
        self.ref_w = float(d.get("frame_width", 0)) if isinstance(d, dict) else 0.0
        self.ref_h = float(d.get("frame_height", 0)) if isinstance(d, dict) else 0.0
        self._pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        self._poly_norm = None

    def _poly(self):
        """Da giac o TOA DO CHUAN HOA 0..1 — doc lap voi do phan giai video."""
        if self._poly_norm is None:
            w = self.ref_w or 1920.0
            h = self.ref_h or 1080.0
            p = self._pts / np.array([w, h], dtype=np.float32)
            self._poly_norm = p.reshape(-1, 1, 2).astype(np.float32)
        return self._poly_norm

    def contains(self, x_norm: float, y_norm: float) -> bool:
        return self._cv2.pointPolygonTest(self._poly(), (float(x_norm), float(y_norm)), False) >= 0


class ShotDetector:
    """Bao goi FeatureExtractor + ContextEstimator + ShotClassifier thanh MOT cua vao.

    Tham so
    -------
    model_path      duong dan file .joblib
    conf            nguong xac suat toi thieu de ghi nhan su kien
                    (0.6 la cau hinh dang chay production)
    fps             chi dung de quy doi frame_idx -> giay
    cooldown_s      hai su kien khong duoc cach nhau gan hon ngan nay
    frame_size      None    -> push() nhan toa do CHUAN HOA 0..1
                    (W, H)  -> push() nhan toa do PIXEL, module tu chia
    max_frame_gap   cach nhau qua ngan nay khung hinh thi coi nhu mat dau
    max_window_span bo cua so neu 5 lan phat hien trai qua ngan nay khung hinh
                    (4 = lien tiep hoan hao; 0 = tat kiem tra)
    net_y           vi tri luoi theo truc y chuan hoa, dung cho dac trung ngu canh
    court_mask      duong dan court_mask.json, hoac None
    """

    def __init__(self, model_path: str, conf: float = 0.6, fps: float = 30.0,
                 cooldown_s: float = 0.20, frame_size=None,
                 max_frame_gap: int = 12, max_window_span: int = 8,
                 net_y: float = 0.5, court_mask=None, verbose: bool = True):
        self.clf = ShotClassifier(model_path)
        fe = dict(getattr(self.clf, "fe_cfg", {}) or {})
        smooth = bool(fe.get("savgol_window"))
        self.fx = FeatureExtractor(
            max_frame_gap=max_frame_gap,
            smooth=smooth,
            savgol_window=int(fe.get("savgol_window", 5) or 5),
            savgol_poly=int(fe.get("savgol_polyorder", 2) or 2),
        )
        self.fx.max_window_span = int(max_window_span or 0)
        self.ctx = ContextEstimator(net_y=net_y)

        known = set(self.fx.PRODUCES) | {"from_side", "from_player"}
        missing = [c for c in self.clf.feature_cols if c not in known]
        if missing:
            raise SystemExit("Model doi %d dac trung ma module khong sinh duoc: %s"
                             % (len(missing), missing))

        self.conf = float(conf)
        self.fps = float(fps)
        self.cooldown_frames = max(int(cooldown_s * fps), 1)
        self.frame_size = frame_size
        self.mask = CourtMask(court_mask) if court_mask else None
        self._last_event_frame = -10 ** 9
        self.n_pushed = 0
        self.n_predicted = 0
        self.n_events = 0
        self.n_out_court = 0
        if verbose:
            print("[ShotDetector] conf=%.2f | cooldown=%d frame | span<=%d | court_mask=%s"
                  % (self.conf, self.cooldown_frames, self.fx.max_window_span,
                     "BAT" if self.mask else "tat"))

    def reset(self) -> None:
        """GOI MOI KHI CAT CANH. Xoa cua so truot va trang thai ngu canh.

        Bo qua buoc nay la nguon sinh su kien gia lon nhat: may quay nhay cho lam
        van toc bieu kien cua bong dao dau, trung khit voi chu ky vat ly cua mot
        cu nay dat.
        """
        self.fx.reset()
        self.ctx.reset()
        # Xoa luon moc cooldown: sau cat canh, su kien cu thuoc ve mot cu quay khac,
        # khong duoc phep chan su kien dau tien cua doan moi.
        self._last_event_frame = -10 ** 9

    def push(self, frame_idx: int, x, y):
        """Day mot lan phat hien bong vao. Tra ve dict su kien, hoac None.

        LUU Y DO TRE: su kien tra ve ung voi khung hinh frame_idx - 2, vi cua so
        truot phan loai DIEM O GIUA cua 5 diem.
        """
        if x is None or y is None:
            return None
        if self.frame_size is not None:
            x = float(x) / float(self.frame_size[0])
            y = float(y) / float(self.frame_size[1])
        x, y = float(x), float(y)
        self.n_pushed += 1

        if self.mask is not None and not self.mask.contains(x, y):
            self.n_out_court += 1
            return None

        sample = self.fx.push(Detection(frame_idx=int(frame_idx), x=x, y=y))
        if sample is None:
            return None

        sample.features["from_side"] = self.ctx.side_of(sample.y)
        sample.features["from_player"] = self.ctx.current_player(sample.y)
        label, proba = self.clf.predict(sample.features)
        self.n_predicted += 1
        conf = float(np.max(proba))

        if int(label) == 0 or conf < self.conf:
            return None
        if sample.frame_idx - self._last_event_frame < self.cooldown_frames:
            return None

        self._last_event_frame = sample.frame_idx
        name = self.clf.label_name(label)
        if name.lower().startswith("racket"):
            self.ctx.on_racket_event(sample.y)
        self.n_events += 1
        return {
            "frame_idx": int(sample.frame_idx),
            "time_s": sample.frame_idx / self.fps,
            "label": int(label),
            "label_name": name,
            "confidence": conf,
            "x_norm": float(sample.x),
            "y_norm": float(sample.y),
            "proba": {self.clf.label_name(c): float(p)
                      for c, p in zip(self.clf.classes_, proba)},
        }

    def stats(self):
        return {
            "da nhan": self.n_pushed,
            "loai vi ngoai san": self.n_out_court,
            "so lan du doan": self.n_predicted,
            "su kien": self.n_events,
            "cua so phai noi suy": getattr(self.fx, "n_interpolated", 0),
            "cua so bi loai vi trai rong": getattr(self.fx, "n_rejected_span", 0),
        }


__all__ = ["ShotDetector", "CourtMask", "Detection", "FeatureSample",
           "FeatureExtractor", "ContextEstimator", "ShotClassifier"]
