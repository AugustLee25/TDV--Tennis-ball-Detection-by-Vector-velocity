#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v10_features.py
===============
V10 — LỌC TÍN HIỆU VẬT LÝ (Savitzky-Golay) + ĐẶC TRƯNG CỬA SỔ TRƯỢT

Xây trên nền LightGBM V9 (giữ nguyên siêu tham số), nâng cấp ở tầng ĐẶC TRƯNG:

  (1) Làm mượt toạ độ bằng Savitzky-Golay TRƯỚC khi lấy đạo hàm, thay vì lấy
      sai phân trực tiếp trên toạ độ thô.
  (2) Bổ sung thống kê trên toàn cửa sổ (std / mean / đếm đổi dấu) để bù phần
      "trí nhớ ngắn hạn" mà LightGBM không có.

--------------------------------------------------------------------------------
HAI RÀNG BUỘC CỦA DỮ LIỆU — ĐỌC TRƯỚC KHI SỬA FILE NÀY
--------------------------------------------------------------------------------
RB1. `data_train.csv` / `data_test.csv` KHÔNG chứa chuỗi frame liên tiếp.
     frame_id nhảy 67 -> 77 -> 86 -> 97; 1063/1064 cặp dòng liên tiếp thuộc hai
     đoạn khác nhau. Mỗi dòng là MỘT cửa sổ trượt 5 frame độc lập.
     => Ngữ cảnh thời gian tối đa cho mỗi mẫu là ĐÚNG 5 điểm.
     => `savgol_filter` chỉ chạy được với window_length = 5. Không thể dài hơn.
     => `rolling()` của pandas trên trục dòng là VÔ NGHĨA ở đây — nó sẽ trộn các
        pha bóng khác nhau. Mọi thống kê phải tính TRONG cửa sổ 5 điểm.

RB2. Toạ độ thô không nằm trong CSV, nhưng TÁI DỰNG LẠI ĐƯỢC chính xác từ vector
     đặc trưng (xem `rebuild_window`). Sai số tái dựng đo được: 2.9e-05 px.

--------------------------------------------------------------------------------
ĐIỂM CẦN BIẾT VỀ TỈ LỆ TÍN HIỆU / NHIỄU (đo thật, không phải ước lượng)
--------------------------------------------------------------------------------
Độ "không mượt" của cửa sổ (RMS lệch so với đường khớp Savitzky-Golay):

    Normal : 1.17 px      <- quỹ đạo trơn
    Ground : 2.90 px      <- có gãy khúc
    Racket : 2.79 px      <- có gãy khúc

Sai số định vị YOLO đo trực tiếp trên `ball_track.csv` (bóng đang bay): 3.1-3.9 px.

=> BIÊN ĐỘ TÍN HIỆU SỰ KIỆN (~2.9 px) XẤP XỈ BIÊN ĐỘ NHIỄU (~3.6 px).
   Trong cửa sổ 5 điểm, tín hiệu và nhiễu nằm cùng một dải tần. Savitzky-Golay
   vì thế sẽ dập nhiễu VÀ dập luôn tín hiệu. Đây là lý do file này bắt buộc chạy
   ablation thay vì mặc định tin rằng làm mượt là tốt.

--------------------------------------------------------------------------------
KẾT QUẢ ĐO — ABLATION ĐÃ CHẠY XONG
--------------------------------------------------------------------------------
                                      Test F1  Số sai   Acc ở nhiễu Full HD thật
    A. V9 gốc (thô)                    0.9568     6          0.8965
    B. Chỉ Savitzky-Golay              0.9425     8          0.9403
    C. Chỉ thống kê, trên toạ độ THÔ   0.9326     9          0.8814
    D. Savgol + thống kê trên THÔ      0.9329     9          0.9342
    E. V10 CHÍNH THỨC (tất cả mượt)    0.9672     4          0.9551   <-- chọn

Ba điều bảng này nói ra:

  1. Trên tập test SẠCH, làm mượt LÀM TỆ ĐI (B < A). Đúng như dự đoán vật lý ở
     trên: cú đánh là điểm gãy, khớp đa thức trơn qua đó sẽ làm mờ nó. Nếu chỉ
     đánh giá trên tập sạch thì đã kết luận sai là "Savitzky-Golay vô dụng".

  2. Ở mức nhiễu THẬT, thứ hạng đảo ngược hoàn toàn (B >> A, +0.0438). Làm mượt
     đổi một ít độ sắc lấy rất nhiều độ bền — mà điều kiện triển khai là nhiễu,
     không phải sạch.

  3. Khác biệt giữa D và E CHỈ LÀ tham số `stats_on_smoothed`. Cùng 26 đặc trưng,
     cùng công thức, chỉ khác nguồn toạ độ để tính thống kê: 0.9342 -> 0.9551.
     `std` và `max` khuếch đại nhiễu, nên phải làm mượt TRƯỚC khi thống kê.

--------------------------------------------------------------------------------
TƯƠNG THÍCH REALTIME
--------------------------------------------------------------------------------
Mọi đặc trưng ở đây đều tính được từ đúng 5 điểm mà `FeatureExtractor` trong
`realtime_inference.py` đang đệm sẵn (WINDOW_SIZE = 5, CENTER_IDX = 2).
=> Không phải đổi logic đệm, không tăng độ trễ, không tăng bộ nhớ.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


# =============================================================================
# SECTION 0 — HẰNG SỐ
# =============================================================================
FRAME_W, FRAME_H = 1920.0, 1080.0
EPS = 1e-12

# Chỉ số trong cửa sổ 5 điểm: 0 = i-2, 1 = i-1, 2 = i (tâm), 3 = i+1, 4 = i+2
CENTER = 2

# 14 đặc trưng của V9 (RFECV đã loại `from_side`)
V9_FEATURES = [
    "x_curr", "y_curr",
    "vx_prev", "vy_prev", "speed_prev", "ax_prev", "ay_prev",
    "vx_next", "vy_next", "speed_next", "ax_next", "ay_next",
    "angle_radian", "from_player",
]

# 12 đặc trưng cửa sổ mới của V10
V10_WINDOW_FEATURES = [
    "sg_resid_mean", "sg_resid_max",          # độ "không mượt" của quỹ đạo
    "speed_mean_w", "speed_std_w",            # thống kê tốc độ trong cửa sổ
    "speed_ratio",                            # hệ số phản hồi khi va chạm
    "accel_mag_mean_w", "accel_mag_std_w",    # thống kê gia tốc
    "angle_mean_w", "angle_max_w",            # thống kê đổi hướng
    "vy_sign_flip", "vx_sign_flip",           # đếm đổi dấu vận tốc
    "jerk_mag",                               # biến thiên gia tốc
]

LABEL_NAMES = {0: "Normal", 1: "Racket", 2: "Ground"}


# =============================================================================
# SECTION 1 — TÁI DỰNG CỬA SỔ TOẠ ĐỘ TỪ VECTOR ĐẶC TRƯNG
# =============================================================================
def rebuild_window(df: pd.DataFrame):
    """
    Suy ngược 5 điểm toạ độ (đã chuẩn hoá) của cửa sổ trượt từ vector đặc trưng.

    Đây là nghịch đảo đại số của `FeatureExtractor`:
        x_{i}   = x_curr
        x_{i-1} = x_curr - vx_prev
        x_{i+1} = x_curr + vx_next
        x_{i-2} = x_{i-1} - (vx_prev - ax_prev)     vì ax_prev = v_prev(i) - v_prev(i-1)
        x_{i+2} = x_{i+1} + (vx_next + ax_next)     vì ax_next = v_next(i+1) - v_next(i)

    Trả về: (WX, WY) — hai mảng (n, 5), toạ độ chuẩn hoá [0, 1].
    """
    x1 = df["x_curr"].to_numpy(float)
    y1 = df["y_curr"].to_numpy(float)

    x0 = x1 - df["vx_prev"].to_numpy(float)
    y0 = y1 - df["vy_prev"].to_numpy(float)
    x2 = x1 + df["vx_next"].to_numpy(float)
    y2 = y1 + df["vy_next"].to_numpy(float)

    xm1 = x0 - (df["vx_prev"].to_numpy(float) - df["ax_prev"].to_numpy(float))
    ym1 = y0 - (df["vy_prev"].to_numpy(float) - df["ay_prev"].to_numpy(float))
    x3 = x2 + (df["vx_next"].to_numpy(float) + df["ax_next"].to_numpy(float))
    y3 = y2 + (df["vy_next"].to_numpy(float) + df["ay_next"].to_numpy(float))

    WX = np.stack([xm1, x0, x1, x2, x3], axis=1)
    WY = np.stack([ym1, y0, y1, y2, y3], axis=1)
    return WX, WY


def verify_rebuild(df: pd.DataFrame, feature_cols) -> float:
    """Kiểm chứng phép tái dựng: dựng lại cửa sổ, tính lại đặc trưng, so với gốc."""
    WX, WY = rebuild_window(df)
    got = kinematics_from_window(WX, WY)
    err = 0.0
    for c in feature_cols:
        if c in got:
            err = max(err, float(np.max(np.abs(got[c] - df[c].to_numpy(float)))))
    return err


# =============================================================================
# SECTION 2 — LỌC TÍN HIỆU VẬT LÝ (SAVITZKY-GOLAY)
# =============================================================================
def smooth_window(WX, WY, window_length: int = 5, polyorder: int = 2):
    """
    Làm mượt toạ độ bằng Savitzky-Golay trước khi lấy đạo hàm.

    Ý tưởng vật lý: trong pha bay tự do, quỹ đạo là đa thức bậc 2 theo thời gian
    (x tuyến tính, y có gia tốc trọng trường). Khớp bình phương tối thiểu một đa
    thức bậc `polyorder` qua cả 5 điểm rồi lấy giá trị khớp = ước lượng vị trí
    dùng TOÀN BỘ 5 điểm thay vì chỉ 2 điểm như phép sai phân.

    Lợi: phương sai nhiễu của vận tốc giảm khoảng 2-3 lần.
    Hại: cú đánh vợt là ĐIỂM GÃY vận tốc — khớp một đa thức trơn qua điểm gãy sẽ
         làm mờ chính tín hiệu cần phát hiện. Đây là đánh đổi phải đo, không đoán.

    window_length = 5 là TRẦN CỨNG do ràng buộc RB1 (xem docstring đầu file).
    """
    if window_length > WX.shape[1]:
        raise ValueError(
            f"window_length={window_length} > số điểm có được ({WX.shape[1]}). "
            "Dữ liệu chỉ cho 5 điểm mỗi mẫu — xem RB1 ở đầu file."
        )
    SX = savgol_filter(WX, window_length, polyorder, axis=1)
    SY = savgol_filter(WY, window_length, polyorder, axis=1)
    return SX, SY


# =============================================================================
# SECTION 3 — ĐỘNG HỌC CƠ BẢN (13 đặc trưng, tính trên toạ độ ĐƯỢC TRUYỀN VÀO)
# =============================================================================
def kinematics_from_window(X, Y) -> dict:
    """
    Tính 13 đặc trưng động học từ cửa sổ 5 điểm.

    Công thức giữ nguyên 1:1 của `FeatureExtractor` để V10 tương thích ngược:
        v_prev = P_i     - P_{i-1}
        v_next = P_{i+1} - P_i
        a_prev = v_prev(i)   - v_prev(i-1)
        a_next = v_next(i+1) - v_next(i)
        angle  = acos(clip(dot(v_prev, v_next) / (|v_prev| |v_next|), -1, 1))

    Tham số X, Y có thể là toạ độ THÔ hoặc ĐÃ LÀM MƯỢT — đó chính là công tắc
    bật/tắt tính năng (1) của V10.
    """
    vx_prev = X[:, 2] - X[:, 1]
    vy_prev = Y[:, 2] - Y[:, 1]
    vx_next = X[:, 3] - X[:, 2]
    vy_next = Y[:, 3] - Y[:, 2]

    ax_prev = vx_prev - (X[:, 1] - X[:, 0])
    ay_prev = vy_prev - (Y[:, 1] - Y[:, 0])
    ax_next = (X[:, 4] - X[:, 3]) - vx_next
    ay_next = (Y[:, 4] - Y[:, 3]) - vy_next

    speed_prev = np.hypot(vx_prev, vy_prev)
    speed_next = np.hypot(vx_next, vy_next)

    norm = speed_prev * speed_next
    safe = norm > EPS
    cosv = np.where(safe, (vx_prev * vx_next + vy_prev * vy_next) / np.where(safe, norm, 1.0), 1.0)
    angle_radian = np.arccos(np.clip(cosv, -1.0, 1.0))

    return {
        "x_curr": X[:, CENTER], "y_curr": Y[:, CENTER],
        "vx_prev": vx_prev, "vy_prev": vy_prev, "speed_prev": speed_prev,
        "ax_prev": ax_prev, "ay_prev": ay_prev,
        "vx_next": vx_next, "vy_next": vy_next, "speed_next": speed_next,
        "ax_next": ax_next, "ay_next": ay_next,
        "angle_radian": angle_radian,
    }


# =============================================================================
# SECTION 4 — THỐNG KÊ TRÊN TOÀN CỬA SỔ ("trí nhớ ngắn hạn" cho LightGBM)
# =============================================================================
def window_statistics(WX, WY, SX, SY, stats_on_smoothed: bool = True) -> dict:
    """
    12 đặc trưng thống kê tính trên toàn bộ cửa sổ 5 điểm.

    VÌ SAO CẦN: 13 đặc trưng động học chỉ đọc 2 trong 4 vector vận tốc và 2 trong
    3 vector gia tốc mà cửa sổ chứa. Phần còn lại bị vứt đi. Nhóm đặc trưng này
    thu hồi lại thông tin đó dưới dạng thống kê bất biến.

    ĐẶC BIỆT: `sg_resid_*` chỉ tồn tại được nhờ bước Savitzky-Golay. Sự kiện theo
    định nghĩa là chỗ quỹ đạo KHÔNG trơn, nên phần dư của phép khớp trơn chính là
    một bộ phát hiện sự kiện trực tiếp — thứ mà bản thân phép làm mượt sinh ra
    như sản phẩm phụ. Đây là đặc trưng DUY NHẤT được phép đọc toạ độ thô.

    -------------------------------------------------------------------------
    `stats_on_smoothed` — THAM SỐ QUAN TRỌNG NHẤT CỦA CẢ FILE
    -------------------------------------------------------------------------
    True  (mặc định): thống kê tính trên toạ độ ĐÃ LÀM MƯỢT.
    False           : thống kê tính trên toạ độ THÔ.

    Đo thật ở mức nhiễu Full HD 3.64/4.04 px:
        stats_on_smoothed = False  ->  Accuracy 0.8814   (TỆ HƠN cả V9 gốc)
        stats_on_smoothed = True   ->  Accuracy 0.9551   (+0.0586 so với V9 gốc)

    Lý do: `std` và `max` là toán tử KHUẾCH ĐẠI NHIỄU. Lấy std của 4 vận tốc thô
    dưới nhiễu 3.6 px cho ra một con số gần như thuần nhiễu. Làm mượt trước rồi
    mới thống kê thì mới đo được biến thiên THẬT của quỹ đạo.

    Đây chính là chỗ ý tưởng "rolling statistics" suýt bị kết luận sai là vô dụng.
    -------------------------------------------------------------------------

    WX, WY: toạ độ thô.  SX, SY: toạ độ đã làm mượt.
    """
    # --- Phần dư Savitzky-Golay: đo độ "không trơn" của quỹ đạo ---------------
    # LUÔN tính từ (thô vs mượt) — đây là ý nghĩa của phần dư.
    resid = np.hypot(WX - SX, WY - SY)                      # (n, 5)
    sg_resid_mean = resid.mean(axis=1)
    sg_resid_max = resid.max(axis=1)

    # Mọi thống kê còn lại lấy trên nguồn toạ độ do `stats_on_smoothed` quyết định
    BX, BY = (SX, SY) if stats_on_smoothed else (WX, WY)

    # --- 4 vector vận tốc trong cửa sổ: v_k = P_{k+1} - P_k, k = 0..3 --------
    vx = np.diff(BX, axis=1)                                # (n, 4)
    vy = np.diff(BY, axis=1)
    speeds = np.hypot(vx, vy)                               # (n, 4)

    speed_mean_w = speeds.mean(axis=1)
    speed_std_w = speeds.std(axis=1)
    # Hệ số phản hồi khi va chạm: nảy đất làm tốc độ giảm, cú đánh làm tăng vọt
    speed_ratio = speeds[:, 2] / (speeds[:, 1] + EPS)

    # --- 3 vector gia tốc: a_k = v_{k+1} - v_k, k = 0..2 ---------------------
    ax = np.diff(vx, axis=1)                                # (n, 3)
    ay = np.diff(vy, axis=1)
    accel_mag = np.hypot(ax, ay)
    accel_mag_mean_w = accel_mag.mean(axis=1)
    accel_mag_std_w = accel_mag.std(axis=1)

    # --- 3 góc đổi hướng tại các điểm trong cửa sổ ---------------------------
    dot = vx[:, :-1] * vx[:, 1:] + vy[:, :-1] * vy[:, 1:]   # (n, 3)
    nrm = speeds[:, :-1] * speeds[:, 1:]
    safe = nrm > EPS
    cosv = np.where(safe, dot / np.where(safe, nrm, 1.0), 1.0)
    angles = np.arccos(np.clip(cosv, -1.0, 1.0))
    angle_mean_w = angles.mean(axis=1)
    angle_max_w = angles.max(axis=1)

    # --- Đếm đổi dấu vận tốc ------------------------------------------------
    # vy đổi dấu (đang xuống -> đi lên) là chữ ký của cú NẢY ĐẤT.
    # vx đổi dấu (đang sang phải -> sang trái) là chữ ký của cú ĐÁNH VỢT.
    vy_sign_flip = (np.diff(np.sign(vy), axis=1) != 0).sum(axis=1).astype(float)
    vx_sign_flip = (np.diff(np.sign(vx), axis=1) != 0).sum(axis=1).astype(float)

    # --- Jerk: biến thiên của gia tốc, bậc đạo hàm cao nhất mà 5 điểm cho phép
    jx = np.diff(ax, axis=1)                                # (n, 2)
    jy = np.diff(ay, axis=1)
    jerk_mag = np.hypot(jx, jy).mean(axis=1)

    return {
        "sg_resid_mean": sg_resid_mean, "sg_resid_max": sg_resid_max,
        "speed_mean_w": speed_mean_w, "speed_std_w": speed_std_w,
        "speed_ratio": speed_ratio,
        "accel_mag_mean_w": accel_mag_mean_w, "accel_mag_std_w": accel_mag_std_w,
        "angle_mean_w": angle_mean_w, "angle_max_w": angle_max_w,
        "vy_sign_flip": vy_sign_flip, "vx_sign_flip": vx_sign_flip,
        "jerk_mag": jerk_mag,
    }


# =============================================================================
# SECTION 5 — HÀM FEATURE ENGINEERING CHÍNH
# =============================================================================
def engineer_features_v10(df: pd.DataFrame,
                          use_savgol: bool = True,
                          add_window_stats: bool = True,
                          stats_on_smoothed: bool = True,
                          savgol_window: int = 5,
                          savgol_poly: int = 2) -> pd.DataFrame:
    """
    Sinh ma trận đặc trưng V10 từ DataFrame gốc.

    Tham số
    -------
    use_savgol        : True  -> động học tính trên toạ độ ĐÃ LÀM MƯỢT (yêu cầu 1)
                        False -> động học tính trên toạ độ THÔ (giống hệt V9)
    add_window_stats  : True  -> thêm 12 đặc trưng thống kê cửa sổ (yêu cầu 2)
    stats_on_smoothed : thống kê lấy trên toạ độ mượt hay thô — xem docstring của
                        `window_statistics`. Đổi tham số này một mình đã làm
                        Accuracy ở mức nhiễu thật xê dịch 0.8814 <-> 0.9551.
    savgol_window     : bắt buộc = 5 với bộ dữ liệu hiện tại (xem RB1)
    savgol_poly       : bậc đa thức khớp. ĐÃ QUÉT: 1 -> thảm hoạ (F1 0.7440, vì
                        khớp đường thẳng triệt tiêu toàn bộ gia tốc); 2 -> tốt
                        nhất; 3 -> gần như không làm mượt, mất hết lợi ích chống
                        nhiễu (Acc ở nhiễu thật 0.8883 so với 0.9551 của bậc 2).

    CẤU HÌNH TRIỂN KHAI CHÍNH THỨC = toàn bộ giá trị mặc định ở trên.

    Trả về DataFrame có cột `label` ở cuối. KHÔNG có NaN: mọi đặc trưng đều tính
    trong cửa sổ 5 điểm đã đầy đủ, nên không phát sinh NaN kiểu `shift()`.
    """
    WX, WY = rebuild_window(df)

    # Luôn khớp Savitzky-Golay để lấy PHẦN DƯ, kể cả khi không dùng để làm mượt.
    # Phần dư là đặc trưng độc lập, không phụ thuộc việc có làm mượt hay không.
    SX, SY = smooth_window(WX, WY, savgol_window, savgol_poly)

    # Công tắc của yêu cầu (1): lấy đạo hàm trên toạ độ nào
    base_X, base_Y = (SX, SY) if use_savgol else (WX, WY)
    feats = kinematics_from_window(base_X, base_Y)

    # Công tắc của yêu cầu (2)
    if add_window_stats:
        feats.update(window_statistics(WX, WY, SX, SY, stats_on_smoothed))

    out = pd.DataFrame(feats, index=df.index)
    out["from_player"] = df["from_player"].to_numpy()

    cols = V9_FEATURES + (V10_WINDOW_FEATURES if add_window_stats else [])
    out = out[cols]

    # Chốt an toàn: inf/NaN không được lọt xuống tầng huấn luyện
    n_bad = int((~np.isfinite(out.to_numpy(float))).sum())
    if n_bad:
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "label" in df.columns:
        out["label"] = df["label"].to_numpy()
    return out


# =============================================================================
# SECTION 6 — HUẤN LUYỆN
# =============================================================================
# Siêu tham số V9 (Optuna trial #125). V10 GIỮ NGUYÊN để mọi thay đổi đo được
# đều quy về tầng đặc trưng, không lẫn với hiệu ứng tuning.
LGBM_V9_PARAMS = dict(
    objective="multiclass", num_class=3, random_state=42, n_jobs=-1, verbose=-1,
    learning_rate=0.015606369402299711, num_leaves=120, max_depth=10,
    min_child_samples=57, min_split_gain=0.00902380075308707,
    subsample=0.6008769993510472, subsample_freq=10,
    colsample_bytree=0.9223827247972327,
    reg_alpha=3.7255050257095e-06, reg_lambda=0.03641917016503629,
    class_weight="balanced",
)


def find_n_estimators_cv(X, y, params, n_splits=5, cap=1500, stopping_rounds=100):
    """
    Chọn số cây bằng early stopping TRÊN CV — không bao giờ trên tập test.

    Vì sao: nếu dừng theo test thì điểm dừng đã học từ test, test mất vai trò
    đánh giá độc lập. Cách đúng (V6 và V9 đều dùng): lấy trung bình
    `best_iteration_` của các fold.

    Đây là bước BẮT BUỘC chạy lại cho V10: số cây tối ưu phụ thuộc số đặc trưng,
    mà V10 có 26 đặc trưng thay vì 14 — không thể mượn nguyên con số 605 của V9.
    """
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    p = dict(params, n_estimators=cap)
    iters = []
    for idx_tr, idx_va in cv.split(X, y):
        est = LGBMClassifier(**p)
        est.fit(X[idx_tr], y[idx_tr],
                eval_set=[(X[idx_va], y[idx_va])], eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(stopping_rounds, verbose=False)])
        iters.append(est.best_iteration_ or cap)
    return int(round(float(np.mean(iters)))), iters


def train_model(X, y, params=None, n_estimators=None):
    """Huấn luyện mô hình cuối trên toàn bộ tập train."""
    from lightgbm import LGBMClassifier
    p = dict(params or LGBM_V9_PARAMS)
    if n_estimators is not None:
        p["n_estimators"] = int(n_estimators)
    return LGBMClassifier(**p).fit(X, y)


# =============================================================================
# SECTION 7 — ĐÁNH GIÁ
# =============================================================================
def evaluate_model(model, X_tr, y_tr, X_te, y_te) -> dict:
    """Gom mọi chỉ số cần thiết vào một dict."""
    from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                                 recall_score, precision_score, confusion_matrix)
    P = model.predict_proba(X_te)
    p = P.argmax(1)
    return {
        "Train F1-Macro": f1_score(y_tr, model.predict(X_tr), average="macro"),
        "Test F1-Macro": f1_score(y_te, p, average="macro"),
        "Test Acc": accuracy_score(y_te, p),
        "Test Log-Loss": log_loss(y_te, P, labels=[0, 1, 2]),
        "Recall Normal": recall_score(y_te, p, labels=[0], average="macro"),
        "Recall Racket": recall_score(y_te, p, labels=[1], average="macro"),
        "Recall Ground": recall_score(y_te, p, labels=[2], average="macro"),
        "Precision Racket": precision_score(y_te, p, labels=[1], average="macro", zero_division=0),
        "Số mẫu sai": int((p != y_te).sum()),
        "n_features": X_te.shape[1],
        "_proba": P, "_pred": p,
        "_cm": confusion_matrix(y_te, p),
    }


def event_metrics(P, y_te, threshold=0.50):
    """
    Chỉ số ở TẦNG SỰ KIỆN — tái hiện đúng điều kiện bắn sự kiện của
    `realtime_inference.py` dòng 1341: argmax != Normal VÀ conf >= threshold.
    """
    p = P.argmax(1)
    conf = P.max(1)
    fire = (p != 0) & (conf >= threshold)
    ev = y_te != 0
    tp = int((fire & ev & (p == y_te)).sum())
    fp = int((fire & ~ev).sum()) + int((fire & ev & (p != y_te)).sum())
    fn = int(ev.sum()) - tp
    return {
        "bắt đúng": tp, "báo nhầm": int((fire & ~ev).sum()), "bỏ sót": fn,
        "F1 sự kiện": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def jitter_test(model, df_raw, y_te, feature_cols, cfg,
                sigma_x_px=3.64, sigma_y_px=4.04, n_repeat=60, seed=0,
                threshold=0.50):
    """
    Đo độ bền trước nhiễu định vị của YOLO — phép thử quyết định của V10.

    Savitzky-Golay sinh ra để chống jitter, nên đánh giá nó trên tập test SẠCH là
    đánh giá sai chỗ. Mức nhiễu mặc định (3.64 / 4.04 px) là giá trị ĐO THẬT từ
    `ball_track.csv` cho quả bóng đang bay, ước lượng bằng MAD của sai phân bậc 3.

    Nhiễu được cộng vào CẢ 5 ĐIỂM của cửa sổ rồi mới tính lại toàn bộ đặc trưng —
    lan truyền đúng đường mà sai số định vị thật sự đi qua.
    """
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.default_rng(seed)
    WX, WY = rebuild_window(df_raw)
    acc, f1m, tp, fa = [], [], [], []

    for _ in range(n_repeat):
        nx = rng.normal(0.0, sigma_x_px / FRAME_W, WX.shape)
        ny = rng.normal(0.0, sigma_y_px / FRAME_H, WY.shape)
        noisy = df_raw.copy()
        # Ghi ngược cửa sổ nhiễu vào các cột mà `rebuild_window` đọc, để toàn bộ
        # chuỗi engineer_features_v10 chạy y hệt lúc suy luận thật.
        nWX, nWY = WX + nx, WY + ny
        k = kinematics_from_window(nWX, nWY)
        for c, v in k.items():
            noisy[c] = v
        g = engineer_features_v10(noisy, **cfg)
        P = model.predict_proba(g[feature_cols].to_numpy(float))
        pr = P.argmax(1)
        acc.append(accuracy_score(y_te, pr))
        f1m.append(f1_score(y_te, pr, average="macro"))
        e = event_metrics(P, y_te, threshold)
        tp.append(e["bắt đúng"]); fa.append(e["báo nhầm"])

    return {
        "Acc": float(np.mean(acc)), "Acc_std": float(np.std(acc)),
        "F1-Macro": float(np.mean(f1m)),
        "bắt đúng": float(np.mean(tp)), "báo nhầm": float(np.mean(fa)),
        "_acc_runs": np.array(acc),
    }


# =============================================================================
# SECTION 8 — BỐN CẤU HÌNH ABLATION
# =============================================================================
# Gộp savgol và rolling stats vào một lần đo thì nếu kết quả thay đổi sẽ không
# biết do cái nào. Tách ra mới quy trách nhiệm được.
ABLATION = {
    # tên cấu hình                      : tham số                                                        # F1 sạch | Acc @nhiễu thật
    "A. V9 gốc (thô, không thống kê)":   dict(use_savgol=False, add_window_stats=False),                  # 0.9568  | 0.8965
    "B. Chỉ Savitzky-Golay":             dict(use_savgol=True,  add_window_stats=False),                  # 0.9425  | 0.9403
    "C. Chỉ thống kê, trên toạ độ THÔ":  dict(use_savgol=False, add_window_stats=True,
                                              stats_on_smoothed=False),                                   # 0.9326  | 0.8814
    "D. Savgol + thống kê trên THÔ":     dict(use_savgol=True,  add_window_stats=True,
                                              stats_on_smoothed=False),                                   # 0.9329  | 0.9342
    "E. V10 CHÍNH THỨC (tất cả mượt)":   dict(use_savgol=True,  add_window_stats=True,
                                              stats_on_smoothed=True),                                    # 0.9672  | 0.9551
}

# Bài học của bảng trên: nếu chỉ chạy A vs D (cách gộp thông thường) thì sẽ kết
# luận "thống kê cửa sổ làm hại, bỏ đi" — vì D < B ở cả hai cột. Chỉ khi tách
# `stats_on_smoothed` thành một trục riêng mới lộ ra rằng vấn đề không nằm ở ý
# tưởng thống kê, mà nằm ở việc thống kê đang đọc toạ độ thô.
