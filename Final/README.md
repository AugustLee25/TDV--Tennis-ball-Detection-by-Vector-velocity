# Final — module phân loại chạm bóng (Racket / Ground)

Phần việc của module này: **cho toạ độ quả bóng theo từng khung hình, trả về sự kiện
vợt đánh hay bóng nảy đất.** Nó KHÔNG phát hiện bóng, KHÔNG bám vết, KHÔNG đọc video —
những phần đó do pipeline của nhóm khác lo.

```
pipeline của nhóm khác                       module này
──────────────────────                       ──────────
video → detector → tracker → (frame_idx, x, y) → ShotDetector.push()
                                                       │
                                   None  hoặc  {label_name, confidence, ...}
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

`opencv-python` chỉ cần nếu dùng `CourtMask` (lọc không gian) — không bắt buộc.

## Dùng

```python
from tennis_shot_api import ShotDetector

det = ShotDetector("models/lgbm_v10_savgol.joblib", conf=0.6, fps=30.0)

for frame_idx, (x, y) in enumerate(toa_do_bong):    # x, y chuẩn hoá 0..1
    ev = det.push(frame_idx, x, y)
    if ev:
        print(ev["time_s"], ev["label_name"], ev["confidence"])
```

Chạy thử ngay:

```bash
python example_usage.py     # ví dụ trên dữ liệu thật
python verify_final.py      # tự kiểm chứng, phải in "DAT"
```

---

## Hợp đồng tích hợp

### Đầu vào — `push(frame_idx, x, y)`

| tham số | kiểu | ghi chú |
|---|---|---|
| `frame_idx` | `int` | chỉ số khung hình, **phải tăng dần** |
| `x`, `y` | `float` | mặc định **chuẩn hoá 0..1**. Muốn truyền pixel thì khởi tạo với `frame_size=(1920,1080)` |

Chỉ gọi `push()` ở những khung hình **thực sự bắt được bóng**. Khung hình không có bóng
thì bỏ qua — đừng tự nội suy rồi đẩy vào, module tự lo việc đó.

### Đầu ra

`None` khi chưa đủ cửa sổ / không phải sự kiện / dưới ngưỡng, hoặc:

```python
{
  "frame_idx": 1234,          # khung hình của SỰ KIỆN (= frame_idx bạn đẩy vào − 2)
  "time_s": 41.13,
  "label": 2,                 # 1 = Racket, 2 = Ground
  "label_name": "Ground",
  "confidence": 0.983,
  "x_norm": 0.287, "y_norm": 0.639,
  "proba": {"Normal": 0.01, "Racket": 0.007, "Ground": 0.983},
}
```

### Tham số khởi tạo

| tham số | mặc định | ý nghĩa |
|---|---|---|
| `conf` | `0.6` | ngưỡng xác suất tối thiểu. Cấu hình production đang dùng |
| `fps` | `30.0` | chỉ để quy đổi `frame_idx` → giây |
| `cooldown_s` | `0.20` | hai sự kiện không được cách nhau gần hơn |
| `frame_size` | `None` | `None` = toạ độ 0..1; `(W,H)` = pixel |
| `max_frame_gap` | `12` | cách nhau quá ngần này khung hình thì coi như mất dấu |
| `max_window_span` | `8` | bỏ cửa sổ nếu 5 lần phát hiện trải quá ngần này khung hình |
| `court_mask` | `None` | đường dẫn `court_mask.json` để lọc không gian |

---

## Năm điều dễ sai khi tích hợp

**1. Trễ 2 khung hình — không phải lỗi.** Cửa sổ trượt cần 5 điểm (`i-2 … i+2`) và phân
loại **điểm ở giữa**. Nên `push()` trả về sự kiện của khung hình `i` khi bạn đẩy vào điểm
`i+2`. Muốn biết quả bóng có đổi hướng không thì phải thấy cả trước lẫn sau — đó là bản
chất bài toán.

**2. Phải gọi `reset()` mỗi khi cắt cảnh.** Chuỗi quỹ đạo giả định các khung hình liên
tiếp thuộc cùng một cú quay. Máy quay nhảy chỗ → vận tốc biểu kiến đảo dấu → **trùng khít
với chữ ký vật lý của một cú nảy đất**. Trên video nghiệm thu 30 phút có 585 lần cắt cảnh;
không reset thì mỗi lần cắt sinh một sự kiện giả.

**3. Truyền toạ độ ở `float64`.** Đo được: hạ xuống `float32` rồi mới tính làm xác suất
lệch tới **0,175**. Cây quyết định là hàm gián đoạn — sai số làm tròn đủ để vượt một ngưỡng
chia và đổi lá.

**4. `from_side` / `from_player` đang dùng heuristic.** Hai đặc trưng ngữ cảnh này (chiếm
1,7% gain) trong tập train đến từ bước gán nhãn, không suy ra được từ mỗi toạ độ bóng.
`ContextEstimator` ước lượng: `from_side` theo nửa sân so với lưới, `from_player` đảo mỗi
khi bắt được một sự kiện Racket. Nếu pipeline của bạn biết chính xác ai đang đánh thì ghi
đè hai giá trị này sẽ tốt hơn.

**5. Nhóm sai còn lại: bóng trong tay người.** Bóng người chơi cầm chuẩn bị giao, hoặc
người nhặt bóng cầm, đều là bóng thật ở trong sân và bị gán nhãn Racket/Ground. Nếu
pipeline của bạn có bounding box người chơi, lọc bỏ những quả bóng nằm trong tay sẽ dọn
được phần lớn nhóm này.

---

## Model

`models/lgbm_v10_savgol.joblib` — LightGBM 3 lớp, 26 đặc trưng, 499 vòng × 3 lớp = 1.497 cây.

| | |
|---|---|
| Test F1-Macro | **0,9672** |
| Test Accuracy | 0,9777 |
| Test Log-Loss | 0,0605 |
| Recall Normal / Racket / Ground | 1,000 / 0,939 / 0,957 |
| Mẫu sai | 4 / 179 |

Đặc trưng quan trọng nhất: `angle_max_w` (25,9%), `ay_prev` (14,1%), `vx_sign_flip` (12,2%)
— ba cái này chiếm hơn một nửa toàn bộ gain.

Bundle mang theo khoá `env_current` ghi phiên bản thư viện đã kiểm chứng:
`python 3.10.20 · sklearn 1.7.2 · lightgbm 4.7.0 · numpy 2.2.6 · scipy 1.15.3`.

---

## Kiểm chứng gói này

`verify_final.py` chạy hai đường độc lập trên cùng 179 mẫu và đối chiếu:

```
OFFLINE  data_test.csv → engineer_features_v10() → model
ONLINE   data_test.csv → rebuild_window() → 5 điểm toạ độ
                       → FeatureExtractor.push() ×5 → model
```

Kết quả lần chạy gần nhất:

| kiểm tra | kết quả |
|---|---|
| Lệch đặc trưng lớn nhất (26 cột × 179 mẫu) | `8,2e-14` |
| Nhãn dự đoán trùng nhau | **179/179** |
| F1-Macro hai đường | **0,967151** — giống hệt |
| Tập mẫu sai | `{37, 76, 117, 166}` — giống hệt |

Lệch xác suất lớn nhất là `3,1e-03` trên 69/179 mẫu, và **không mẫu nào đổi nhãn**. Nguyên
nhân: đặc trưng lệch `1e-16` (đúng bằng epsilon của `float64`, do thứ tự phép tính khác
nhau giữa đường vector hoá và đường từng mẫu) đủ để vượt một ngưỡng chia trong cây. Ensemble
cây là hàm gián đoạn nên chênh lệch ở mức bit sinh ra bước nhảy hữu hạn ở đầu ra. Đây là
bản chất, không phải lỗi đóng gói.

---

## Nội dung gói

```
Final/
├── tennis_shot_api.py      module duy nhất cần import (566 dòng)
├── example_usage.py        ví dụ chạy được ngay
├── verify_final.py         tự kiểm chứng
├── requirements.txt        phiên bản đã kiểm chứng
├── court_mask.json         đa giác mặt sân (tuỳ chọn)
├── models/
│   └── lgbm_v10_savgol.joblib
└── reference/              chỉ để tham khảo, không cần lúc chạy
    ├── model_benchmark_V10_features.py   đường ống sinh đặc trưng lúc train
    ├── select_court.py                   công cụ khoanh lại mặt sân
    └── data_test.csv                     dữ liệu cho verify_final.py
```

`tennis_shot_api.py` được **trích tự động bằng `ast`** từ `realtime_inference.py`, không
chép tay. Muốn dựng lại sau khi sửa bản gốc: `python scripts/tools/build_final_package.py`.
