#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
select_court.py — khoanh mặt sân một lần, dùng mãi mãi.

VÌ SAO CẦN CÔNG CỤ NÀY
    Bốn lượt nghiệm thu đã chứng minh: 15% sự kiện Racket sinh ra từ đốm sáng trên
    khán đài, và 40,9% sự kiện Ground rơi ngoài dải y mà lớp Ground chiếm lúc train.
    Không có tham số nào chữa được, vì mô hình không sai — nó chỉ đang được hỏi về
    những pixel mà nó chưa bao giờ được dạy.

    Cách rẻ nhất để sửa không phải train lại, mà là ĐỪNG HỎI. Một đa giác bao quanh
    mặt sân biến "quả bóng trên khán đài" từ một bài toán thị giác máy tính thành
    một phép so sánh hình học O(1).

CÁCH DÙNG
    python scripts/tools/select_court.py
    python scripts/tools/select_court.py --source VideoForTest.mp4 --frame 3000

PHÍM
    chuột trái     thêm một đỉnh
    chuột phải / z bỏ đỉnh vừa thêm
    c              xoá hết, chọn lại
    , .            lùi / tiến 30 khung hình      (tìm khung hình thấy rõ cả sân)
    [ ]            lùi / tiến 300 khung hình
    s              lưu court_mask.json rồi thoát
    q / ESC        thoát, không lưu

MẸO CHỌN ĐỈNH
    * Chọn khung hình ở giữa một pha bóng toàn cảnh, ĐỪNG chọn khung hình đầu video —
      phần mở đầu thường là cảnh cận hoặc bảng điểm.
    * Khoanh RỘNG hơn vạch biên một chút (khoảng 5-10% mỗi phía). Bóng ra ngoài vạch
      vẫn là bóng trong cuộc; khoanh sát quá sẽ cắt mất những cú đánh thật ở góc sân.
    * 4 đỉnh là đủ cho một sân hình thang. Thêm đỉnh nếu muốn cắt bớt phần khán đài
      lọt vào góc trên.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:                 # gần như luôn là chạy nhầm interpreter
    import sys as _s
    _s.exit(
        f"\n[LỖI] Không import được '{exc.name}' — đang chạy bằng:\n"
        f"       {_s.executable}\n\n"
        "Đây là interpreter KHÁC với môi trường tennis_ml. Trong PowerShell, lệnh 'python'\n"
        "trỏ tới bản Python cài hệ thống chứ không phải conda env của dự án.\n\n"
        "Chạy bằng đúng interpreter:\n"
        "   C:\\Users\\dumbe\\.conda\\envs\\tennis_ml\\python.exe scripts\\tools\\select_court.py --frame 3000\n\n"
        "Hoặc gõ gọn:\n"
        "   .\\select_court.bat\n"
    )

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WIN = "Chon mat san  |  chuot trai = them dinh  |  z = bo  |  c = xoa  |  s = luu  |  q = thoat"


# --------------------------------------------------------------------------- #
class Picker:
    """Giữ danh sách đỉnh ở TOẠ ĐỘ GỐC của video, không phải toạ độ cửa sổ.

    Cửa sổ hiển thị thường bị thu nhỏ để vừa màn hình. Nếu lưu thẳng toạ độ chuột
    thì đa giác sẽ lệch đúng bằng hệ số thu nhỏ — một lỗi âm thầm, chỉ lộ ra khi
    thấy tỉ lệ loại bỏ cao bất thường lúc chạy thật. Nên mọi thứ quy về toạ độ gốc
    ngay tại chỗ nhận sự kiện chuột.
    """

    def __init__(self, scale: float) -> None:
        self.pts: list[list[int]] = []
        self.scale = scale

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append([int(round(x / self.scale)), int(round(y / self.scale))])
        elif event == cv2.EVENT_RBUTTONDOWN and self.pts:
            self.pts.pop()


# --------------------------------------------------------------------------- #
def draw(frame: np.ndarray, pts: list[list[int]], scale: float,
         frame_idx: int, total: int) -> np.ndarray:
    vis = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else frame.copy()

    if pts:
        sp = np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in pts], np.int32)

        if len(sp) >= 3:                       # tô nền mờ để thấy ngay vùng sẽ được giữ lại
            overlay = vis.copy()
            cv2.fillPoly(overlay, [sp], (0, 200, 0))
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
            cv2.polylines(vis, [sp], True, (0, 255, 0), 2, cv2.LINE_AA)
        elif len(sp) == 2:
            cv2.line(vis, tuple(sp[0]), tuple(sp[1]), (0, 255, 0), 2, cv2.LINE_AA)

        for i, (px, py) in enumerate(sp):
            cv2.circle(vis, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, str(i + 1), (px + 9, py - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    bar = [
        f"frame {frame_idx}/{total}   dinh: {len(pts)}",
        "chuot trai=them  z/chuot phai=bo  c=xoa  , .=+-30f  [ ]=+-300f  s=luu  q=thoat",
    ]
    if len(pts) < 3:
        bar.append("Can it nhat 3 dinh moi luu duoc.")
    for i, line in enumerate(bar):
        y = 26 + i * 24
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Khoanh đa giác mặt sân, lưu ra JSON.")
    ap.add_argument("--source", default="VideoForTest.mp4", help="Video nguồn")
    ap.add_argument("--out", default="court_mask.json", help="File JSON đầu ra")
    ap.add_argument("--frame", type=int, default=26325,
                    help="Khung hình bắt đầu. Mặc định 26325 — đã quét cả video và đây là "
                         "một trong ba khung hình thấy trọn mặt sân (26325 / 47925 / 31365)")
    ap.add_argument("--max-width", type=int, default=1600,
                    help="Bề rộng cửa sổ tối đa; ảnh lớn hơn sẽ được thu nhỏ để vừa màn hình")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        print(f"[LỖI] Không thấy video: {args.source}")
        return 2

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[LỖI] OpenCV không mở được: {args.source}")
        return 2

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, args.max_width / max(W, 1))
    print(f"[Video] {args.source} — {W}x{H}, {total} khung hình")
    print(f"[Cửa sổ] thu nhỏ x{scale:.3f} để vừa màn hình (toạ độ vẫn lưu theo {W}x{H})")

    idx = max(0, min(args.frame, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        print("[LỖI] Không đọc được khung hình.")
        cap.release()
        return 2

    picker = Picker(scale)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, picker.on_mouse)

    def seek(delta: int) -> None:
        nonlocal idx, frame
        new = max(0, min(idx + delta, max(total - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, new)
        ok2, fr2 = cap.read()
        if ok2:
            idx, frame = new, fr2

    while True:
        cv2.imshow(WIN, draw(frame, picker.pts, scale, idx, total))
        k = cv2.waitKey(20) & 0xFF

        if k in (ord("q"), 27):
            print("[Thoát] Không lưu gì.")
            break

        if k == ord("c"):
            picker.pts.clear()
        elif k == ord("z"):
            if picker.pts:
                picker.pts.pop()
        elif k == ord(","):
            seek(-30)
        elif k == ord("."):
            seek(30)
        elif k == ord("["):
            seek(-300)
        elif k == ord("]"):
            seek(300)

        elif k == ord("s"):
            if len(picker.pts) < 3:
                print("[!] Cần ít nhất 3 đỉnh mới tạo được đa giác.")
                continue

            poly = np.array(picker.pts, np.int32).reshape(-1, 1, 2)
            area = abs(cv2.contourArea(poly))
            payload = {
                "version": 1,
                "source": os.path.basename(args.source),
                "frame_index": idx,
                "frame_width": W,
                "frame_height": H,
                "points": [[int(x), int(y)] for x, y in picker.pts],
                "area_px": float(area),
                "area_frac": float(area / (W * H)),
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            preview = os.path.splitext(args.out)[0] + "_preview.jpg"
            cv2.imwrite(preview, draw(frame, picker.pts, scale, idx, total))

            print(f"\n[LƯU] {args.out}")
            print(f"   {len(picker.pts)} đỉnh, tham chiếu {W}x{H}, khung hình {idx}")
            print(f"   diện tích sân = {area:,.0f} px ({100 * area / (W * H):.1f}% khung hình)")
            print(f"   ảnh xem lại  -> {preview}")
            print("\nChạy thật:")
            print("   python realtime_inference.py --source VideoForTest.mp4 ^")
            print("     --model models\\lgbm_v10_savgol.joblib ^")
            print("     --yolo yolo26x.pt --yolo-imgsz 640 --yolo-half ^")
            print("     --yolo-max-box-frac 0.042 --max-window-span 8 ^")
            print(f"     --court-mask {args.out} --conf 0.25")
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
