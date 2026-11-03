"""GestureSign - Demo cua so truc tiep OpenCV (Bao cao tien do: Buoc 3 - Phan doan cu chi).

Hien thi:
1. Video truc tiep tu webcam (OpenCV).
2. Ve khung xuong MediaPipe Holistic (Ban tay trai, phai, than tren va moi).
3. Thuat toan Segmenter tu dong phan doan cu chi (nhan biet khi tay di chuyen,
   chot phan doan khi tay dung yen >= 300ms hoac ha tay xuong).
4. HUD hien thi trang thai va thong so ky thuat (Frames, Duration, FPS).

Phim tat:
- 'q' hoac 'ESC': Thoat ung dung.
- 'r': Reset phien ghi nhan.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Cau hinh ho tro UTF-8 tren Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    FaceLandmarksConnections,
    HandLandmarksConnections,
    PoseLandmarksConnections,
    drawing_utils,
)

# Duong dan mac dinh den file model MediaPipe Holistic
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "web",
    "public",
    "vendor",
    "mediapipe",
    "models",
    "holistic_landmarker.task",
)


@dataclass
class CompletedSegment:
    segment_id: int
    frames_count: int
    duration_ms: float
    fps: float
    partial: bool


class GestureSegmenter:
    """Thuat toan ngat doan cu chi (Segmenter) tuong thich logic Buoc 3 cua GestureSign."""

    def __init__(
        self,
        gap_ms: float = 300.0,
        move_speed: float = 0.6,
        min_frames: int = 8,
        max_ms: float = 4000.0,
    ) -> None:
        self.gap_ms = gap_ms
        self.move_speed = move_speed
        self.min_frames = min_frames
        self.max_ms = max_ms

        self.state = "idle"  # "idle" hoac "signing"
        self.samples_count = 0
        self.start_time = 0.0
        self.last_move_time = 0.0
        self.prev_hand_pts: Optional[List[Tuple[float, float]]] = None
        self.smoothed_speed = 0.0

    def reset(self) -> None:
        self.state = "idle"
        self.samples_count = 0
        self.start_time = 0.0
        self.last_move_time = 0.0
        self.prev_hand_pts = None
        self.smoothed_speed = 0.0

    def update(
        self,
        t_ms: float,
        left_hand_pts: Optional[List[Tuple[float, float]]],
        right_hand_pts: Optional[List[Tuple[float, float]]],
        shoulder_width: float,
    ) -> Optional[dict]:
        """Cap nhat frame va tra ve ket qua phan doan neu vua ket thuc cu chi."""
        curr_pts: List[Tuple[float, float]] = []
        if left_hand_pts:
            curr_pts.extend(left_hand_pts)
        if right_hand_pts:
            curr_pts.extend(right_hand_pts)

        has_hands = len(curr_pts) > 0
        inst_speed = 0.0

        if has_hands and self.prev_hand_pts and len(curr_pts) == len(self.prev_hand_pts):
            dists = [
                math.hypot(c[0] - p[0], c[1] - p[1])
                for c, p in zip(curr_pts, self.prev_hand_pts)
            ]
            avg_dist = sum(dists) / max(1, len(dists))
            ref_scale = max(0.05, shoulder_width)
            # Toc do di chuyen tuong doi so voi vai (ti le vai / giay)
            inst_speed = avg_dist / ref_scale * 30.0

        if has_hands:
            self.smoothed_speed = 0.6 * self.smoothed_speed + 0.4 * inst_speed
            self.prev_hand_pts = curr_pts
        else:
            self.smoothed_speed = 0.0
            self.prev_hand_pts = None

        is_moving = has_hands and (self.smoothed_speed >= self.move_speed)

        # Chuyen trang thai
        if self.state == "idle":
            if is_moving:
                self.state = "signing"
                self.start_time = t_ms
                self.last_move_time = t_ms
                self.samples_count = 1
                return None
            return None

        # Dang trong trang thai 'signing'
        self.samples_count += 1
        duration = t_ms - self.start_time

        if is_moving:
            self.last_move_time = t_ms

        # Kiem tra dieu kien ket thuc cu chi
        quiet_duration = t_ms - self.last_move_time
        exceeded_max = duration >= self.max_ms
        stopped = quiet_duration >= self.gap_ms

        if stopped or exceeded_max:
            kept_frames = self.samples_count
            self.state = "idle"
            self.prev_hand_pts = None

            if kept_frames >= self.min_frames:
                fps = (kept_frames / (duration / 1000.0)) if duration > 0 else 30.0
                return {
                    "frames_count": kept_frames,
                    "duration_ms": duration,
                    "fps": fps,
                }
            return None

        return None


# Cac duong noi khung than tren (Upper Body): vai, khuyu tay, co tay
UPPER_POSE_CONNECTIONS = [
    (11, 12),  # Vai trai - Vai phai
    (11, 13),  # Vai trai - Khuyu tay trai
    (13, 15),  # Khuyu tay trai - Co tay trai
    (12, 14),  # Vai phai - Khuyu tay phai
    (14, 16),  # Khuyu tay phai - Co tay phai
]


def draw_filtered_face(frame: np.ndarray, face_landmarks, h: int, w: int) -> None:
    """Ve khung xuong mat tinh gon: Chi ve Moi (Lips), Mat (Eyes) va Mui (Nose).

    Loai bo hoan toan 468 cham day dac cua MediaPipe Face Mesh tren tran, ma, cam.
    """
    n_pts = len(face_landmarks)

    def get_pt(idx: int) -> Optional[Tuple[int, int]]:
        if idx >= n_pts:
            return None
        lm = face_landmarks[idx]
        if not (math.isfinite(lm.x) and math.isfinite(lm.y)):
            return None
        return (int(lm.x * w), int(lm.y * h))

    # 1. Moi (Lips) - Mau cam san ho
    lips_color = (0, 140, 255)
    for c in FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
        pt1 = get_pt(c.start)
        pt2 = get_pt(c.end)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, lips_color, 1, cv2.LINE_AA)

    # 2. Mat trai & Mat phai (Eyes) - Mau vang / cyan
    eyes_color = (0, 240, 255)
    for c in list(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE) + list(
        FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
    ):
        pt1 = get_pt(c.start)
        pt2 = get_pt(c.end)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, eyes_color, 1, cv2.LINE_AA)

    # 3. Mui (Nose) - Mau xanh ngoc
    nose_color = (0, 255, 128)
    for c in FaceLandmarksConnections.FACE_LANDMARKS_NOSE:
        pt1 = get_pt(c.start)
        pt2 = get_pt(c.end)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, nose_color, 1, cv2.LINE_AA)

    # Ve cham nho tai cac khop moi, mat, mui
    key_indices = set()
    for c in (
        list(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
        + list(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
        + list(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
        + list(FaceLandmarksConnections.FACE_LANDMARKS_NOSE)
    ):
        key_indices.add(c.start)
        key_indices.add(c.end)

    for idx in key_indices:
        pt = get_pt(idx)
        if pt:
            cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_upper_body(frame: np.ndarray, pose_landmarks, h: int, w: int) -> None:
    """Ve khung than tren (vai, tay) thay vi ve ca chan/hong bi che khuat."""
    body_color = (255, 255, 255)
    n_pts = len(pose_landmarks)
    for idx1, idx2 in UPPER_POSE_CONNECTIONS:
        if idx1 < n_pts and idx2 < n_pts:
            lm1 = pose_landmarks[idx1]
            lm2 = pose_landmarks[idx2]
            vis1 = getattr(lm1, 'visibility', 1.0)
            vis2 = getattr(lm2, 'visibility', 1.0)
            if (vis1 is None or vis1 > 0.3) and (vis2 is None or vis2 > 0.3):
                if math.isfinite(lm1.x) and math.isfinite(lm1.y) and math.isfinite(lm2.x) and math.isfinite(lm2.y):
                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                    cv2.line(frame, pt1, pt2, body_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, pt1, 4, (0, 200, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt2, 4, (0, 200, 255), -1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    segmenter: GestureSegmenter,
    latest_segment: Optional[CompletedSegment],
    total_segments: int,
    fps_display: float,
) -> None:
    """Ve bang thong so (Heads-Up Display) len goc man hinh bang OpenCV."""
    # Bang nen xam trong suot goc tren ben trai
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (460, 185), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (460, 185), (80, 80, 80), 1)

    # Tieu de bao cao
    cv2.putText(
        frame,
        "GESTURESIGN - BAO CAO TIEN DO (BUOC 3)",
        (22, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Module: MediaPipe Holistic + Sign Segmenter",
        (22, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Trang thai hien tai
    if segmenter.state == "signing":
        blink = int(time.time() * 4) % 2 == 0
        dot_color = (0, 0, 255) if blink else (50, 50, 160)
        cv2.circle(frame, (30, 80), 7, dot_color, -1)
        cv2.putText(
            frame,
            f"DANG GHI CU CHI... ({segmenter.samples_count} frames)",
            (48, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.circle(frame, (30, 80), 6, (120, 120, 120), -1)
        cv2.putText(
            frame,
            "CHO CU CHI TAY (Di chuyen tay de bat dau)",
            (48, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Thong tin phan doan gan nhat
    cv2.line(frame, (20, 100), (450, 100), (60, 60, 60), 1)
    if latest_segment:
        status_txt = (
            f"Segment #{latest_segment.segment_id}: {latest_segment.frames_count} frames | "
            f"{latest_segment.duration_ms / 1000.0:.2f}s | {latest_segment.fps:.1f} FPS"
        )
        cv2.putText(
            frame,
            "PHAN DOAN THANH CONG GAN NHAT:",
            (22, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_txt,
            (22, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Chua co cu chi nao (Dung tay 300ms de chot).",
            (22, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    # Dong footer huong dan
    cv2.putText(
        frame,
        f"Tong da phan doan: {total_segments} | Camera: {fps_display:.1f} FPS | [q]: Thoat",
        (22, 172),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )


def run_demo(camera_id: int = 0, model_path: str = DEFAULT_MODEL_PATH) -> None:
    if not os.path.exists(model_path):
        print(f"[LOI] Khong tim thay file mo hinh MediaPipe tai: {model_path}")
        print("Vui long chay: cd web && npm run assets de tai mo hinh.")
        sys.exit(1)

    print("=" * 65)
    print(" GESTURESIGN - BAO CAO TIEN DO BUOC 3: PHAN DOAN CU CHI TAY")
    print(" Khoi tao MediaPipe Holistic Landmarker...")
    print("=" * 65)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo webcam ID: {camera_id}.")
        print("Hay kiem tra camera da cam hoac thu: python demo_webcam.py --camera-id 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    segmenter = GestureSegmenter()
    total_segments = 0
    latest_segment: Optional[CompletedSegment] = None

    prev_time = time.time()
    fps_display = 30.0
    start_app_time = time.time()

    print("[THONG BAO] Da mo webcam thanh cong. Cua so OpenCV dang chay.")
    print("-> Huong dan thao tac:")
    print("   1. Di chuyen tay: Trang thai bao 'DANG GHI CU CHI'")
    print("   2. Giu yen tay 300ms hoac ha tay: He thong chot 'PHAN DOAN THANH CONG'")
    print("   3. Nhan phim 'q' hoac 'ESC' de thoat.")
    print("-" * 65)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CANH BAO] Khong nhan duoc khung hinh tu camera.")
            break

        # Lat guong hinh anh
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Tinh FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

        # Chuyen sang RGB cho MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((curr_time - start_app_time) * 1000)

        # Nhan dien khung xuong
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        left_pts = None
        right_pts = None
        shoulder_width = 0.25

        # 1. Ban tay trai
        if results.left_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )
            left_pts = [(lm.x, lm.y) for lm in results.left_hand_landmarks]

        # 2. Ban tay phai
        if results.right_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )
            right_pts = [(lm.x, lm.y) for lm in results.right_hand_landmarks]

        # 3. Than tren (Pose - Chi ve khung vai va tay)
        if results.pose_landmarks:
            draw_upper_body(frame, results.pose_landmarks, h, w)
            try:
                ls = results.pose_landmarks[11]
                rs = results.pose_landmarks[12]
                shoulder_width = math.hypot(ls.x - rs.x, ls.y - rs.y)
            except IndexError:
                pass

        # 4. Khuon mat (Chi ve Moi, Mat va Mui - khong ve 468 cham day dac)
        if results.face_landmarks:
            draw_filtered_face(frame, results.face_landmarks, h, w)

        # Cap nhat Segmenter
        seg_result = segmenter.update(
            timestamp_ms,
            left_pts,
            right_pts,
            shoulder_width,
        )

        if seg_result:
            total_segments += 1
            latest_segment = CompletedSegment(
                segment_id=total_segments,
                frames_count=seg_result["frames_count"],
                duration_ms=seg_result["duration_ms"],
                fps=seg_result["fps"],
                partial=False,
            )
            print(
                f"[SEGMENT #{total_segments}] Da chot cu chi: "
                f"{latest_segment.frames_count} frames | "
                f"{latest_segment.duration_ms / 1000.0:.2f}s | "
                f"{latest_segment.fps:.1f} FPS"
            )

        # Ve bang HUD len man hinh
        draw_hud(frame, segmenter, latest_segment, total_segments, fps_display)

        # Hien thi cua so
        cv2.imshow("GestureSign - Bao cao tien do: Buoc 3 (Segmenter)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' hoac ESC
            break
        elif key == ord("r"):
            segmenter.reset()
            print("[THONG BAO] Da reset bo phan doan.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[THONG BAO] Da dong ung dung.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GestureSign OpenCV Live Demo - Buoc 3 (Phan doan cu chi tay)"
    )
    parser.add_argument("--camera-id", type=int, default=0, help="ID camera (mac dinh: 0)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Duong dan file holistic_landmarker.task",
    )
    args = parser.parse_args()

    run_demo(camera_id=args.camera_id, model_path=args.model_path)
