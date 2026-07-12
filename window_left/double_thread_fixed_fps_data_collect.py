"""
Fixed 30 FPS double-threaded Leap Motion + webcam data collection.

Difference vs `single_thread_fixed_fps_data_collect.py`:
  - Webcam frames are captured on a background thread.
  - The main loop blocks until a *fresh* frame arrives (never consumes stale buffered frames).

This keeps CSV rows and video frames aligned by `frame_number`, while reducing the risk
of reading old frames from internal camera buffers when the processing loop lags.
"""

import leap
import cv2
import time
import csv
import os
import argparse
import sys
import threading
import queue
from datetime import datetime


# ──────────────────────────── constants ────────────────────────────
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
BONES = ["metacarpal", "proximal", "intermediate", "distal"]

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ~0.033 s per tick
DEFAULT_DURATION = 25  # seconds


# ──────────────────────────── helpers ──────────────────────────────
def zero_hand_dict(prefix):
    """Return a dict with all hand fields set to 0."""
    d = {}

    # palm
    for k in ["x", "y", "z", "vx", "vy", "vz", "nx", "ny", "nz", "dx", "dy", "dz", "width"]:
        d[f"{prefix}_palm_{k}"] = 0.0

    # arm
    for k in ["wrist_x", "wrist_y", "wrist_z", "elbow_x", "elbow_y", "elbow_z"]:
        d[f"{prefix}_{k}"] = 0.0

    # fingers & bones
    for f in FINGERS:
        for b in BONES:
            for k in ["sx", "sy", "sz", "ex", "ey", "ez", "width"]:
                d[f"{prefix}_{f}_{b}_{k}"] = 0.0

    return d


def build_csv_header():
    """Build the comprehensive CSV header for both hands."""
    header = [
        "frame_number",  # 0-based index matching the video frame
        "system_time",
        "leap_timestamp",
        "leap_frame_id",
    ]

    for side in ["left", "right"]:
        header += [
            f"{side}_confidence",
            f"{side}_grab_strength",
            f"{side}_pinch_strength",
        ]

        # palm
        header += [
            f"{side}_palm_{k}"
            for k in ["x", "y", "z", "vx", "vy", "vz", "nx", "ny", "nz", "dx", "dy", "dz", "width"]
        ]

        # arm
        header += [f"{side}_{k}" for k in ["wrist_x", "wrist_y", "wrist_z", "elbow_x", "elbow_y", "elbow_z"]]

        # fingers & bones
        for f in FINGERS:
            for b in BONES:
                header += [f"{side}_{f}_{b}_{k}" for k in ["sx", "sy", "sz", "ex", "ey", "ez", "width"]]

    return header


def extract_hand_row(event):
    """Extract a flat dict of hand data from a TrackingEvent (or None → zeros)."""
    row = {}

    # Defaults – both hands zeroed
    row.update(zero_hand_dict("left"))
    row.update(zero_hand_dict("right"))
    for side in ["left", "right"]:
        row[f"{side}_confidence"] = 0.0
        row[f"{side}_grab_strength"] = 0.0
        row[f"{side}_pinch_strength"] = 0.0

    row["leap_timestamp"] = 0
    row["leap_frame_id"] = 0

    if event is None:
        return row

    row["leap_timestamp"] = event.timestamp
    row["leap_frame_id"] = event.tracking_frame_id

    for hand in event.hands:
        side = "left" if str(hand.type).endswith("Left") else "right"

        row[f"{side}_confidence"] = hand.confidence
        row[f"{side}_grab_strength"] = hand.grab_strength
        row[f"{side}_pinch_strength"] = hand.pinch_strength

        # palm
        row[f"{side}_palm_x"] = hand.palm.position.x
        row[f"{side}_palm_y"] = hand.palm.position.y
        row[f"{side}_palm_z"] = hand.palm.position.z

        row[f"{side}_palm_vx"] = hand.palm.velocity.x
        row[f"{side}_palm_vy"] = hand.palm.velocity.y
        row[f"{side}_palm_vz"] = hand.palm.velocity.z

        row[f"{side}_palm_nx"] = hand.palm.normal.x
        row[f"{side}_palm_ny"] = hand.palm.normal.y
        row[f"{side}_palm_nz"] = hand.palm.normal.z

        row[f"{side}_palm_dx"] = hand.palm.direction.x
        row[f"{side}_palm_dy"] = hand.palm.direction.y
        row[f"{side}_palm_dz"] = hand.palm.direction.z

        row[f"{side}_palm_width"] = hand.palm.width

        # arm  (LeapC: prev = elbow, next = wrist)
        row[f"{side}_wrist_x"] = hand.arm.next_joint.x
        row[f"{side}_wrist_y"] = hand.arm.next_joint.y
        row[f"{side}_wrist_z"] = hand.arm.next_joint.z

        row[f"{side}_elbow_x"] = hand.arm.prev_joint.x
        row[f"{side}_elbow_y"] = hand.arm.prev_joint.y
        row[f"{side}_elbow_z"] = hand.arm.prev_joint.z

        # digits & bones
        for fi, digit in enumerate(hand.digits):
            finger = FINGERS[fi]
            for bi, bone in enumerate(digit.bones):
                bone_name = BONES[bi]

                sx, sy, sz = bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z
                ex, ey, ez = bone.next_joint.x, bone.next_joint.y, bone.next_joint.z

                row[f"{side}_{finger}_{bone_name}_sx"] = sx
                row[f"{side}_{finger}_{bone_name}_sy"] = sy
                row[f"{side}_{finger}_{bone_name}_sz"] = sz
                row[f"{side}_{finger}_{bone_name}_ex"] = ex
                row[f"{side}_{finger}_{bone_name}_ey"] = ey
                row[f"{side}_{finger}_{bone_name}_ez"] = ez
                row[f"{side}_{finger}_{bone_name}_width"] = bone.width

    return row


# ──────────────────────── Leap listener ────────────────────────────
class LatestFrameListener(leap.Listener):
    """Caches the most recent tracking event so the main loop can grab it."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._latest_event = None
        self._latest_event_time = None  # time.perf_counter() when _latest_event was updated
        self.connected = False
        self.device_serial = None

    def on_connection_event(self, event):
        self.connected = True
        print("Connected to Leap Motion")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        self.device_serial = info.serial
        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        now = time.perf_counter()
        with self._lock:
            self._latest_event = event
            self._latest_event_time = now

    def get_latest(self):
        """Return the most recent tracking event (may be None if none received yet)."""
        with self._lock:
            return self._latest_event

    def get_latest_with_time(self):
        """
        Return (event, system_time) for the most recent tracking event.

        system_time is a time.perf_counter() captured in on_tracking_event,
        or None if no event has been received yet.
        """
        with self._lock:
            return self._latest_event, self._latest_event_time


# ──────────────────────── camera helpers ───────────────────────────
def find_working_camera():
    """
    Find a working camera.  Tries camera index 1 first (common on Windows
    when an IR camera occupies index 0), then falls back to index 0.

    Returns:
        cv2.VideoCapture or None
    """
    if sys.platform == "win32":
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Camera 1 opened successfully")
                return cap
            cap.release()

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("Camera 0 opened successfully")
            return cap
        cap.release()

    return None


def create_video_writer(output_path, fps, width, height):
    """Try several codecs and return the first working VideoWriter."""
    for codec in ["mp4v", "MJPG", "XVID"]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


class BufferlessVideoCapture:
    """
    Background camera reader that keeps only the latest frame (StackOverflow pattern).

    - Reader thread continuously calls `cap.read()`
    - A 1-slot queue stores only the most recent frame
    - `read()` blocks until a frame is available and *consumes* it (removes from queue)
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._last_capture_time = None  # time.perf_counter() when the most recent frame was captured
        self._time_lock = threading.Lock()

        self._thread = threading.Thread(target=self._reader, name="BufferlessVideoCaptureReader")
        self._thread.daemon = True
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # If the camera is failing, avoid a tight spin.
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            with self._time_lock:
                self._last_capture_time = now

            # Keep only most recent unconsumed frame
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                # Extremely rare due to get_nowait above; safe to ignore.
                pass

    def read(self, timeout_s=None):
        """
        Block until a frame is available, then consume it.

        Returns:
            (ret, frame)
        """
        if timeout_s is None:
            frame = self._q.get()
            if frame is None:
                return False, None
            return True, frame

        try:
            frame = self._q.get(timeout=timeout_s)
        except queue.Empty:
            return False, None
        if frame is None:
            return False, None
        return True, frame

    def get_last_capture_time(self):
        """
        Return time.perf_counter() for when the most recent frame was captured
        by the reader thread, or None if no frame has been captured yet.
        """
        with self._time_lock:
            return self._last_capture_time

    def release(self):
        self._stop.set()
        # Unblock any waiting consumer
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        # Best-effort join (thread is daemon anyway)
        self._thread.join(timeout=1.0)
        self._cap.release()

    def get(self, prop_id):
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        return self._cap.set(prop_id, value)


# ──────────────────────── main loop ────────────────────────────────
def main():
    """Double-threaded, fixed-30-FPS data collection loop (fresh webcam frames)."""
    print("=" * 60)
    print("  Fixed 30 FPS  –  Leap Motion + Webcam Data Collection")
    print("  (double-threaded bufferless webcam)")
    print("=" * 60)
    print()

    # ── configuration ──
    parser = argparse.ArgumentParser(description="Collect Leap Motion and Video Data.")
    parser.add_argument("--user", type=str, default="user1", help="Name of the user for organizing the saving directory (default: user1)")
    args = parser.parse_args()

    duration = DEFAULT_DURATION
    fps = TARGET_FPS
    total_frames = duration * fps

    # ── file paths ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    user_dataset_dir = os.path.join(base_dir, "dataset", args.user)
    leap_data_dir = os.path.join(user_dataset_dir, "leap_data")
    video_data_dir = os.path.join(user_dataset_dir, "video_data")
    
    os.makedirs(leap_data_dir, exist_ok=True)
    os.makedirs(video_data_dir, exist_ok=True)

    csv_path = os.path.join(leap_data_dir, f"leap_data_{timestamp}.csv")
    video_path = os.path.join(video_data_dir, f"sign_language_{timestamp}.mp4")
    frame_ts_path = os.path.join(video_data_dir, f"video_frames_{timestamp}.csv")

    # ── initialise Leap Motion ──
    print("Initializing Leap Motion...")
    listener = LatestFrameListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    # ── initialise camera ──
    print("Initializing camera...")
    raw_cap = find_working_camera()
    if raw_cap is None or not raw_cap.isOpened():
        print("Error: Could not open any camera")
        print("  1. Check that the camera is connected and not used by another app")
        print("  2. Check camera permissions")
        connection.remove_listener(listener)
        return

    # warm-up: flush a few frames so auto-exposure / white-balance settle
    print("Warming up camera...")
    for _ in range(15):
        raw_cap.read()

    cap = BufferlessVideoCapture(raw_cap)

    # camera dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ── video writer ──
    video_writer = create_video_writer(video_path, fps, width, height)
    if video_writer is None:
        print("Error: Could not initialise video writer")
        cap.release()
        connection.remove_listener(listener)
        return

    # ── CSV files ──
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    fieldnames = build_csv_header()
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()

    frame_ts_file = open(frame_ts_path, "w", newline="", encoding="utf-8")
    frame_ts_writer = csv.writer(frame_ts_file)
    frame_ts_writer.writerow(
        [
            "frame_number",
            "system_time",
            "cam_capture_time",
            "leap_update_time",
            "tick_drift_ms",
            "video_write_ms",
            "cam_read_ms",
            "leap_read_ms",
            "csv_write_ms",
        ]
    )
    frame_ts_file.flush()

    try:
        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            print("Leap connection established. Waiting for device...")
            time.sleep(1.0)

            print(f"\nRecording: {duration}s  @  {fps} FPS  ({total_frames} frames)")
            print(f"  Leap CSV  -> {csv_path}")
            print(f"  Video     -> {video_path}")
            print(f"  Frame TS  -> {frame_ts_path}\n")

            # countdown
            for c in [3, 2, 1]:
                print(f"  Starting in {c}...")
                time.sleep(1.0)
            print("  GO!\n")

            # ── fixed-rate loop ──
            recording_start = time.perf_counter()
            frame_number = 0
            missed_cam_frames = 0
            last_printed_second = -1
            while frame_number < total_frames:
                tick_start = time.perf_counter()

                # Print elapsed time indicator once per second
                elapsed_since_start = tick_start - recording_start
                current_second = int(elapsed_since_start)
                if current_second != last_printed_second:
                    remaining = duration - current_second
                    print(f"  [t]  {current_second}s / {duration}s elapsed  ({remaining}s remaining)", flush=True)
                    last_printed_second = current_second

                # 1) Read one webcam frame (blocks until a new frame arrives if empty)
                cam_read_start = time.perf_counter()
                ret, cam_frame = cap.read()
                cam_read_ms = (time.perf_counter() - cam_read_start) * 1000
                cam_capture_time = cap.get_last_capture_time()

                # 2) Grab latest Leap tracking data
                leap_read_start = time.perf_counter()
                tracking_event, leap_update_time = listener.get_latest_with_time()
                leap_read_ms = (time.perf_counter() - leap_read_start) * 1000
                row = extract_hand_row(tracking_event)
                row["frame_number"] = frame_number
                row["system_time"] = tick_start

                # 3) Save both
                csv_write_start = time.perf_counter()
                csv_writer.writerow(row)
                csv_write_ms = (time.perf_counter() - csv_write_start) * 1000

                video_write_ms = 0.0
                if ret and cam_frame is not None:
                    write_start = time.perf_counter()
                    video_writer.write(cam_frame)
                    video_write_ms = (time.perf_counter() - write_start) * 1000
                else:
                    missed_cam_frames += 1

                # Record frame timestamp + drift for QA
                tick_drift_ms = (time.perf_counter() - tick_start) * 1000
                frame_ts_writer.writerow(
                    [
                        frame_number,
                        tick_start,
                        "" if cam_capture_time is None else cam_capture_time,
                        "" if leap_update_time is None else leap_update_time,
                        f"{tick_drift_ms:.2f}",
                        f"{video_write_ms:.2f}",
                        f"{cam_read_ms:.2f}",
                        f"{leap_read_ms:.2f}",
                        f"{csv_write_ms:.2f}",
                    ]
                )

                frame_number += 1

                # 4) Sleep the remainder of the tick to hold 30 FPS (may be negative if we had to wait for a new cam frame)
                elapsed = time.perf_counter() - tick_start
                sleep_time = FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # ── summary ──
            total_elapsed = time.perf_counter() - recording_start
            actual_fps = frame_number / total_elapsed if total_elapsed > 0 else 0

            csv_file.flush()
            frame_ts_file.flush()

            print("Recording complete!")
            print(f"  Frames recorded  : {frame_number}")
            print(f"  Missed cam frames: {missed_cam_frames}")
            print(f"  Actual duration  : {total_elapsed:.3f}s")
            print(f"  Effective FPS    : {actual_fps:.2f}")
            print(f"  Leap CSV         : {csv_path}")
            print(f"  Video            : {video_path}")
            print(f"  Frame timestamps : {frame_ts_path}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        csv_file.close()
        frame_ts_file.close()
        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        connection.remove_listener(listener)


if __name__ == "__main__":
    main()

