#!/usr/bin/env python3
"""Integrated manual robot dashboard.

Controls:
    Arrow keys  Hold to drive and tank turn (same calibration as feb.py)
    I           Toggle intake relay on PCA9685 channel 5
    F           Toggle flywheel relay on PCA9685 channel 4
    P           Pivot forward until the GPIO27 forward limit is reached
    B           Pivot backward until the GPIO17 backward limit is reached
    =           Hold for manual pivot forward (limit override)
    -           Hold for manual pivot backward (limit override)
    Space/Esc   Stop every motor and relay

The upper display is split evenly between the camera and LD19 lidar. The
BNO085 readings and actuator controls are below both views.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

from feb import (
    FORWARD_MAX_US,
    LEFT_MOTORS,
    MOTOR_REVERSED,
    NEUTRAL_US,
    REVERSE_MAX_US,
    RIGHT_MOTORS,
    RAMP_STEP_US,
    START_THROTTLE_PERCENT,
    TURN_THROTTLE_PERCENT,
    UPDATE_MS,
    FourEscDriver,
)
from imut import IMUConnection
from lidar import LidarPacket, LidarReader, open_lidar_serial
from lsp import LimitInputs


# The physical testing shows that P/= reaches the switch on GPIO27 and B/-
# reaches the switch on GPIO17. Keep these assignments matched to the actual
# pivot directions so a command away from a pressed limit is permitted.
FORWARD_LIMIT_GPIO = 27
BACKWARD_LIMIT_GPIO = 17
IMU_ADDRESS = 0x4B
PIVOT_SPEED_PERCENT = 20
LIMIT_POLL_MS = 20
PIVOT_MAX_TRAVEL_SECONDS = 30.0
SENSOR_UPDATE_MS = 40
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
MM_PER_INCH = 25.4

# Physical lidar placement. Robot coordinates use +Y forward and +X right.
LIDAR_LEFT_OF_CENTER_IN = 4.0
CHASSIS_WIDTH_IN = 16.0
CHASSIS_LENGTH_IN = 17.5
# The photos show the optical center behind the intake's leading edge rather
# than exactly on the front edge of the chassis.
LIDAR_BEHIND_FRONT_EDGE_IN = 2.0
LIDAR_FROM_CENTER_FORWARD_IN = (
    CHASSIS_LENGTH_IN / 2.0 - LIDAR_BEHIND_FRONT_EDGE_IN
)
# The LD19 defines its physical front as raw 0 degrees and increases clockwise.
# Its front faces outward, away from the intake, which is robot-left here.
LIDAR_ANGLE_OFFSET_DEG = 270.0
# The low inboard/right view is occluded by the intake bracket, chassis, and
# wheel visible in the mounting photos. Angles use 0=robot front, 90=right.
LIDAR_BLOCKED_CENTER_DEG = 90.0
LIDAR_BLOCKED_WIDTH_DEG = 160.0
CHASSIS_MASK_MARGIN_IN = 0.5

LIDAR_MIN_RANGE_MM = 20
LIDAR_MAX_RANGE_M = 12.0
LIDAR_MIN_INTENSITY = 1
LIDAR_POINT_TTL_SECONDS = 0.40


def angle_in_sector(angle_deg: float, center_deg: float, width_deg: float) -> bool:
    """Return whether an angle lies inside a circular sector."""
    width = max(0.0, min(360.0, width_deg))
    difference = (angle_deg - center_deg + 180.0) % 360.0 - 180.0
    return abs(difference) <= width / 2.0


class CameraSource:
    name = "camera"

    def read(self):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class Picamera2Source(CameraSource):
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        rotate_180: bool,
    ) -> None:
        try:
            from libcamera import Transform
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError(
                "Picamera2 is unavailable. Install it with: "
                "sudo apt install -y python3-picamera2"
            ) from error

        cameras = Picamera2.global_camera_info()
        if not cameras:
            raise RuntimeError("Picamera2 found no cameras")
        if not 0 <= index < len(cameras):
            raise RuntimeError(
                f"Camera {index} is unavailable; detected {len(cameras)} camera(s)"
            )

        info = cameras[index]
        self.name = f"Picamera2 {index}: {info.get('Model', 'unknown')}"
        self.camera = Picamera2(index)
        config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            transform=Transform(hflip=rotate_180, vflip=rotate_180),
            buffer_count=4,
        )
        self.camera.configure(config)
        self.camera.start()

    def read(self):
        try:
            return True, self.camera.capture_array("main")
        except RuntimeError:
            return False, None

    def close(self) -> None:
        try:
            self.camera.stop()
        finally:
            self.camera.close()


class OpenCVCameraSource(CameraSource):
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        rotate_180: bool,
    ) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is missing: python3 -m pip install opencv-python") from error

        self.cv2 = cv2
        self.camera = cv2.VideoCapture(index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.rotate_180 = rotate_180
        self.name = f"OpenCV/USB camera {index}"
        if not self.camera.isOpened():
            raise RuntimeError(f"OpenCV could not open camera {index}")

    def read(self):
        ok, frame = self.camera.read()
        if ok and self.rotate_180:
            frame = self.cv2.flip(frame, -1)
        return ok, frame

    def close(self) -> None:
        self.camera.release()


def create_camera(args: argparse.Namespace) -> CameraSource:
    if args.camera_backend == "picamera2":
        return Picamera2Source(
            args.camera,
            args.camera_width,
            args.camera_height,
            args.rotate_180,
        )
    if args.camera_backend == "opencv":
        return OpenCVCameraSource(
            args.camera,
            args.camera_width,
            args.camera_height,
            args.rotate_180,
        )

    try:
        return Picamera2Source(
            args.camera,
            args.camera_width,
            args.camera_height,
            args.rotate_180,
        )
    except Exception as picamera_error:
        try:
            return OpenCVCameraSource(
                args.camera,
                args.camera_width,
                args.camera_height,
                args.rotate_180,
            )
        except Exception as opencv_error:
            raise RuntimeError(
                f"No usable camera. Picamera2: {picamera_error}; "
                f"OpenCV: {opencv_error}"
            ) from opencv_error


class CameraReader(threading.Thread):
    def __init__(self, camera: CameraSource, output: queue.Queue) -> None:
        super().__init__(name="present-camera", daemon=True)
        self.camera = camera
        self.output = output
        self.stop_event = threading.Event()

    def _send_latest(self, item) -> None:
        try:
            self.output.put_nowait(item)
        except queue.Full:
            try:
                self.output.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output.put_nowait(item)
            except queue.Full:
                pass

    def run(self) -> None:
        consecutive_failures = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = self.camera.read()
                if ok and frame is not None:
                    consecutive_failures = 0
                    self._send_latest(("frame", frame))
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        self._send_latest(("error", "Camera stopped returning frames"))
                        time.sleep(0.2)
        except Exception as error:
            if not self.stop_event.is_set():
                self._send_latest(("error", str(error)))

    def stop(self) -> None:
        self.stop_event.set()


class IMUReader(threading.Thread):
    def __init__(self, imu: IMUConnection, output: queue.Queue) -> None:
        super().__init__(name="present-imu", daemon=True)
        self.imu = imu
        self.output = output
        self.stop_event = threading.Event()

    def _send_latest(self, item) -> None:
        try:
            self.output.put_nowait(item)
        except queue.Full:
            try:
                self.output.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output.put_nowait(item)
            except queue.Full:
                pass

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._send_latest(("reading", self.imu.read()))
            except Exception as error:
                self._send_latest(("error", str(error)))
            time.sleep(0.05)

    def stop(self) -> None:
        self.stop_event.set()


@dataclass
class RobotLidarPoint:
    x_mm: float
    y_mm: float
    distance_mm: int
    intensity: int
    received_at: float


class PresentApp:
    def __init__(
        self,
        root: tk.Tk,
        driver: FourEscDriver,
        limits: LimitInputs,
        imu: IMUConnection | None,
        imu_error: str,
        camera: CameraSource | None,
        camera_error: str,
        lidar_connection,
        lidar_port: str,
        lidar_error: str,
        args: argparse.Namespace,
    ) -> None:
        self.root = root
        self.driver = driver
        self.limits = limits
        self.imu = imu
        self.camera = camera
        self.lidar_connection = lidar_connection
        self.args = args

        self.closed = False
        self.throttle_percent = START_THROTTLE_PERCENT
        self.current = NEUTRAL_US.copy()
        self.target = NEUTRAL_US.copy()
        self.drive_keys: set[str] = set()
        self.release_jobs: dict[str, str] = {}
        self.latched_keys: set[str] = set()
        self.latch_release_jobs: dict[str, str] = {}

        self.intake_on = False
        self.flywheel_on = False
        self.pivot_direction = 0
        self.pivot_mode = "stopped"
        self.pivot_target_limit: int | None = None
        self.pivot_ignored_limit: int | None = None
        self.pivot_started = 0.0

        self.motor_job = None
        self.sensor_job = None
        self.limit_job = None

        self.camera_queue: queue.Queue = queue.Queue(maxsize=2)
        self.camera_reader = (
            CameraReader(camera, self.camera_queue) if camera is not None else None
        )
        self.camera_photo = None
        self.latest_camera_frame = None

        self.imu_queue: queue.Queue = queue.Queue(maxsize=3)
        self.imu_reader = IMUReader(imu, self.imu_queue) if imu is not None else None

        self.lidar_queue: queue.Queue = queue.Queue(maxsize=300)
        self.lidar_reader = (
            LidarReader(lidar_connection, self.lidar_queue)
            if lidar_connection is not None
            else None
        )
        self.lidar_points: dict[int, RobotLidarPoint] = {}
        self.lidar_latest_packet = 0.0
        self.lidar_speed = 0
        self.lidar_stream_error = lidar_error

        self.root.title("Robot Presentation Dashboard")
        self.root.geometry("1500x960")
        self.root.minsize(1100, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        self.drive_status_var = tk.StringVar(value="Drive: STOPPED")
        self.pivot_status_var = tk.StringVar(value="Pivot: STOPPED")
        self.camera_status_var = tk.StringVar(
            value=camera_error or (camera.name if camera is not None else "Camera unavailable")
        )
        self.lidar_status_var = tk.StringVar(
            value=lidar_error or f"LiDAR: {lidar_port}"
        )
        self.intake_status_var = tk.StringVar(value="Intake: OFF")
        self.flywheel_status_var = tk.StringVar(value="Flywheel: OFF")
        self.motor_status_var = tk.StringVar(value="Motors: " + ", ".join(map(str, NEUTRAL_US)))
        self.lidar_range_m = tk.DoubleVar(value=args.lidar_display_range_m)
        self.imu_values = {
            name: tk.StringVar(value="--")
            for name in ("Yaw", "Pitch", "Roll", "Accel X", "Accel Y", "Accel Z")
        }
        self.imu_status_var = tk.StringVar(value=imu_error or "Waiting for IMU data...")
        self.limit_vars = [
            tk.StringVar(value="Forward limit: CHECKING"),
            tk.StringVar(value="Backward limit: CHECKING"),
        ]

        self.build_ui()
        self.bind_keys()

        if self.camera_reader is not None:
            self.camera_reader.start()
        if self.imu_reader is not None:
            self.imu_reader.start()
        if self.lidar_reader is not None:
            self.lidar_reader.start()

        self.update_motors()
        self.poll_limits()
        self.update_sensors()

    def build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        visuals = ttk.Frame(outer)
        visuals.grid(row=0, column=0, sticky="nsew")
        visuals.rowconfigure(0, weight=1)
        visuals.columnconfigure(0, weight=1, uniform="view")
        visuals.columnconfigure(1, weight=1, uniform="view")

        camera_box = ttk.LabelFrame(visuals, text="CAMERA — LEFT HALF", padding=4)
        camera_box.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        camera_box.rowconfigure(0, weight=1)
        camera_box.columnconfigure(0, weight=1)
        self.camera_label = tk.Label(
            camera_box,
            text="Waiting for camera...",
            background="#111827",
            foreground="white",
            font=("Arial", 15, "bold"),
        )
        self.camera_label.grid(row=0, column=0, sticky="nsew")
        ttk.Label(camera_box, textvariable=self.camera_status_var).grid(
            row=1, column=0, sticky="ew", pady=(3, 0)
        )

        lidar_box = ttk.LabelFrame(visuals, text="LIDAR — RIGHT HALF", padding=4)
        lidar_box.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        lidar_box.rowconfigure(0, weight=1)
        lidar_box.columnconfigure(0, weight=1)
        self.lidar_canvas = tk.Canvas(
            lidar_box,
            background="#071015",
            highlightthickness=0,
        )
        self.lidar_canvas.grid(row=0, column=0, sticky="nsew")
        lidar_footer = ttk.Frame(lidar_box)
        lidar_footer.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        ttk.Label(lidar_footer, textvariable=self.lidar_status_var).pack(side="left")
        ttk.Label(lidar_footer, text="Range").pack(side="left", padx=(12, 3))
        ttk.Scale(
            lidar_footer,
            from_=1.0,
            to=8.0,
            variable=self.lidar_range_m,
            orient="horizontal",
            length=130,
        ).pack(side="left")

        imu_box = ttk.LabelFrame(outer, text="BNO085 IMU — BELOW BOTH VIEWS", padding=7)
        imu_box.grid(row=1, column=0, sticky="ew", pady=(6, 3))
        for column in range(6):
            imu_box.columnconfigure(column, weight=1)
        units = ("°", "°", "°", "m/s²", "m/s²", "m/s²")
        for column, ((name, variable), unit) in enumerate(zip(self.imu_values.items(), units)):
            card = ttk.Frame(imu_box)
            card.grid(row=0, column=column, sticky="ew", padx=4)
            ttk.Label(card, text=name, font=("Arial", 10, "bold")).pack()
            ttk.Label(card, textvariable=variable, font=("Arial", 17, "bold")).pack()
            ttk.Label(card, text=unit).pack()
        ttk.Label(imu_box, textvariable=self.imu_status_var).grid(
            row=1, column=0, columnspan=6, pady=(4, 0)
        )

        # Motor, relay, pivot, and emergency-stop controls remain keyboard-only.
        # The only on-screen control is the requested drive-throttle slider.
        self.forward_limit_label = None
        self.backward_limit_label = None
        throttle_row = ttk.LabelFrame(outer, text="DRIVE THROTTLE", padding=7)
        throttle_row.grid(row=2, column=0, sticky="ew", pady=(3, 0))
        ttk.Label(throttle_row, text="Drive throttle").pack(side="left")
        self.throttle_label = ttk.Label(
            throttle_row,
            text=f"{self.throttle_percent}%",
            width=5,
        )
        self.throttle_label.pack(side="right")
        slider = ttk.Scale(
            throttle_row,
            from_=1,
            to=100,
            value=self.throttle_percent,
            orient="horizontal",
            command=self.update_throttle,
        )
        slider.pack(side="left", fill="x", expand=True, padx=8)

    def make_drive_button(self, parent, text: str, control: str, column: int) -> None:
        button = ttk.Button(parent, text=text)
        button.grid(row=0, column=column, sticky="ew", padx=2)
        button.bind("<ButtonPress-1>", lambda _event: self.press_drive(control))
        button.bind("<ButtonRelease-1>", lambda _event: self.release_drive(control, immediate=True))
        button.bind("<Leave>", lambda _event: self.release_drive(control, immediate=True))

    def make_manual_pivot_button(
        self,
        parent,
        text: str,
        direction: int,
        column: int,
    ) -> None:
        button = ttk.Button(parent, text=text)
        button.grid(row=0, column=column, sticky="ew", padx=2)
        button.bind("<ButtonPress-1>", lambda _event: self.start_manual_pivot(direction))
        button.bind("<ButtonRelease-1>", lambda _event: self.release_manual_pivot(direction))
        button.bind("<Leave>", lambda _event: self.release_manual_pivot(direction))

    def bind_keys(self) -> None:
        self.root.bind_all("<KeyPress>", self.key_press)
        self.root.bind_all("<KeyRelease>", self.key_release)
        self.root.focus_set()

    def key_press(self, event) -> None:
        key = event.keysym.lower()
        if key in ("up", "down", "left", "right"):
            self.cancel_release(key)
            self.press_drive(key)
        elif key in ("i", "f", "p", "b"):
            actions = {
                "i": self.toggle_intake,
                "f": self.toggle_flywheel,
                "p": lambda: self.start_auto_pivot(1),
                "b": lambda: self.start_auto_pivot(-1),
            }
            self.press_latched_key(key, actions[key])
        elif key in ("equal", "plus"):
            self.cancel_release("manual_forward")
            if "manual_forward" not in self.drive_keys:
                self.drive_keys.add("manual_forward")
                self.start_manual_pivot(1)
        elif key in ("minus", "underscore"):
            self.cancel_release("manual_backward")
            if "manual_backward" not in self.drive_keys:
                self.drive_keys.add("manual_backward")
                self.start_manual_pivot(-1)
        elif key in ("space", "escape"):
            self.stop_all()

    def key_release(self, event) -> None:
        key = event.keysym.lower()
        if key in ("up", "down", "left", "right"):
            self.release_drive(key)
        elif key in ("i", "f", "p", "b"):
            self.release_latched_key(key)
        elif key in ("equal", "plus"):
            self.schedule_manual_release("manual_forward", 1)
        elif key in ("minus", "underscore"):
            self.schedule_manual_release("manual_backward", -1)

    def cancel_release(self, name: str) -> None:
        job = self.release_jobs.pop(name, None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass

    def press_latched_key(self, name: str, action) -> None:
        job = self.latch_release_jobs.pop(name, None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        if name in self.latched_keys:
            return
        self.latched_keys.add(name)
        action()

    def release_latched_key(self, name: str) -> None:
        old_job = self.latch_release_jobs.pop(name, None)
        if old_job is not None:
            try:
                self.root.after_cancel(old_job)
            except tk.TclError:
                pass
        self.latch_release_jobs[name] = self.root.after(
            100,
            lambda: self.finish_latched_release(name),
        )

    def finish_latched_release(self, name: str) -> None:
        self.latch_release_jobs.pop(name, None)
        self.latched_keys.discard(name)

    def press_drive(self, control: str) -> None:
        if self.closed:
            return
        self.drive_keys.add(control)
        self.apply_drive_controls()

    def release_drive(self, control: str, immediate: bool = False) -> None:
        self.cancel_release(control)
        if immediate:
            self.finish_drive_release(control)
        else:
            self.release_jobs[control] = self.root.after(
                60,
                lambda: self.finish_drive_release(control),
            )

    def finish_drive_release(self, control: str) -> None:
        self.release_jobs.pop(control, None)
        self.drive_keys.discard(control)
        self.apply_drive_controls()

    def schedule_manual_release(self, key_name: str, direction: int) -> None:
        self.cancel_release(key_name)
        self.release_jobs[key_name] = self.root.after(
            60,
            lambda: self.finish_manual_release(key_name, direction),
        )

    def finish_manual_release(self, key_name: str, direction: int) -> None:
        self.release_jobs.pop(key_name, None)
        self.drive_keys.discard(key_name)
        self.release_manual_pivot(direction)

    def update_throttle(self, value: str) -> None:
        self.throttle_percent = int(round(float(value)))
        self.throttle_label.configure(text=f"{self.throttle_percent}%")
        self.apply_drive_controls()

    def motor_pulse(
        self,
        motor_index: int,
        direction: int,
        throttle_percent: int | None = None,
    ) -> int:
        if MOTOR_REVERSED[motor_index]:
            direction *= -1
        neutral = NEUTRAL_US[motor_index]
        throttle = self.throttle_percent if throttle_percent is None else throttle_percent
        fraction = max(0.0, min(1.0, throttle / 100.0))
        if direction > 0:
            return int(neutral + fraction * (FORWARD_MAX_US[motor_index] - neutral))
        if direction < 0:
            return int(neutral - fraction * (neutral - REVERSE_MAX_US[motor_index]))
        return neutral

    def set_side_targets(
        self,
        left_direction: int,
        right_direction: int,
        left_throttle: int | None = None,
        right_throttle: int | None = None,
    ) -> None:
        targets = NEUTRAL_US.copy()
        for index in LEFT_MOTORS:
            targets[index] = self.motor_pulse(index, left_direction, left_throttle)
        for index in RIGHT_MOTORS:
            targets[index] = self.motor_pulse(index, right_direction, right_throttle)
        self.target = targets

    def apply_drive_controls(self) -> None:
        if self.closed:
            return
        if "up" in self.drive_keys:
            self.set_side_targets(1, 1)
            status = f"Drive: FORWARD {self.throttle_percent}%"
        elif "down" in self.drive_keys:
            self.set_side_targets(-1, -1)
            status = f"Drive: REVERSE {self.throttle_percent}%"
        elif "left" in self.drive_keys:
            self.set_side_targets(-1, 1, TURN_THROTTLE_PERCENT, TURN_THROTTLE_PERCENT)
            status = f"Drive: TANK LEFT {TURN_THROTTLE_PERCENT}%"
        elif "right" in self.drive_keys:
            self.set_side_targets(1, -1, TURN_THROTTLE_PERCENT, TURN_THROTTLE_PERCENT)
            status = f"Drive: TANK RIGHT {TURN_THROTTLE_PERCENT}%"
        else:
            self.target = NEUTRAL_US.copy()
            status = "Drive: STOPPED"
        self.drive_status_var.set(status)

    @staticmethod
    def step_value(current: int, target: int) -> int:
        if current < target:
            return min(current + RAMP_STEP_US, target)
        if current > target:
            return max(current - RAMP_STEP_US, target)
        return current

    def update_motors(self) -> None:
        if self.closed:
            return
        previous = self.current.copy()
        self.current = [
            self.step_value(current, target)
            for current, target in zip(self.current, self.target)
        ]
        if self.current != previous:
            try:
                self.driver.set_all(self.current)
            except Exception as error:
                self.hardware_error(error)
                return
        self.motor_status_var.set("Motor µs: " + ", ".join(map(str, self.current)))
        self.motor_job = self.root.after(UPDATE_MS, self.update_motors)

    def toggle_intake(self) -> None:
        if self.closed:
            return
        requested = not self.intake_on
        try:
            self.driver.set_intake(requested)
        except Exception as error:
            self.hardware_error(error)
            return
        self.intake_on = requested
        self.intake_status_var.set(f"Intake: {'ON' if requested else 'OFF'}")

    def toggle_flywheel(self) -> None:
        if self.closed:
            return
        requested = not self.flywheel_on
        try:
            self.driver.set_flywheel(requested)
        except Exception as error:
            self.hardware_error(error)
            return
        self.flywheel_on = requested
        self.flywheel_status_var.set(f"Flywheel: {'ON' if requested else 'OFF'}")

    def read_limits(self) -> tuple[bool, bool]:
        return self.limits.triggered(0), self.limits.triggered(1)

    def start_auto_pivot(self, direction: int) -> None:
        self.start_pivot(direction, "auto")

    def start_manual_pivot(self, direction: int) -> None:
        self.start_pivot(direction, "manual")

    def start_pivot(self, direction: int, mode: str) -> None:
        if self.closed:
            return
        try:
            states = self.read_limits()
        except Exception as error:
            self.stop_pivot(f"Limit read failed: {error}")
            return

        if mode == "auto" and states[0] and states[1]:
            self.stop_pivot("Both limits triggered/open — check wiring")
            return

        target_limit = 0 if direction > 0 else 1
        opposite_limit = 1 - target_limit
        if mode == "auto" and states[target_limit]:
            name = "FORWARD" if target_limit == 0 else "BACKWARD"
            self.stop_pivot(f"{name} LIMIT ALREADY REACHED")
            return

        if mode == "auto":
            # Allow automatic motion away from the starting switch until the
            # mechanism has physically released it.
            self.pivot_ignored_limit = (
                opposite_limit if states[opposite_limit] else None
            )
            self.pivot_target_limit = target_limit
        else:
            # Manual =/- is an intentional limit override. It remains
            # hold-to-run and stops when the key is released.
            self.pivot_ignored_limit = None
            self.pivot_target_limit = None
        self.pivot_started = time.monotonic()
        try:
            self.driver.set_pivot(direction)
        except Exception as error:
            self.hardware_error(error)
            return

        self.pivot_direction = direction
        self.pivot_mode = mode
        direction_name = "FORWARD" if direction > 0 else "BACKWARD"
        mode_name = (
            "AUTO TO LIMIT"
            if mode == "auto"
            else "MANUAL LIMIT OVERRIDE — HOLD"
        )
        self.pivot_status_var.set(
            f"Pivot: {direction_name} — {mode_name} ({PIVOT_SPEED_PERCENT}%)"
        )

    def release_manual_pivot(self, direction: int) -> None:
        if self.pivot_mode == "manual" and self.pivot_direction == direction:
            self.stop_pivot("Pivot: STOPPED — MANUAL KEY RELEASED")

    def stop_pivot(self, status: str = "Pivot: STOPPED") -> None:
        was_moving = self.pivot_direction != 0
        self.pivot_direction = 0
        self.pivot_mode = "stopped"
        self.pivot_target_limit = None
        self.pivot_ignored_limit = None
        if was_moving:
            try:
                self.driver.set_pivot(0)
            except Exception as error:
                self.hardware_error(error)
                return
        self.pivot_status_var.set(status)

    def update_limit_labels(self, states: tuple[bool, bool]) -> None:
        labels = (self.forward_limit_label, self.backward_limit_label)
        names = ("Forward", "Backward")
        for index, (label, name) in enumerate(zip(labels, names)):
            if states[index]:
                self.limit_vars[index].set(f"{name} limit: TRIGGERED / OPEN")
                if label is not None:
                    label.configure(background="#b91c1c", foreground="white")
            else:
                self.limit_vars[index].set(f"{name} limit: CLEAR")
                if label is not None:
                    label.configure(background="#166534", foreground="white")

    def poll_limits(self) -> None:
        if self.closed:
            return
        try:
            states = self.read_limits()
            self.update_limit_labels(states)
        except Exception as error:
            self.stop_pivot(f"Limit read failed: {error}")
            self.limit_job = self.root.after(LIMIT_POLL_MS, self.poll_limits)
            return

        now = time.monotonic()
        if self.pivot_mode == "manual":
            # Do not stop manual override motion at either limit. Releasing
            # =/- or pressing Space/Escape still stops immediately.
            pass
        elif states[0] and states[1]:
            self.stop_pivot("Pivot: STOPPED — BOTH LIMITS OPEN")
        elif self.pivot_direction != 0:
            if self.pivot_ignored_limit is not None:
                if not states[self.pivot_ignored_limit]:
                    self.pivot_ignored_limit = None

            # The destination switch always stops motion. The switch at the
            # starting end is ignored only until it clears while moving away.
            target_hit = (
                self.pivot_target_limit is not None
                and states[self.pivot_target_limit]
            )
            opposite_limit = (
                None
                if self.pivot_target_limit is None
                else 1 - self.pivot_target_limit
            )
            opposite_fault = (
                opposite_limit is not None
                and states[opposite_limit]
                and opposite_limit != self.pivot_ignored_limit
            )

            if target_hit:
                name = "FORWARD" if self.pivot_target_limit == 0 else "BACKWARD"
                self.stop_pivot(f"Pivot: STOPPED — {name} LIMIT REACHED")
            elif opposite_fault:
                name = "FORWARD" if opposite_limit == 0 else "BACKWARD"
                self.stop_pivot(f"Pivot: SAFETY STOP — {name} LIMIT/OPEN WIRE")
            elif (
                self.pivot_mode == "auto"
                and now - self.pivot_started > PIVOT_MAX_TRAVEL_SECONDS
            ):
                self.stop_pivot("Pivot: STOPPED — 30 SECOND LIMIT TIMEOUT")

        self.limit_job = self.root.after(LIMIT_POLL_MS, self.poll_limits)

    def ingest_lidar_packet(self, packet: LidarPacket, now: float) -> None:
        self.lidar_latest_packet = now
        self.lidar_speed = packet.speed_deg_per_sec
        sensor_x = -self.args.lidar_left_of_center_in * MM_PER_INCH
        sensor_y = self.args.lidar_forward_of_center_in * MM_PER_INCH
        half_width = self.args.chassis_width_in * MM_PER_INCH / 2.0
        half_length = self.args.chassis_length_in * MM_PER_INCH / 2.0
        margin = self.args.chassis_mask_margin_in * MM_PER_INCH

        for raw in packet.points:
            if (
                raw.distance_mm < LIDAR_MIN_RANGE_MM
                or raw.distance_mm > LIDAR_MAX_RANGE_M * 1000.0
                or raw.intensity < LIDAR_MIN_INTENSITY
            ):
                continue

            direction = -1.0 if self.args.lidar_counterclockwise else 1.0
            angle = (
                self.args.lidar_angle_offset_deg + direction * raw.angle_deg
            ) % 360.0

            # Do not treat measurements through the photographed intake/
            # chassis/wheel obstruction as usable free-space information.
            if (
                self.args.lidar_blocked_sector
                and angle_in_sector(
                    angle,
                    self.args.lidar_blocked_center_deg,
                    self.args.lidar_blocked_width_deg,
                )
            ):
                continue

            radians = math.radians(angle)
            x_mm = sensor_x + raw.distance_mm * math.sin(radians)
            y_mm = sensor_y + raw.distance_mm * math.cos(radians)

            # Hide returns from the robot body, including the portion to the
            # right of the lidar that is physically blocked by the chassis.
            if (
                -half_width - margin <= x_mm <= half_width + margin
                and -half_length - margin <= y_mm <= half_length + margin
            ):
                continue

            bin_number = int(angle * 2.0) % 720
            self.lidar_points[bin_number] = RobotLidarPoint(
                x_mm,
                y_mm,
                raw.distance_mm,
                raw.intensity,
                now,
            )

    def update_lidar(self) -> None:
        now = time.monotonic()
        if self.lidar_reader is not None:
            for _ in range(300):
                try:
                    kind, payload = self.lidar_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "packet":
                    self.ingest_lidar_packet(payload, now)
                elif kind == "error":
                    self.lidar_stream_error = str(payload)

        cutoff = now - LIDAR_POINT_TTL_SECONDS
        self.lidar_points = {
            key: point
            for key, point in self.lidar_points.items()
            if point.received_at >= cutoff
        }
        points = list(self.lidar_points.values())

        if self.lidar_stream_error:
            self.lidar_status_var.set(f"LiDAR error: {self.lidar_stream_error}")
        elif self.lidar_latest_packet and now - self.lidar_latest_packet < 0.6:
            nearest = min((point.distance_mm for point in points), default=None)
            rpm = self.lidar_speed / 6.0
            nearest_text = f", nearest {nearest / 1000.0:.2f} m" if nearest else ""
            self.lidar_status_var.set(
                f"LiDAR live: {len(points)} points, {rpm:.0f} RPM{nearest_text}"
            )
        elif self.lidar_reader is not None:
            self.lidar_status_var.set("Waiting for LiDAR packets — check USB")

        self.draw_lidar(points)

    def draw_lidar(self, points: list[RobotLidarPoint]) -> None:
        canvas = self.lidar_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 200)
        center_x = width / 2.0
        center_y = height / 2.0
        range_mm = max(1.0, self.lidar_range_m.get()) * 1000.0
        pixels_per_mm = 0.46 * min(width, height) / range_mm

        def screen(x_mm: float, y_mm: float) -> tuple[float, float]:
            return center_x + x_mm * pixels_per_mm, center_y - y_mm * pixels_per_mm

        canvas.create_line(center_x, 6, center_x, height - 6, fill="#28414c")
        canvas.create_line(6, center_y, width - 6, center_y, fill="#28414c")
        ring_m = 1.0
        while ring_m <= self.lidar_range_m.get() + 0.001:
            radius = ring_m * 1000.0 * pixels_per_mm
            canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                outline="#1d3540",
            )
            canvas.create_text(
                center_x + 4,
                center_y - radius - 7,
                text=f"{ring_m:g}m",
                fill="#6c8b98",
                anchor="w",
            )
            ring_m += 1.0

        canvas.create_text(center_x, 10, text="ROBOT FRONT", fill="#d8edf5")
        canvas.create_text(8, center_y - 9, text="LEFT", fill="#829ba5", anchor="w")
        canvas.create_text(
            width - 8,
            center_y - 9,
            text="RIGHT",
            fill="#829ba5",
            anchor="e",
        )

        sensor_x = -self.args.lidar_left_of_center_in * MM_PER_INCH
        sensor_y = self.args.lidar_forward_of_center_in * MM_PER_INCH
        sx, sy = screen(sensor_x, sensor_y)

        # Show the photographed blind region instead of presenting the missing
        # inboard/right measurements as if the area had been scanned and clear.
        if self.args.lidar_blocked_sector:
            sector_width = max(
                0.0,
                min(360.0, self.args.lidar_blocked_width_deg),
            )
            start_angle = self.args.lidar_blocked_center_deg - sector_width / 2.0
            segment_count = max(8, int(sector_width / 5.0))
            wedge_points = [(sx, sy)]
            for index in range(segment_count + 1):
                angle = start_angle + sector_width * index / segment_count
                radians = math.radians(angle)
                wedge_points.append(
                    screen(
                        sensor_x + range_mm * math.sin(radians),
                        sensor_y + range_mm * math.cos(radians),
                    )
                )
            flat_points = [coordinate for point in wedge_points for coordinate in point]
            canvas.create_polygon(
                *flat_points,
                fill="#3a2024",
                outline="#7f3b44",
                stipple="gray25",
            )
            label_angle = math.radians(self.args.lidar_blocked_center_deg)
            label_distance = min(range_mm * 0.30, 900.0)
            label_x, label_y = screen(
                sensor_x + label_distance * math.sin(label_angle),
                sensor_y + label_distance * math.cos(label_angle),
            )
            canvas.create_text(
                label_x,
                label_y,
                text="BLOCKED\nINTAKE / CHASSIS / WHEEL",
                fill="#f3a6ad",
                justify="center",
                font=("Arial", 9, "bold"),
            )

        half_width = self.args.chassis_width_in * MM_PER_INCH / 2.0
        half_length = self.args.chassis_length_in * MM_PER_INCH / 2.0
        x1, y1 = screen(-half_width, half_length)
        x2, y2 = screen(half_width, -half_length)
        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill="#293038",
            outline="#9ca3af",
            stipple="gray50",
        )
        canvas.create_text(center_x, center_y, text="ROBOT", fill="#cbd5e1")

        # The intake spans the photographed front edge of the robot.
        intake_left = screen(-half_width, half_length)
        intake_right = screen(half_width, half_length)
        canvas.create_line(
            intake_left[0],
            intake_left[1],
            intake_right[0],
            intake_right[1],
            fill="#65d46e",
            width=5,
        )
        canvas.create_text(
            center_x,
            intake_left[1] - 10,
            text="INTAKE",
            fill="#8af394",
            font=("Arial", 9, "bold"),
        )

        canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="#facc15", outline="")
        # The LD19 physical front/raw-0 direction faces away from the intake.
        canvas.create_line(sx, sy, sx - 30, sy, fill="#facc15", width=3, arrow="last")
        canvas.create_text(
            sx,
            sy + 13,
            text=(
                f"LiDAR ({self.args.lidar_left_of_center_in:g} in left)\n"
                "raw 0° / sensor front → LEFT"
            ),
            fill="#facc15",
            anchor="n",
            justify="center",
        )

        closest = min(points, key=lambda point: point.distance_mm, default=None)
        for point in points:
            if math.hypot(point.x_mm, point.y_mm) > range_mm:
                continue
            px, py = screen(point.x_mm, point.y_mm)
            strength = max(0.0, min(1.0, point.intensity / 180.0))
            green = int(150 + 90 * strength)
            blue = int(155 + 95 * strength)
            canvas.create_oval(
                px - 2,
                py - 2,
                px + 2,
                py + 2,
                fill=f"#28{green:02x}{blue:02x}",
                outline="",
            )
        if closest is not None and math.hypot(closest.x_mm, closest.y_mm) <= range_mm:
            px, py = screen(closest.x_mm, closest.y_mm)
            canvas.create_oval(px - 6, py - 6, px + 6, py + 6, outline="#ff4d4d", width=2)

    def update_camera(self) -> None:
        latest_frame = None
        while True:
            try:
                kind, payload = self.camera_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "frame":
                latest_frame = payload
            else:
                self.camera_status_var.set(f"Camera error: {payload}")
        if latest_frame is not None:
            self.latest_camera_frame = latest_frame
            self.camera_status_var.set(self.camera.name if self.camera is not None else "Camera")

        if self.latest_camera_frame is None:
            return
        try:
            import cv2

            frame = self.latest_camera_frame
            available_width = max(self.camera_label.winfo_width() - 4, 320)
            available_height = max(self.camera_label.winfo_height() - 4, 200)
            scale = min(
                available_width / frame.shape[1],
                available_height / frame.shape[0],
            )
            output_width = max(1, int(frame.shape[1] * scale))
            output_height = max(1, int(frame.shape[0] * scale))
            if output_width != frame.shape[1] or output_height != frame.shape[0]:
                frame = cv2.resize(frame, (output_width, output_height))
            ok, encoded = cv2.imencode(
                ".png",
                frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
            if not ok:
                raise RuntimeError("Could not encode camera image")
            self.camera_photo = tk.PhotoImage(data=encoded.tobytes(), format="png")
            self.camera_label.configure(image=self.camera_photo, text="")
        except Exception as error:
            self.camera_status_var.set(f"Camera display error: {error}")

    def update_imu(self) -> None:
        latest = None
        latest_error = None
        while True:
            try:
                kind, payload = self.imu_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "reading":
                latest = payload
            else:
                latest_error = payload
        if latest is not None:
            for name, value in zip(self.imu_values, latest):
                self.imu_values[name].set(f"{value:.2f}")
            self.imu_status_var.set("BNO085 live at 0x4B")
        elif latest_error is not None:
            self.imu_status_var.set(f"IMU read error: {latest_error}")

    def update_sensors(self) -> None:
        if self.closed:
            return
        self.update_camera()
        self.update_lidar()
        self.update_imu()
        self.sensor_job = self.root.after(SENSOR_UPDATE_MS, self.update_sensors)

    def stop_all(self) -> None:
        if self.closed:
            return
        self.drive_keys.clear()
        self.target = NEUTRAL_US.copy()
        self.current = NEUTRAL_US.copy()
        self.intake_on = False
        self.flywheel_on = False
        self.pivot_direction = 0
        self.pivot_mode = "stopped"
        self.pivot_target_limit = None
        self.pivot_ignored_limit = None
        self.drive_status_var.set("Drive: STOPPED")
        self.pivot_status_var.set("Pivot: STOPPED — ALL STOP")
        self.intake_status_var.set("Intake: OFF")
        self.flywheel_status_var.set("Flywheel: OFF")
        try:
            self.driver.stop_all_outputs()
        except Exception as error:
            self.hardware_error(error)

    def hardware_error(self, error: Exception) -> None:
        if self.closed:
            return
        try:
            self.driver.stop_all_outputs()
        except Exception:
            pass
        messagebox.showerror(
            "Motor controller error",
            (
                f"PCA9685 communication failed:\n\n{error}\n\n"
                "Disconnect all motor power now. The program cannot confirm "
                "that every output stopped."
            ),
            parent=self.root,
        )
        self.shutdown()

    def cancel_jobs(self) -> None:
        for job in (self.motor_job, self.sensor_job, self.limit_job):
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
        for jobs in (self.release_jobs, self.latch_release_jobs):
            for job in jobs.values():
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
            jobs.clear()

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.cancel_jobs()

        if self.camera_reader is not None:
            self.camera_reader.stop()
        if self.imu_reader is not None:
            self.imu_reader.stop()
        if self.lidar_reader is not None:
            self.lidar_reader.stop()

        close_error = None
        try:
            self.driver.close()
        except Exception as error:
            close_error = error

        if self.lidar_connection is not None:
            try:
                self.lidar_connection.close()
            except Exception:
                pass
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        if self.imu is not None:
            try:
                self.imu.close()
            except Exception:
                pass
        self.limits.close()

        for reader in (self.camera_reader, self.imu_reader, self.lidar_reader):
            if reader is not None:
                reader.join(timeout=0.5)

        if close_error is not None:
            messagebox.showerror(
                "Stop could not be confirmed",
                str(close_error),
                parent=self.root,
            )
        self.root.destroy()


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def sector_width(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 360.0:
        raise argparse.ArgumentTypeError("must be between 0 and 360 degrees")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated robot presentation dashboard")
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument(
        "--rotate-180",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rotate the current upside-down camera (default: on)",
    )
    parser.add_argument("--lidar-port", default="auto")
    parser.add_argument("--lidar-baud", type=int, default=230400)
    parser.add_argument(
        "--lidar-display-range-m",
        type=positive_float,
        default=4.0,
    )
    parser.add_argument(
        "--lidar-left-of-center-in",
        type=float,
        default=LIDAR_LEFT_OF_CENTER_IN,
    )
    parser.add_argument(
        "--lidar-forward-of-center-in",
        type=float,
        default=LIDAR_FROM_CENTER_FORWARD_IN,
        help="front-mounted lidar position measured forward from robot center",
    )
    parser.add_argument(
        "--lidar-angle-offset-deg",
        type=float,
        default=LIDAR_ANGLE_OFFSET_DEG,
        help=(
            "270 maps the photographed outward-facing LD19 front/raw 0 degrees "
            "to robot-left"
        ),
    )
    parser.add_argument("--lidar-counterclockwise", action="store_true")
    parser.add_argument(
        "--lidar-blocked-sector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hide and mark the photographed inboard/right obstruction",
    )
    parser.add_argument(
        "--lidar-blocked-center-deg",
        type=float,
        default=LIDAR_BLOCKED_CENTER_DEG,
        help="center of blocked sector in robot coordinates (90 is right)",
    )
    parser.add_argument(
        "--lidar-blocked-width-deg",
        type=sector_width,
        default=LIDAR_BLOCKED_WIDTH_DEG,
        help="width of blocked intake/chassis/wheel sector",
    )
    parser.add_argument("--chassis-width-in", type=positive_float, default=CHASSIS_WIDTH_IN)
    parser.add_argument("--chassis-length-in", type=positive_float, default=CHASSIS_LENGTH_IN)
    parser.add_argument(
        "--chassis-mask-margin-in",
        type=positive_float,
        default=CHASSIS_MASK_MARGIN_IN,
    )
    return parser


def show_startup_error(error: Exception) -> int:
    print(f"present.py failed: {error}", file=sys.stderr)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Robot dashboard failed", str(error), parent=root)
        root.destroy()
    except tk.TclError:
        pass
    return 1


def main() -> int:
    args = build_parser().parse_args()

    driver = None
    limits = None
    imu = None
    camera = None
    lidar_connection = None
    imu_error = ""
    camera_error = ""
    lidar_error = ""
    lidar_port = "not connected"

    try:
        driver = FourEscDriver()
        limits = LimitInputs(FORWARD_LIMIT_GPIO, BACKWARD_LIMIT_GPIO)

        try:
            imu = IMUConnection(IMU_ADDRESS)
        except Exception as error:
            imu_error = f"IMU unavailable: {error}"

        try:
            camera = create_camera(args)
        except Exception as error:
            camera_error = f"Camera unavailable: {error}"

        try:
            lidar_connection, lidar_port = open_lidar_serial(
                args.lidar_port,
                args.lidar_baud,
            )
        except Exception as error:
            lidar_error = f"LiDAR unavailable: {error}"

        root = tk.Tk()
        app = PresentApp(
            root,
            driver,
            limits,
            imu,
            imu_error,
            camera,
            camera_error,
            lidar_connection,
            lidar_port,
            lidar_error,
            args,
        )
        root.mainloop()
        if not app.closed:
            app.shutdown()
        return 0
    except Exception as error:
        if lidar_connection is not None:
            try:
                lidar_connection.close()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        if imu is not None:
            try:
                imu.close()
            except Exception:
                pass
        if limits is not None:
            limits.close()
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass
        return show_startup_error(error)


if __name__ == "__main__":
    raise SystemExit(main())