"""Shared command-line options for hand-tracking inputs.

Every teleop / debug / calibration script selects its input the same way:

    --input {avp,quest3,pico4,noitom,camera,realsense,video,replay}
    --file FILE                      # video / replay only

plus one option group per device.  Scripts change a default with
``parser.set_defaults(...)`` instead of redefining the option.
"""

from __future__ import annotations

import argparse

from .pico4 import (
    DEFAULT_DIRECT_PORT,
    DEFAULT_RELAY_HOST,
    DEFAULT_RELAY_PORT,
    DEFAULT_UDP_BROADCAST_PORT,
)

INPUT_CHOICES = ("avp", "quest3", "pico4", "noitom", "camera", "realsense", "video", "replay")

DEFAULT_FILES = {"video": "data/right.mp4", "replay": "data/avp1.pkl"}

# Headset / glove inputs have their own configs; MediaPipe-based inputs share one.
_CONFIG_DIRS = {"avp": "avp", "quest3": "quest3", "pico4": "pico4", "noitom": "noitom"}


def config_dir_for(input_type: str) -> str:
    """Return the ``config/<optimizer>/<dir>/`` name used by an input."""
    return _CONFIG_DIRS.get(input_type, "mediapipe")


def input_file(args: argparse.Namespace) -> str:
    """Return ``--file``, falling back to the default for video / replay."""
    return args.file or DEFAULT_FILES.get(args.input, "")


def add_input_args(
    parser: argparse.ArgumentParser,
    default: str = "video",
    choices: tuple[str, ...] = INPUT_CHOICES,
    recording: bool = False,
) -> None:
    """Register ``--input`` / ``--file`` and the per-device option groups."""
    # Prefix matching would let a typo with a single candidate, such as
    # ``--noitom-local-i``, silently bind to ``--noitom-local-ip``.
    parser.allow_abbrev = False

    source = parser.add_argument_group("input")
    source.add_argument("--input", default=default, choices=choices,
                        help="Input source (default: %(default)s)")
    source.add_argument("--file", default=None, metavar="FILE",
                        help="File for video / replay input (default: "
                             + ", ".join(f"{k}: {v}" for k, v in DEFAULT_FILES.items()) + ")")
    source.add_argument("--show-video", action="store_true",
                        help="Show RGB / landmark preview for camera-based inputs")

    avp = parser.add_argument_group("avp (Apple Vision Pro)")
    avp.add_argument("--avp-ip", default="192.168.50.127", metavar="IP",
                     help="Vision Pro IP (default: %(default)s)")

    quest3 = parser.add_argument_group("quest3 (Meta Quest 3 HTS)")
    quest3.add_argument("--quest3-port", type=int, default=9000, metavar="PORT",
                        help="Listener port (default: %(default)s)")
    quest3.add_argument("--quest3-protocol", default="udp", choices=["udp", "tcp"],
                        help="Transport protocol (default: %(default)s)")

    pico4 = parser.add_argument_group("pico4")
    pico4.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"],
                       help="relay daemon or direct TCP server (default: %(default)s)")
    pico4.add_argument("--pico4-relay-host", default=DEFAULT_RELAY_HOST, metavar="HOST",
                       help="relay: daemon host (default: %(default)s)")
    pico4.add_argument("--pico4-relay-port", type=int, default=DEFAULT_RELAY_PORT, metavar="PORT",
                       help="relay: daemon port (default: %(default)s)")
    pico4.add_argument("--pico4-port", type=int, default=DEFAULT_DIRECT_PORT, metavar="PORT",
                       help="direct: TCP listen port (default: %(default)s)")
    pico4.add_argument("--pico4-broadcast-port", type=int, default=DEFAULT_UDP_BROADCAST_PORT, metavar="PORT",
                       help="direct: UDP broadcast port (default: %(default)s)")

    noitom = parser.add_argument_group("noitom (PNS-G gloves via Axis Studio)")
    noitom.add_argument("--noitom-local-ip", default="192.168.5.25", metavar="IP",
                        help="This machine's IP, must match the Axis Studio destination (default: %(default)s)")
    noitom.add_argument("--noitom-local-port", type=int, default=8000, metavar="PORT",
                        help="Local UDP port (default: %(default)s)")
    noitom.add_argument("--noitom-server-ip", default="192.168.5.33", metavar="IP",
                        help="Axis Studio (Windows) IP (default: %(default)s)")
    noitom.add_argument("--noitom-server-port", type=int, default=9000, metavar="PORT",
                        help="Axis Studio BVH broadcast port (default: %(default)s)")

    playback = parser.add_argument_group("playback (video / replay)")
    playback.add_argument("--speed", type=float, default=1.0, metavar="X",
                          help="Playback speed (default: %(default)s)")
    playback.add_argument("--no-loop", action="store_true",
                          help="Stop at the end of the file instead of looping")

    if recording:
        record = parser.add_argument_group("recording")
        record.add_argument("--record", action="store_true",
                            help="Record input data to a pickle")
        record.add_argument("--record-file", default=None, metavar="FILE",
                            help="Recording output path (default: auto-generated)")


def open_input(args: argparse.Namespace, hand_side: str):
    """Create the input device selected by ``--input``.

    Backends are imported lazily: camera / video inputs need OpenCV and
    MediaPipe, which headset-only setups should not have to install.
    """
    if args.input == "avp":
        from .visionpro import VisionPro
        return VisionPro(ip=args.avp_ip)
    if args.input == "quest3":
        from .quest3 import Quest3
        return Quest3(port=args.quest3_port, protocol=args.quest3_protocol)
    if args.input == "pico4":
        from .pico4 import Pico4
        return Pico4(
            mode=args.pico4_mode,
            relay_host=args.pico4_relay_host,
            relay_port=args.pico4_relay_port,
            port=args.pico4_port,
            broadcast_port=args.pico4_broadcast_port,
        )
    if args.input == "noitom":
        from .noitom import NoitomInput
        return NoitomInput(
            local_ip=args.noitom_local_ip,
            local_port=args.noitom_local_port,
            server_ip=args.noitom_server_ip,
            server_port=args.noitom_server_port,
        )
    if args.input == "camera":
        from .camera import Camera
        return Camera(camera_id=0, show_preview=True)
    if args.input == "realsense":
        from .realsense import Realsense
        return Realsense(hand_side=hand_side, show_video=args.show_video)
    if args.input == "video":
        from .video import Video
        return Video(
            video_path=input_file(args),
            hand_side=hand_side,
            show_video=args.show_video,
            playback_speed=args.speed,
            loop=not args.no_loop,
        )
    if args.input == "replay":
        from .mediapipe_replay import MediaPipeReplay
        return MediaPipeReplay(
            record_path=input_file(args),
            playback_speed=args.speed,
            loop=not args.no_loop,
        )
    raise ValueError(f"Unknown input: {args.input}")
