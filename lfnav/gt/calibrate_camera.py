"""Fit the camera field of view. The Minecraft default is not assumed.

Place one high-contrast block at a known yaw or pitch relative to the agent, read the
pixel of that block in obs['rgb'], then call fov_deg_from_pixel. This script does the
placement only when --make-json is passed and minedojo imports. The fit itself is in
lfnav/gt/fov.py and is covered by unit tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfnav.gt.fov import fov_deg_from_pixel
from lfnav.io import atomic_write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel", type=float, required=True, help="pixel coordinate of the known block")
    parser.add_argument("--size", type=int, required=True, help="image width if the angle is yaw, height if it is pitch")
    parser.add_argument("--angle-deg", type=float, required=True, help="signed angle of that block from the optical axis")
    parser.add_argument("--axis", choices=("horizontal", "vertical"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fov = fov_deg_from_pixel(args.pixel, args.size, args.angle_deg)
    atomic_write_json(args.out, {"fov_deg": fov, "fov_axis": args.axis, "pixel": args.pixel, "size": args.size, "angle_deg": args.angle_deg})
    print(json.dumps({"fov_deg": fov, "fov_axis": args.axis}))


if __name__ == "__main__":
    main()
