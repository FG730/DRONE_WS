#!/usr/bin/env python3

import runpy
from pathlib import Path


if __name__ == "__main__":
    script = Path.home() / "drone_ws" / "scripts" / "move_target_realistic.py"
    runpy.run_path(str(script), run_name="__main__")
