"""DA-VGGT ScanNet-50 reconstruction using the shared eval_chunked protocol.

The default input scenes are listed in eval/scannet50_scenes.txt.
Use --scene_list for a custom list or --scenes for individual scenes.
See README.md for data layouts and commands.
"""
import sys
from pathlib import Path

# Direct execution from eval/ must resolve this repository's vggt package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.recon_eval import main


if __name__ == "__main__":
    raise SystemExit(main("scannet"))
