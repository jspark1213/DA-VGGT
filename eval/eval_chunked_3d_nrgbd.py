"""DA-VGGT nrgbd reconstruction using the shared eval_chunked protocol.

See README.md for data layouts, commands and protocol provenance.
"""
import sys
from pathlib import Path

# Direct execution from eval/ must resolve this repository's vggt package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.recon_eval import main


if __name__ == "__main__":
    raise SystemExit(main("nrgbd"))
