
---

# `run_demo.sh` (one-line to run everything)

```bash
#!/usr/bin/env bash
set -euo pipefail
python src/generate_synthetic.py --out data/raw/f1_telemetry.npy
python src/feature_engineer.py --in data/raw/f1_telemetry.npy --out data/processed/features.npz
python src/train_hybrid.py --features data/processed/features.npz --seed 42
python src/predict_alerts.py --features data/processed/features.npz --model models/hybrid_ensemble.npz
