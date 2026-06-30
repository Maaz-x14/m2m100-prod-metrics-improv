#!/usr/bin/env bash
# =============================================================================
# setup.sh — Bootstrap a DigitalOcean GPU Droplet for M2M100 inference
# Run as root (or with sudo) immediately after first SSH login.
# =============================================================================
set -euo pipefail

### ── 0. Variables ────────────────────────────────────────────────────────────
APP_USER="m2m"
APP_DIR="/opt/m2m100"
PYTHON_VERSION="3.11"
HF_MODEL="Mavkif/m2m100_rup_ur_to_rur"
HF_TOKENIZER="Mavkif/m2m100_rup_tokenizer_both"
API_PORT=8000

echo "━━━ [1/8] System packages ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
apt-get update -qq
apt-get install -y -qq \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
    git curl wget htop nvtop \
    ufw

### ── 1. Firewall ─────────────────────────────────────────────────────────────
echo "━━━ [2/8] Firewall ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow ${API_PORT}/tcp comment "M2M100 API"
ufw --force enable
ufw status verbose

### ── 2. App user ─────────────────────────────────────────────────────────────
echo "━━━ [3/8] App user ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
id -u ${APP_USER} &>/dev/null || useradd -m -s /bin/bash ${APP_USER}
mkdir -p ${APP_DIR}
cp -r "$(dirname "$0")"/../server/* ${APP_DIR}/
chown -R ${APP_USER}:${APP_USER} ${APP_DIR}

### ── 3. Python venv + deps ───────────────────────────────────────────────────
echo "━━━ [4/8] Python environment ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo -u ${APP_USER} python${PYTHON_VERSION} -m venv ${APP_DIR}/venv
VENV="${APP_DIR}/venv/bin"

# PyTorch with CUDA 12.1 wheels
sudo -u ${APP_USER} ${VENV}/pip install --quiet --upgrade pip
sudo -u ${APP_USER} ${VENV}/pip install --quiet \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Rest of requirements
sudo -u ${APP_USER} ${VENV}/pip install --quiet \
    -r ${APP_DIR}/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

### ── 4. Pre-download the model ───────────────────────────────────────────────
echo "━━━ [5/8] Downloading model weights ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Downloads to HF cache (~/.cache/huggingface) as app user
sudo -u ${APP_USER} ${VENV}/python - <<'PYEOF'
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import os

model_id = os.environ.get("HF_MODEL", "")
tokenizer_id = os.environ.get("HF_TOKENIZER", "Mavkif/m2m100_rup_tokenizer_both")
if not model_id:
    raise SystemExit("Set HF_MODEL to the Urdu→Roman Urdu checkpoint before running setup.sh")
print(f"  Fetching tokenizer ({tokenizer_id}) …")
M2M100Tokenizer.from_pretrained(tokenizer_id)
print(f"  Fetching model weights ({model_id}) …")
M2M100ForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True)
print("  Done.")
PYEOF

### ── 5. Systemd service ──────────────────────────────────────────────────────
echo "━━━ [6/8] Systemd service ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat > /etc/systemd/system/m2m100.service <<EOF
[Unit]
Description=M2M100 Urdu → Roman Urdu Transliteration API
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${VENV}/uvicorn app:app \
    --host 0.0.0.0 \
    --port ${API_PORT} \
    --workers 1 \
    --loop uvloop \
    --log-level info
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# GPU environment
Environment=CUDA_VISIBLE_DEVICES=0
Environment=TOKENIZERS_PARALLELISM=false
Environment=OMP_NUM_THREADS=4

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable m2m100
systemctl start m2m100

### ── 6. Verify ───────────────────────────────────────────────────────────────
echo "━━━ [7/8] Waiting for service to start …  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sleep 20   # model load takes ~15 s on first start
curl -sf http://localhost:${API_PORT}/health | python3 -m json.tool || {
    echo "Health check failed — check: journalctl -u m2m100 -n 50"
    exit 1
}

### ── 7. Quick smoke-test ─────────────────────────────────────────────────────
echo "━━━ [8/8] Smoke test ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST http://localhost:${API_PORT}/translate \
    -H "Content-Type: application/json" \
    -d '{"text": "مجھے کھانا چاہیے", "num_beams": 4}' \
    | python3 -m json.tool

echo ""
echo "━━━ Setup complete! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  API endpoint : http://<DROPLET_IP>:${API_PORT}"
echo "  Health       : http://<DROPLET_IP>:${API_PORT}/health"
echo "  Logs         : journalctl -u m2m100 -f"
