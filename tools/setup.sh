#!/bin/bash
# 서버 환경 셋업: venv + Blackwell(cu128) PyTorch + 의존성
# 사용: nohup bash setup.sh > setup.log 2>&1 &
set -e
cd "$(dirname "$0")"

echo "=== [1/4] venv 생성 ==="
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

echo "=== [2/4] PyTorch (CUDA 12.8 / Blackwell sm_120) 설치 ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/4] 나머지 의존성 설치 ==="
pip install -r requirements.txt

echo "=== [4/4] 설치 검증 (GPU 2,3번만 노출) ==="
CUDA_VISIBLE_DEVICES=2,3 python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("cuda build   :", torch.version.cuda)
print("cuda avail   :", torch.cuda.is_available())
print("device count :", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}")
    # sm_120 커널이 실제로 도는지 작은 연산으로 확인
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).sum().item()
    print("matmul OK (bf16):", y is not None)
PY
echo "===SETUP_DONE==="
