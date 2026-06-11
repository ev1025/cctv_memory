"""download_all.py — 로컬 운영용: VLM + 임베딩 모델만 받는다 (T2I 는 3d_vision 레포).
서버 비교에서 정확했던 VLM 계열 + video-memory 임베딩(bge-m3)만 선택.
GPU 불필요(다운로드만). gated/네트워크 실패는 graceful 하게 FAIL 로 기록하고 계속 진행."""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import time
import models
from huggingface_hub import snapshot_download

VLM_KEYS = ["qwen2-vl", "qwen2.5-vl", "qwen3-vl", "internvl3", "pixtral"]   # 비교했던 계열
EMBED_KEYS = ["bge-m3"]                                                      # video-memory 임베딩
ids = ([models.VLM_REGISTRY[k]["id"] for k in VLM_KEYS]
       + [models.EMBED_REGISTRY[k]["id"] for k in EMBED_KEYS])


def main():
    print(f"받을 모델 {len(ids)}개\n")
    ok, fail = [], []
    for i, mid in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {mid} ...", flush=True)
        t = time.time()
        try:
            snapshot_download(mid)
            print(f"   OK ({time.time()-t:.0f}s)", flush=True)
            ok.append(mid)
        except Exception as e:
            print(f"   FAIL: {str(e)[:120]}", flush=True)
            fail.append(mid)
    print(f"\n=== 완료: 성공 {len(ok)} / 실패 {len(fail)} ===")
    for m in fail:
        print(f"  FAIL {m}")
    print("DOWNLOAD_ALL_DONE")


if __name__ == "__main__":
    main()
