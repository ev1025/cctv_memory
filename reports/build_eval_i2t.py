"""reports/build_eval_i2t.py — eval_parts/*.json → 통합 HTML.

input 프롬프트 + 제공 이미지(34장) + 이미지별 4종 모델 답(위험/유형 ✅❌) + SCORE 요약.
eval_i2t 가 모델별 json 을 남기면, 이 스크립트가 4종을 한 표로 합친다.

실행:  python -m reports.build_eval_i2t   (이미지는 DATASET_DIR 에서 로드)
"""
import config

import os
import json
from PIL import Image

import report_utils as R

DATASET = os.environ.get(
    "DATASET_DIR", r"C:\Users\eg287\OneDrive\바탕 화면\project\전시회\vlm\dataset\images")
ORDER = ["qwen2-vl", "qwen2.5-vl", "qwen3-vl", "internvl3"]


def _thumb_b64(p):
    img = Image.open(p).convert("RGB")
    img.thumbnail((220, 220))
    return R.b64(img)


def main():
    parts = os.path.join(config.REPORT_DIR, "eval_parts")
    meta = json.load(open(os.path.join(parts, "_meta.json"), encoding="utf-8"))
    items = meta["items"]   # [[fn, cat, risk], ...]
    models = [m for m in ORDER if os.path.exists(os.path.join(parts, f"{m}.json"))]
    res = {}
    for m in models:
        rows = json.load(open(os.path.join(parts, f"{m}.json"), encoding="utf-8"))["rows"]
        res[m] = {r[0]: r for r in rows}   # fn → [fn, cat, risk, ra, rhit, ta, thit]

    n = len(items)
    nrisk = sum(1 for _, _, r in items if r)
    nnorm = n - nrisk

    # ── SCORE 요약 ──
    summ = ""
    for m in models:
        rows = list(res[m].values())
        recall = sum(1 for r in rows if r[2] and r[4]) / max(nrisk, 1)
        halluc = sum(1 for r in rows if (not r[2]) and (not r[4])) / max(nnorm, 1)
        typ = sum(r[6] for r in rows) / max(len(rows), 1)
        summ += (f'<tr><th>{R.esc(m)}</th><td class="ok"><b>{recall*100:.0f}%</b></td>'
                 f'<td>{halluc*100:.0f}%</td><td>{typ*100:.0f}%</td></tr>')
    summary = (f'<table><thead><tr><th>VLM (4bit)</th><th>위험 Recall ↑</th>'
               f'<th>환각률 ↓</th><th>유형 정확도</th></tr></thead><tbody>{summ}</tbody></table>')

    # ── 이미지별 4종 답 표 ──
    mh = "".join(f"<th>{R.esc(m)}</th>" for m in models)
    body_rows = ""
    for fn, cat, risk in items:
        p = os.path.join(DATASET, fn)
        cells = ""
        for m in models:
            r = res[m].get(fn)
            if r:
                rm = "✅" if r[4] else "❌"
                tm = "✅" if r[6] else "❌"
                cells += (f'<td><b>{rm}</b> 위험:"{R.esc(r[3][:16])}"<br>'
                          f'<b>{tm}</b> 유형:"{R.esc(r[5][:12])}"</td>')
            else:
                cells += "<td>-</td>"
        thumb = f'<img src="{_thumb_b64(p)}" style="height:70px;border-radius:4px">' if os.path.exists(p) else ""
        body_rows += (f'<tr><th>{thumb}<br>{R.esc(fn)}<br>'
                      f'<small>정답: {cat}/{"위험" if risk else "정상"}</small></th>{cells}</tr>')
    detail = f'<table><thead><tr><th>이미지 (정답)</th>{mh}</tr></thead><tbody>{body_rows}</tbody></table>'

    body = (
        f'<div class="orow"><b>INPUT 프롬프트 ① 위험 유무</b><code>{R.esc(meta["risk_q"])}</code></div>'
        f'<div class="orow"><b>INPUT 프롬프트 ② 유형 분류</b><code>{R.esc(meta["type_q"])}</code></div>'
        f'<div class="orow"><b>제공 이미지</b>: 위험 {nrisk}장(fire·smoke·fall·machine) + 정상 {nnorm}장 = {n}장</div>'
        f'<div class="orow sub">위험 Recall=위험을 위험으로 인지(↑·놓치면 치명), 환각률=정상을 위험으로 오판(↓), 유형=fire/smoke/fall/machine/normal 분류</div>'
        f'<div class="io out"><span class="lbl">SCORE</span>모델별 핵심 지표</div>{summary}'
        f'<div class="io out"><span class="lbl">상세</span>이미지별 4종 답 (위험/유형 ✅❌)</div>{detail}'
    )
    out = os.path.join(config.REPORT_DIR, "eval_i2t.html")
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(R.page("i2t 분류 평가", "Image→Text VLM 4종 분류 평가 (이미지별 답 + SCORE)", body))
    print(f"REPORT: {out}\nBUILD_DONE")


if __name__ == "__main__":
    main()
