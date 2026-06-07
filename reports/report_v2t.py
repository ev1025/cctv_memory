"""report_v2t.py — ③ video-to-text 리포트: 영상 → YOLO(person+car) → VLM 시계열·이상행동 JSON."""
import config

import os
import cv2
from PIL import Image

import report_utils as R
from image_to_text import VLMCaptioner

VIDEO = os.environ.get("TEST_VIDEO", "person-bicycle-car-detection.mp4")
BACKEND = os.environ.get("SMOKE_VLM", "qwen2.5-vl")
MAX_TRIGGERS = int(os.environ.get("MAX_TRIGGERS", "3"))


def collect(vlm, yolo):
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps)))
    idx, buf, trig = 0, [], []
    while len(trig) < MAX_TRIGGERS:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            res = yolo(frame, conf=0.4, classes=[0, 2], verbose=False)[0]   # person, car
            cls = [int(c) for c in res.boxes.cls] if len(res.boxes) else []
            if 0 in cls:                                  # 사람이 있을 때만 트리거(이상행동은 사람)
                buf.append((idx, cls.count(0), res.plot()))
                if len(buf) >= 3:
                    imgs = [Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)) for _, _, a in buf]
                    out = vlm.caption_frames(imgs, config.VLM_ANOMALY_PROMPT)
                    trig.append({"frames": [i for i, _, _ in buf], "persons": max(n for _, n, _ in buf),
                                 "imgs": [R.b64(im) for im in imgs], "out": out})
                    buf = []
        idx += 1
    cap.release()
    return trig


def main():
    from ultralytics import YOLO
    yolo = YOLO("yolov8n.pt")
    vlm = VLMCaptioner(BACKEND).load()
    trig = collect(vlm, yolo)
    vlm.unload()

    vid = ""
    for i, t in enumerate(trig, 1):
        box = "".join(f'<img src="{u}">' for u in t["imgs"])
        aj = R.extract_json(t["out"])
        reasoning = aj.get("reasoning", "") if aj else ""
        badge = ""
        if aj:
            risk = aj.get("risk_level", "?")
            color = {"high": "#e05260", "unknown": "#d29922"}.get(str(risk).lower(), "#3fb950")
            badge = f'<span class="badge" style="background:{color}">{R.esc(risk)} / {R.esc(aj.get("type"))}</span>'
        vid += (f'<div class="sub2">트리거 {i} · 프레임 {t["frames"]} (사람 {t["persons"]}명)</div>'
                f'<div class="frames">{box}</div>'
                f'<div class="orow"><b>VLM 추론(CoT)</b>: {R.esc(reasoning)}</div>'
                f'<div class="orow"><b>판정</b> {badge}<code>{R.esc(t["out"])}</code></div>')
    if not trig:
        vid = '<div class="orow">사람이 감지된 트리거 구간이 없습니다(영상/임계값 확인).</div>'

    body = (
        f'<div class="io in"><span class="lbl">INPUT</span>영상 {R.esc(VIDEO)}</div>'
        f'<div class="orow sub"><b>1차 YOLO</b>: YOLOv8n(yolov8n.pt) · conf=0.4 · classes=[0,2]=person,car · COCO 사전학습</div>'
        f'<div class="orow sub"><b>샘플링</b>: 사람 감지 프레임만 1fps · 트리거당 3장 → VLM 1-call</div>'
        f'<div class="orow"><b>2차 VLM 프롬프트(입력 · XML · Visual CoT · ASK-HINT detection_cues)</b>'
        f'<code>{R.esc(config.VLM_ANOMALY_PROMPT)}</code></div>'
        f'<div class="io out"><span class="lbl">OUTPUT</span>YOLO 박스 + VLM 추론(reasoning) · 이상행동 JSON</div>'
        f'{vid}'
    )
    out = os.path.join(config.REPORT_DIR, "report_v2t.html")
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(R.page("video-to-text", f"③ video-to-text — 영상 → 텍스트 (YOLO → VLM {R.esc(BACKEND)})", body))
    print(f"REPORT: {out}\nV2T_DONE")


if __name__ == "__main__":
    main()
