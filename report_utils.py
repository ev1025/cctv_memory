"""report_utils.py — 리포트 공통 유틸(base64 이미지, escape, JSON 추출, CSS, 페이지 래퍼, 프레임 추출)."""
import io
import re
import json
import base64
import html

import cv2
import numpy as np
from PIL import Image


def b64(pil):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def esc(t):
    return html.escape(str(t))


def extract_json(t):
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def first_person_frame(video, yolo, classes=(0, 2)):
    """영상에서 사람(class 0)이 처음 감지되는 프레임을 PIL 로 반환(없으면 None)."""
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    idx, found = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % fps == 0:
            res = yolo(frame, conf=0.4, classes=list(classes), verbose=False)[0]
            cls = {int(c) for c in res.boxes.cls} if len(res.boxes) else set()
            if 0 in cls:
                found = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                break
        idx += 1
    cap.release()
    return found


CSS = """
body{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;max-width:1040px;margin:0 auto;padding:24px}
h1{font-size:21px}
.io{margin:14px 0 6px;font-weight:bold;border-top:1px dashed #30363d;padding-top:10px}
.io .lbl{display:inline-block;padding:3px 12px;border-radius:6px;color:#fff;margin-right:10px;font-size:13px}
.io.in .lbl{background:#1f6feb}.io.out .lbl{background:#3fb950}
.frames{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.frames img{height:170px;border-radius:6px;border:1px solid #30363d}
.orow{margin:8px 0;line-height:1.55}.sub{color:#8b949e;font-size:13px}.sub2{font-weight:bold;margin-top:14px;color:#58a6ff}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;color:#fff;font-weight:bold;font-size:12px}
code{display:block;background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px;margin-top:6px;color:#7ee787;white-space:pre-wrap;font-size:13px}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}
th,td{border:1px solid #30363d;padding:8px 10px;text-align:left;vertical-align:top}
thead th{background:#21262d}tbody th{background:#161b22;color:#58a6ff;white-space:nowrap}
"""


def page(title, h1, body):
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body>'
            f'<h1>{h1}</h1>{body}</body></html>')


# ── OpenCV 전처리 / 평가 ─────────────────────────────────────────────────────
def clahe(pil):
    """저조도 대비 향상(CLAHE, LAB 의 L 채널). 어두운 CCTV → 화재/객체 가시성↑. PIL→PIL."""
    lab = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB))


def fire_score(pil):
    """불꽃 색(빨강~주황 + 높은 채도·명도) 픽셀 비율 [0~1] (HSV). 화재 1차 객관 측정(채점 보조)."""
    hsv = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 150), (35, 255, 255))     # 빨강~주황
    m2 = cv2.inRange(hsv, (160, 90, 150), (180, 255, 255))  # 빨강 wrap-around
    mask = cv2.bitwise_or(m1, m2)
    return round(float((mask > 0).mean()), 4)


def hist_similarity(pil1, pil2):
    """두 이미지의 색 히스토그램 유사도 [-1~1] (HSV correlation). img2img 색 보존도 등."""
    def _h(p):
        hsv = cv2.cvtColor(np.array(p.convert("RGB")), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()
    return round(float(cv2.compareHist(_h(pil1), _h(pil2), cv2.HISTCMP_CORREL)), 4)


def ssim(pil1, pil2, size=(256, 256)):
    """구조 유사도 SSIM [0~1] (회색조). img2img 전후 구조 보존도. (간이 구현, OpenCV 만 사용)"""
    g1 = cv2.cvtColor(cv2.resize(np.array(pil1.convert("RGB")), size), cv2.COLOR_RGB2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(cv2.resize(np.array(pil2.convert("RGB")), size), cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = cv2.GaussianBlur(g1, (11, 11), 1.5), cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    s2 = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return round(float(ssim_map.mean()), 4)
