"""cctv_meta.py — CCTV 관제 메타 레이어(카메라·날짜·열화상 온도·고온 알림).

[왜 별도 모듈]
  현재 색인 데이터(ChromaDB)에는 video_id·구간·VLM 캡션·severity 만 있고
  '카메라 번호 / 촬영 일자 / 열화상 온도' 는 없다. 실제 운영에선 이 값들이
  - camera_id/date : CCTV 메타(파일명 규칙 or DMS) 에서,
  - 온도/열화상   : 열화상 카메라 피드/로그 에서
  들어온다. 아직 그 소스가 없으므로 여기서 **결정적으로 파생**하되,
  `outputs/vmem/cctv_map.json` 이 있으면 그 값으로 **덮어쓴다(=실데이터 교체 지점)**.

[교체 방법] 실데이터가 생기면:
  1) cctv_map.json 에 {video_id: {camera_id, camera_name, date, thermal_id}} 채우거나
  2) _synth_temp() 를 실제 열화상 온도 조회로 바꾸면 됨. 프론트/엔드포인트는 그대로.
"""
import config

import hashlib
import json
import os
from datetime import datetime, timedelta

# 데모용 카메라 풀 — 실데이터 오면 cctv_map.json 이 우선.
_CAMERA_POOL = ["CAM01", "CAM02", "CAM03", "CAM04", "CAM05", "CAM06", "CAM07", "CAM08"]
_DEFAULT_DATE = "2026-06-08"

# 온도 임계(℃) — 열화상 고온 알림 기준. 실데이터 연동 시 운영값으로 조정.
TEMP_WARN = 50.0    # 주의
TEMP_CRIT = 65.0    # 위험


def _load_overrides():
    p = os.path.join(config.MEMORY_DIR, "cctv_map.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _recorded_at(video_id):
    """cctv_map.json 의 recorded_at(녹화 시작 wall-clock) → datetime, 없으면 None.
    실 CCTV 전환 시 이 값은 파일명(strftime) 또는 NVR 메타에서 채운다."""
    r = _load_overrides().get(video_id, {}).get("recorded_at")
    if r:
        try:
            return datetime.fromisoformat(r)
        except ValueError:
            return None
    return None


def abs_clock(video_id, start_s):
    """녹화 시작시각 + 구간 오프셋 = 사건의 절대 시각(datetime). recorded_at 없으면 None."""
    base = _recorded_at(video_id)
    return base + timedelta(seconds=float(start_s or 0)) if base else None


def video_meta(video_id, indexed_at=None):
    """video_id → {camera_id, camera_name, date}. cctv_map.json(camera_id·camera_name·recorded_at) 우선.
    date 는 녹화시각(recorded_at) → cctv_map date → 색인시각 → 기본값 순으로 결정."""
    ov = _load_overrides().get(video_id, {})
    h = int(hashlib.md5(video_id.encode("utf-8")).hexdigest(), 16)
    cam = ov.get("camera_id") or _CAMERA_POOL[h % len(_CAMERA_POOL)]
    rec = _recorded_at(video_id)
    date = (rec.date().isoformat() if rec
            else ov.get("date") or (indexed_at[:10] if indexed_at and len(indexed_at) >= 10 else _DEFAULT_DATE))
    return {"camera_id": cam, "camera_name": ov.get("camera_name") or cam, "date": date}


def _synth_temp(seg):
    """구간 → 열화상 온도(℃) 파생(데모, 디스플레이용). event_type/severity 가 높을수록 고온.

    실데이터 연동 시 이 함수만 '해당 시각 열화상 최고온도 조회' 로 교체하면 된다.
    (참고: 화재 감지는 별도 열화상 카메라가 담당 — 여기 온도는 콘솔 표시용 합성값.)
    """
    base = {"flammable": 70.0, "smoking": 58.0, "fall": 46.0, "vehicle_interaction": 42.0,
            "loitering": 40.0, "normal": 36.0, "unknown": 36.0}
    sev = int(seg.get("severity") or 0)
    t = base.get(seg.get("event_type") or "normal", 36.0) + sev * 6.5
    t += (int(seg.get("start_s") or 0) % 7) * 0.5            # 구간별 미세 변동(데모 현실감)
    t += min(int(seg.get("person_count") or 0), 3) * 0.4
    return round(t, 1)


def temp_level(temp):
    return "위험" if temp >= TEMP_CRIT else "주의" if temp >= TEMP_WARN else "정상"


# ── [열화상 전환 지점] ────────────────────────────────────────────────────────
#   지금 온도는 _synth_temp() 합성값(플레이스홀더)이다. 나중에 실제 열화상으로 전환할 때:
#     1) outputs/vmem/thermal/{video_id}.json = {"<구간시작초>": 온도℃, ...} 를 넣으면
#        seg_temp() 가 합성값 대신 그 값을 자동 사용한다(코드 수정 불필요).
#     2) outputs/vmem/videos/{video_id}__thermal.mp4 를 넣으면 PIP 가 실제 열화상을 재생한다.
THERMAL_DIR = os.path.join(config.MEMORY_DIR, "thermal")
_THERMAL_CACHE = {}


def _real_temp(video_id, start_s):
    """실제 열화상 온도(있으면) — thermal/{video_id}.json 에서 조회, 없으면 None."""
    if not video_id:
        return None
    if video_id not in _THERMAL_CACHE:
        try:
            with open(os.path.join(THERMAL_DIR, f"{video_id}.json"), encoding="utf-8") as f:
                _THERMAL_CACHE[video_id] = json.load(f)
        except Exception:
            _THERMAL_CACHE[video_id] = None
    table = _THERMAL_CACHE[video_id]
    return table.get(str(int(start_s))) if table and start_s is not None else None


def seg_temp(seg):
    """구간 → (온도℃, 등급). 실제 열화상 온도가 있으면 그 값, 없으면 합성값(플레이스홀더)."""
    t = _real_temp(seg.get("video_id"), seg.get("start_s"))
    if t is None:
        t = _synth_temp(seg)
    return round(float(t), 1), temp_level(t)


def _fmt_ts(s):
    s = int(s or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def enrich_videos(videos):
    """list_videos() 결과에 camera_id/camera_name/date 부착."""
    out = []
    for v in videos:
        m = video_meta(v["video_id"], v.get("indexed_at"))
        out.append({**v, **m})
    return out


def list_cameras(videos):
    """색인된 영상 → 카메라 목록(+영상 수·구간 수) & 가용 날짜."""
    cams, dates = {}, set()
    for v in videos:
        m = video_meta(v["video_id"], v.get("indexed_at"))
        dates.add(m["date"])
        c = cams.setdefault(m["camera_id"], {
            "camera_id": m["camera_id"], "camera_name": m["camera_name"],
            "videos": [], "segments": 0, "date": m["date"]})
        c["videos"].append(v["video_id"])
        c["segments"] += v.get("segments", 0)
    return sorted(cams.values(), key=lambda c: c["camera_id"]), sorted(dates, reverse=True)


def build_alerts(all_segments, date=None, cams=None):
    """전체 사건 → 특이사건 알림(severity ≥ 1). date·cams(set) 로 필터, 심각도·온도 높은 순.

    화재→행동 전환에 따라 알림 기준은 '합성 고온'이 아니라 '사건 심각도'다(낙상·배회·흡연·인화물 등).
    온도(temp)는 디스플레이용으로 함께 싣는다.
    반환: [{id, camera_id, camera_name, video_id, date, ts, seg_start, seg_end,
            temp, level, severity, event_type, caption, thumb, person_count, dwell_s}]
    """
    alerts = []
    for seg in all_segments:
        if int(seg.get("severity") or 0) < 1:
            continue                                   # 통상 활동(severity 0)은 알림 아님
        temp, level = seg_temp(seg)
        m = video_meta(seg["video_id"], seg.get("indexed_at"))
        if date and m["date"] != date:
            continue
        if cams and m["camera_id"] not in cams:
            continue
        start = seg["start_s"]
        ac = abs_clock(seg["video_id"], start)
        alerts.append({
            "id": f"{seg['video_id']}:{int(start)}",
            "camera_id": m["camera_id"], "camera_name": m["camera_name"],
            "video_id": seg["video_id"], "date": m["date"],
            "ts": ac.strftime("%H:%M:%S") if ac else _fmt_ts(start), "seg_start": start, "seg_end": seg["end_s"],
            "temp": temp, "level": level, "severity": int(seg.get("severity") or 0),
            "event_type": seg.get("event_type") or "normal", "caption": seg.get("caption") or "",
            "thumb": seg.get("thumb"), "person_count": seg.get("person_count") or 0,
            "dwell_s": seg.get("dwell_s") or 0,
        })
    alerts.sort(key=lambda a: (a["severity"], a["temp"]), reverse=True)
    return alerts


def build_history(all_segments, date=None, cams=None):
    """전체 구간 이력(시간순) — 평소 보는 단일 타임라인. 고온 구간은 is_alert=True 로 마킹.

    date·cams(set) 로 필터. 다중 카메라면 (카메라, 시간) 순으로 병합 정렬.
    반환 행: {camera_id, video_id, date, ts, start_s, end_s, caption, risk_type,
             severity, temp, level, is_alert, thumb}
    """
    rows = []
    for seg in all_segments:
        m = video_meta(seg["video_id"], seg.get("indexed_at"))
        if date and m["date"] != date:
            continue
        if cams and m["camera_id"] not in cams:
            continue
        temp, level = seg_temp(seg)
        sev = int(seg.get("severity") or 0)
        ac = abs_clock(seg["video_id"], seg["start_s"])
        rows.append({
            "camera_id": m["camera_id"], "video_id": seg["video_id"], "date": m["date"],
            "ts": ac.strftime("%H:%M:%S") if ac else _fmt_ts(seg["start_s"]), "start_s": seg["start_s"], "end_s": seg["end_s"],
            "caption": seg.get("caption") or "", "event_type": seg.get("event_type") or "normal",
            "severity": sev, "temp": temp, "level": level,
            "is_alert": sev >= 1, "thumb": seg.get("thumb"),
            "dwell_s": seg.get("dwell_s") or 0, "person_count": seg.get("person_count") or 0,
        })
    rows.sort(key=lambda r: (r["camera_id"], r["start_s"]))
    return rows
