"""memory/event_builder.py — 구간 → '사건(Event)' 병합.

[왜] 5초 구간 단위로 저장하면 "어떤 사람이 02:14 들어와 5분간 서성이다 나갔다"가
     수십 개 레코드로 흩어진다. '이력'이 되려면 연속 구간을 하나의 사건(시작~종료+요약)으로
     묶어야 한다. 검색·타임라인의 단위 = 1 사건 = 1 레코드.

[입력] segmenter 구간 + 구간별 VLM 결과(parse_event) + tracker 추적 타임라인
[규칙]
  - 활동(activity) 있는 구간만 사건에 포함(정적 배경은 제외 — 관제 기준).
  - 인접(공백 ≤ EVENT_MERGE_GAP_S) 활동 구간들을 한 사건으로 병합.
  - 사건 유형 = 구성 구간 중 가장 심각/특이한 유형(severity 최댓값, normal 보다 특이 우선).
  - 배회(loitering)는 VLM 이 아니라 tracker dwell_s/이동범위로 판정해 normal 사건을 승격.
"""
import config  # ★ torch 보다 먼저
from dataclasses import dataclass, field
from memory.tracker import tracks_in_window


@dataclass
class Event:
    event_id: str
    video_id: str
    event_type: str
    start_s: float
    end_s: float
    severity: int
    summary: str                       # 대표 캡션(검색 문서)
    person_count: int = 0
    track_ids: list = field(default_factory=list)
    has_vehicle: bool = False
    objects: list = field(default_factory=list)
    dwell_s: float = 0.0               # 사건 내 최장 사람 체류시간
    seg_indices: list = field(default_factory=list)
    rep_seg: int = 0                   # 대표(썸네일) 구간 index


def _is_special(et):
    return et not in ("normal", "unknown")


def _seg_score(rec):
    """대표 구간 선택용 점수 — severity 우선, 동률이면 특이 유형 우선."""
    return (rec["lab"]["severity"], _is_special(rec["lab"]["event_type"]))


def build_events(video_id, segments, seg_results, tracks):
    """segments[i] + seg_results[i]=(caption,label) + tracks(dict) → [Event…].

    seg_results 의 label = {activity, event_type, objects, severity}.
    """
    gap = config.EVENT_MERGE_GAP_S

    # ── 1) 활동 구간만 모아 인접 병합 (같은 유형 버킷끼리만) ──
    #    통상활동(ambient)은 길게 병합하되, 특이 이벤트(fall/smoking/…)는 그 자체로 분리되어
    #    타임라인에서 정확한 시각으로 잡히게 한다(과병합 방지).
    groups, cur = [], None
    for i, (seg, (cap, lab)) in enumerate(zip(segments, seg_results)):
        if not lab.get("activity"):
            cur = None                                   # 정적 구간에서 사건 끊김
            continue
        bucket = "ambient" if lab["event_type"] in ("normal", "unknown") else lab["event_type"]
        rec = {"i": i, "start": seg.start_s, "end": seg.end_s, "cap": cap, "lab": lab}
        if cur is not None and cur["bucket"] == bucket and (seg.start_s - cur["end"]) <= gap:
            cur["items"].append(rec)
            cur["end"] = seg.end_s
        else:
            cur = {"start": seg.start_s, "end": seg.end_s, "items": [rec], "bucket": bucket}
            groups.append(cur)

    # ── 2) 그룹 → Event(유형·심각도·요약 + tracker 보강) ──
    events = []
    for gi, g in enumerate(groups):
        items = g["items"]
        dom = max(items, key=_seg_score)                 # 가장 특이/심각한 구간 = 대표
        event_type = dom["lab"]["event_type"]
        severity = max(r["lab"]["severity"] for r in items)
        objects = sorted({o for r in items for o in r["lab"].get("objects", [])})
        summary = dom["cap"]

        persons, vehicles = _active_actors(tracks, g["start"], g["end"])
        dwell_s = max((t.dwell_s for t in persons), default=0.0)

        # 배회 승격 — 정적/통상 활동인데 오래 머문 사람이 있으면 loitering
        if event_type in ("normal", "unknown") and _has_loiterer(persons):
            event_type = "loitering"
            severity = max(severity, 1)
            summary = f"사람이 약 {int(dwell_s)}초간 머무름(배회 의심)"

        events.append(Event(
            event_id=f"{video_id}:e{gi}", video_id=video_id, event_type=event_type,
            start_s=g["start"], end_s=g["end"], severity=severity, summary=summary,
            person_count=len(persons), track_ids=sorted(t.track_id for t in persons),
            has_vehicle=bool(vehicles), objects=objects, dwell_s=round(dwell_s, 2),
            seg_indices=[r["i"] for r in items], rep_seg=dom["i"],
        ))

    # ── 3) 어떤 사건에도 안 잡힌 배회자(track) 보강 ──
    covered = [(e.start_s, e.end_s) for e in events]
    extra = 0
    for t in tracks.values():
        if not (t.is_person and _loiters(t)):
            continue
        if any(s <= t.last_seen_s and t.first_seen_s <= e for s, e in covered):
            continue                                     # 이미 사건이 덮음
        _, veh = _active_actors(tracks, t.first_seen_s, t.last_seen_s)
        events.append(Event(
            event_id=f"{video_id}:L{t.track_id}", video_id=video_id, event_type="loitering",
            start_s=t.first_seen_s, end_s=t.last_seen_s, severity=1,
            summary=f"사람이 약 {int(t.dwell_s)}초간 머무름(배회 의심)",
            person_count=1, track_ids=[t.track_id], has_vehicle=bool(veh),
            objects=[], dwell_s=round(t.dwell_s, 2), seg_indices=[], rep_seg=-1,
        ))
        extra += 1

    events.sort(key=lambda e: e.start_s)
    print(f"[event_builder] 구간 {len(segments)} → 사건 {len(events)} "
          f"(병합 그룹 {len(groups)} + 배회 보강 {extra})", flush=True)
    return events


def _active_actors(tracks, start_s, end_s):
    """[start_s,end_s] 구간에 활동한 (사람 track 목록, 차량 track 목록)."""
    active = tracks_in_window(tracks, start_s, end_s)
    persons = [t for t in active if t.is_person]
    vehicles = [t for t in active if not t.is_person]
    return persons, vehicles


def _loiters(track):
    return (track.dwell_s >= config.LOITER_DWELL_S
            and track.displacement() <= config.LOITER_MAX_MOVE_PX)


def _has_loiterer(persons):
    return any(_loiters(t) for t in persons)
