import { useEffect, useState } from 'react'
import { listHistory, levelClass, eventLabel } from '../api'

// 전체 CCTV에서 발생하는 상황을 실시간 표기하는 통합 이력(그리드 우측). 행마다 CAM 라벨.
// 이상(고온) 상황을 위로, 클릭 시 해당 카메라 포커스 + 그 시각으로 이동.
export default function SituationLog({ date, onSelect }) {
  const [segs, setSegs] = useState([])

  useEffect(() => {
    const load = () => listHistory(date, null).then(j => setSegs(j.segments || []))
    load()
    const id = setInterval(load, 8000)   // 실시간 갱신(폴링) — 새 상황이 색인되면 반영
    return () => clearInterval(id)
  }, [date])

  // 영상 시간(구간 시작초) 내림차순 — 최신 시각이 위로(실시간 로그). 동시각은 카메라순.
  const rows = [...segs].sort((a, b) =>
    (b.start_s - a.start_s) || a.camera_id.localeCompare(b.camera_id))

  return (
    <div className="card-box situ">
      <div className="log-head">
        <div className="log-title">실시간 상황 <span className="log-count">{segs.length}건</span></div>
      </div>
      <div className="loglist situ-list">
        {rows.map(s => {
          const key = `${s.video_id}:${Math.floor(s.start_s)}`
          const lc = levelClass(s.level)
          return (
            <div key={key} className={`logrow ${s.is_alert ? 'alert ' + lc : ''}`}
                 onClick={() => onSelect && onSelect(s)} title={`${s.camera_id} ${s.ts} 보기`}>
              <span className="logcam">{s.camera_id}</span>
              <span className="logts">{s.ts}</span>
              <span className="logcap">{s.caption}</span>
              <span className="logtag">{s.event_type && s.event_type !== 'normal'
                ? <span className={`tag ${s.event_type}`}>{eventLabel(s.event_type)}</span> : null}</span>
              <span className={`logtemp ${lc}`}>{s.is_alert ? `${s.temp.toFixed(1)}℃` : ''}</span>
            </div>
          )
        })}
        {!segs.length && <div className="hint">표시할 상황이 없습니다</div>}
      </div>
    </div>
  )
}
