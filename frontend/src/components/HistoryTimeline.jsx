import { useEffect, useState } from 'react'
import { listHistory, levelClass, eventLabel } from '../api'

// 선택한 카메라의 구간 이력 로그(CCTV 서비스풍, 차분). 고온 구간은 좌측 얇은 색선 + 온도색으로만 구분.
// CAM 번호는 행마다가 아니라 헤더 오른쪽에 한 번. 시각순.
export default function HistoryTimeline({ date, cams, cam, activeKey, onSeek }) {
  const [segs, setSegs] = useState([])
  useEffect(() => {
    listHistory(date, cams).then(j => setSegs(j.segments || []))
  }, [date, (cams || []).join(',')])

  return (
    <div className="card-box">
      <div className="log-head">
        <div className="log-title">특이사항 <span className="log-count">{segs.length}건</span></div>
      </div>
      <div className="loglist">
        {segs.map(s => {
          const key = `${s.video_id}:${Math.floor(s.start_s)}`
          const lc = levelClass(s.level)
          return (
            <div key={key} className={`logrow ${s.is_alert ? 'alert ' + lc : ''} ${key === activeKey ? 'on' : ''}`}
                 onClick={() => onSeek(s.video_id, s.start_s, s)}>
              <span className="logts">{s.ts}</span>
              <span className="logcap">{s.caption}</span>
              <span className="logtag">{s.event_type && s.event_type !== 'normal'
                ? <span className={`tag ${s.event_type}`}>{eventLabel(s.event_type)}</span> : null}</span>
              <span className={`logtemp ${lc}`}>{s.is_alert ? `${s.temp.toFixed(1)}℃` : ''}</span>
            </div>
          )
        })}
        {!segs.length && <div className="hint">이 일자에 색인된 구간이 없습니다</div>}
      </div>
    </div>
  )
}
