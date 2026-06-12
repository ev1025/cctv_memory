import { useEffect, useState, useRef } from 'react'
import { listHistory, levelClass, eventLabel } from '../api'

// 선택한 카메라의 구간 이력 로그(CCTV 서비스풍, 차분). 고온 구간은 좌측 얇은 색선 + 온도색으로만 구분.
// 현재 재생 시각(viewSec)에 해당하는 행을 강조 + 자동 스크롤 → 어떤 이력을 보는지 명확히.
export default function HistoryTimeline({ date, cams, cam, marked, maxSec = 86400, onSeek }) {
  const [segs, setSegs] = useState([])
  const activeRef = useRef(null)
  useEffect(() => {
    listHistory(date, cams).then(j => setSegs(j.segments || []))
  }, [date, (cams || []).join(',')])
  // 클릭한(강조) 행을 '목록 내부에서만' 스크롤(페이지는 안 움직임) → 어떤 이력을 눌렀는지 보이게
  useEffect(() => {
    const el = activeRef.current
    if (!el) return
    const list = el.closest('.loglist')
    if (!list) return
    const er = el.getBoundingClientRect(), lr = list.getBoundingClientRect()
    list.scrollTop += (er.top - lr.top) - (lr.height / 2 - er.height / 2)   // 행을 목록 중앙으로(컨테이너만 스크롤)
  }, [marked, segs.length])

  const shown = segs.filter(s => s.abs_s == null || s.abs_s <= maxSec)   // 오늘 현재시각 이후(미래) 이력 숨김
  return (
    <div className="card-box">
      <div className="log-head">
        <div className="log-title">특이사항 <span className="log-count">{shown.length}건</span></div>
      </div>
      <div className="loglist">
        {shown.map(s => {
          const key = `${s.video_id}:${Math.floor(s.start_s)}`
          const lc = levelClass(s.level)
          const active = marked != null && marked === `${s.video_id}:${Math.round(s.start_s)}`   // 그 구간 행 강조
          return (
            <div key={key} ref={active ? activeRef : null}
                 className={`logrow ${s.is_alert ? 'alert ' + lc : ''} ${active ? 'on' : ''}`}
                 onClick={() => onSeek(s.video_id, s.start_s, s)}>
              <span className="logtag">{s.event_type && s.event_type !== 'normal'
                ? <span className={`tag ${s.event_type}`}>{eventLabel(s.event_type)}</span> : null}</span>
              <span className="logts">{s.ts}</span>
              <span className="logcap" title={s.caption}>{s.caption}</span>
              <span className={`logtemp ${lc}`}>{s.is_alert ? `${s.temp.toFixed(1)}℃` : ''}</span>
            </div>
          )
        })}
        {!segs.length && <div className="hint">이 일자에 색인된 구간이 없습니다</div>}
      </div>
    </div>
  )
}
