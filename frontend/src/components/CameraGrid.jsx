import PlaylistPlayer from './PlaylistPlayer'

// 2×2 관제 월(데이터 적어 4분할). 각 셀이 '공통 시각(viewSec)'의 클립을 재생 → 시간 옮기면 4개 다 점프.
// alertCams[camera_id] = 'danger' | 'warning' 이면 셀 테두리를 빨강/노랑으로 점멸(경보).
export default function CameraGrid({ cameras, alertCams = {}, thermal, viewSec, viewToken, playing, live = false, maxSec = 86400, onSelect }) {
  const cells = Array.from({ length: 4 }, (_, i) => cameras[i] || null)
  return (
    <div className={`cam-grid ${thermal ? 'thermal' : ''}`}>
      {cells.map((c, i) => {
        if (!c) return (
          <div className="cam-cell off" key={`off-${i}`}>
            <span className="off-text">신호 없음</span>
          </div>
        )
        const alert = alertCams[c.camera_id]           // 'danger' | 'warning' | undefined
        return (
          <div className={`cam-cell live ${alert ? `cam-alert-${alert}` : ''}`} key={c.camera_id}
               onClick={() => onSelect(c)} title={`${c.camera_id} 선택`}>
            <PlaylistPlayer playlist={c.playlist} startSec={viewSec} seekToken={viewToken}
                            thermal={thermal} playing={playing} live={live} maxSec={maxSec} mini />
            {alert && (
              <span className={`cam-alert-badge ${alert}`}>
                {alert === 'danger' ? '● 위험' : '● 주의'}
              </span>
            )}
            <span className="cam-cell-label"><span className="ld" />{c.camera_id}</span>
          </div>
        )
      })}
    </div>
  )
}
