import PlaylistPlayer from './PlaylistPlayer'

// 3×3 관제 월. 각 셀이 '공통 시각(viewSec)'의 클립을 재생 → 시간 옮기면 9개 다 점프.
export default function CameraGrid({ cameras, thermal, viewSec, viewToken, playing, live = false, maxSec = 86400, onSelect }) {
  const cells = Array.from({ length: 9 }, (_, i) => cameras[i] || null)
  return (
    <div className={`cam-grid ${thermal ? 'thermal' : ''}`}>
      {cells.map((c, i) => c ? (
        <div className="cam-cell live" key={c.camera_id} onClick={() => onSelect(c)}
             title={`${c.camera_id} 선택`}>
          <PlaylistPlayer playlist={c.playlist} startSec={viewSec} seekToken={viewToken}
                          thermal={thermal} playing={playing} live={live} maxSec={maxSec} mini />
          <span className="cam-cell-label"><span className="ld" />{c.camera_id}</span>
        </div>
      ) : (
        <div className="cam-cell off" key={`off-${i}`}>
          <span className="off-text">신호 없음</span>
        </div>
      ))}
    </div>
  )
}
