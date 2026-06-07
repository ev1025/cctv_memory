import { fmt } from '../api'

export default function Timeline({ segments, videoId, onSeek }) {
  if (!videoId) {
    return <div className="timeline"><div className="hint">영상을 선택하면 구간 이력이 표시됩니다</div></div>
  }
  return (
    <div className="timeline">
      <div className="tl-title">구간 이력 ({segments.length}) · 클릭 → 점프</div>
      {segments.map((s, i) => {
        const risky = s.risk_type && s.risk_type !== 'none'
        return (
          <div key={i} className={`tlrow ${risky ? 'risk' : ''}`} onClick={() => onSeek(videoId, s.start_s)}>
            <span className="tlts">{fmt(s.start_s)}</span>
            <span className={`tag ${s.risk_type || 'none'}`}>{s.risk_type || 'none'}</span>
            <span className="tlcap">{s.caption}</span>
          </div>
        )
      })}
    </div>
  )
}
