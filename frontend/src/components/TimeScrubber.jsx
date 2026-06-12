import { useState } from 'react'

// 24시간(00:00:00~23:59:59) 슬라이더(초단위). 우측 HH:MM:SS 입력(24h)하면 그 시각으로 점프.
const DAY = 86400
const pad = (n) => String(n).padStart(2, '0')
const hms = (sec) => { sec = ((Math.floor(sec) % DAY) + DAY) % DAY; return `${pad(Math.floor(sec / 3600))}:${pad(Math.floor((sec % 3600) / 60))}:${pad(sec % 60)}` }
const toSec = (v) => { const a = (v || '').split(':').map(Number); if (a.length < 2 || a.some(isNaN)) return null; return (((a[0] || 0) * 3600 + (a[1] || 0) * 60 + (a[2] || 0)) % DAY + DAY) % DAY }

export default function TimeScrubber({ absSec, markers = [], maxSec = DAY, live = false, onGoLive, onScrub }) {
  const [edit, setEdit] = useState(null)
  const value = edit != null ? edit : hms(absSec)
  const clamp = (s) => Math.min(s, maxSec)                          // 미래로는 못 감
  const commit = () => { const s = toSec(value); if (s != null) onScrub(clamp(s)); setEdit(null) }
  const futurePct = Math.max(0, Math.min(100, (maxSec / DAY) * 100))
  const [hover, setHover] = useState(null)               // 재생바 위 마우스 위치의 시각 툴팁
  const onTrackMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const pct = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
    setHover({ x: pct * rect.width, sec: pct * DAY })
  }

  return (
    <div className="scrubber">
      <div className="sc-track" onMouseMove={onTrackMove} onMouseLeave={() => setHover(null)}>
        {hover && (
          <span className={`sc-hovertip ${hover.sec > maxSec ? 'future' : ''}`} style={{ left: `${hover.x}px` }}>
            {hms(hover.sec)}
          </span>
        )}
        {futurePct < 100 && <span className="sc-future" style={{ left: `${futurePct}%` }} title="아직 녹화 전" />}
        {markers.filter((m) => m.abs_s <= maxSec).map((m, i) => (
          <span key={i} className={`sc-mk ${m.level || ''}`}
                style={{ left: `${(m.abs_s / DAY) * 100}%` }} title={m.caption || ''} />
        ))}
        <input type="range" min={0} max={DAY - 1} step={1} value={Math.floor(Math.min(absSec, maxSec))}
               onChange={(e) => onScrub(clamp(Number(e.target.value)))} className="sc-range" />
      </div>
      {/* 입력창 바로 왼쪽: 실시간이면 LIVE 깜빡, 아니면 실시간 버튼(같은 자리 교체) */}
      {live ? (
        <span className="sc-live" title="실시간"><span className="b" />LIVE</span>
      ) : (
        <button className="sc-golive" onClick={onGoLive} title="실시간으로"><span className="b" />실시간</button>
      )}
      <input type="text" inputMode="numeric" maxLength={8} className="sc-time" value={value}
             onChange={(e) => setEdit(e.target.value)} onBlur={commit}
             onKeyDown={(e) => { if (e.key === 'Enter') { commit(); e.target.blur() } }} />
    </div>
  )
}
