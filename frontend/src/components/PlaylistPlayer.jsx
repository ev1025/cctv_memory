import { useRef, useEffect } from 'react'

// 카메라 1대의 playlist(=[{video_id, day_offset, duration}] 오름차순)를 받아
// startSec(24h 절대초) 위치의 클립을 재생하고, 끝나면 다음 클립으로 자동 전환.
//   - 외부 점프: seekToken 이 바뀔 때만 그 시각으로 이동
//   - mini: 그리드 셀(작게·음소거·그 시각 프레임만 정지표시 → GPU 디코드 절약). 포커스만 연속 재생.
//   - playing: 포커스 재생/일시정지(컨트롤 바). onTime: 슬라이더 동기화.
export default function PlaylistPlayer({ playlist, startSec, seekToken, thermal, mini, playing = true, live = false, maxSec = 86400, onTime }) {
  const vRef = useRef(null)
  const posRef = useRef(startSec || 0)
  const pl = playlist || []

  function locate(sec) {
    // 실시간(live)은 '현재'를 재생해야 하므로 미래컷 적용 안 함. 리뷰(비실시간)에서만 maxSec 초과를 '아직 녹화 전'으로.
    if (!live && sec > maxSec) return { clip: null, offset: 0, gap: true, future: true }
    for (const c of pl) {
      if (sec >= c.day_offset && sec < c.day_offset + c.duration)
        return { clip: c, offset: sec - c.day_offset, gap: false }
    }
    const next = pl.find((c) => c.day_offset >= sec && (live || c.day_offset <= maxSec))   // 빈 구간 → 다음 클립
    return { clip: next || null, offset: 0, gap: true }
  }

  function load(sec) {
    const p = vRef.current
    if (!p) return
    posRef.current = sec
    const loc = locate(sec)
    if (!loc.clip) { try { p.removeAttribute('src'); p.load() } catch (e) {} ; return }
    const changed = p.dataset.vid !== loc.clip.video_id || p.dataset.th !== String(thermal)
    if (changed) {
      p.dataset.vid = loc.clip.video_id
      p.dataset.th = String(thermal)
      p.src = `/video/${encodeURIComponent(loc.clip.video_id)}${thermal ? '?type=thermal' : ''}`
      p.load()
    }
    const go = () => {
      try { p.currentTime = loc.gap ? 0 : loc.offset } catch (e) {}
      if (!playing) { try { p.pause() } catch (e2) {} } else p.play().catch(() => {})
    }
    if (changed) p.addEventListener('loadedmetadata', go, { once: true })
    else go()
  }

  // 외부 점프(seekToken/thermal/playlist) → startSec 위치로
  useEffect(() => { load(startSec || 0) }, [seekToken, thermal, pl.length])
  // 재생/일시정지 토글(포커스)
  useEffect(() => {
    const p = vRef.current
    if (!p) return
    if (playing) p.play().catch(() => {}); else p.pause()
  }, [playing])

  function handleEnded() {
    const loc = locate(posRef.current)
    const curOff = loc.clip ? loc.clip.day_offset : -1
    const curEnd = loc.clip ? loc.clip.day_offset + loc.clip.duration : posRef.current
    const next = pl.find((c) => c.day_offset >= curEnd - 0.5 && c.day_offset > curOff && (live || c.day_offset <= maxSec))
    if (next) { load(next.day_offset); if (onTime) onTime(next.day_offset) }   // 리뷰는 현재시각 도달 시 정지, 실시간은 계속
  }
  function handleTime(e) {
    const loc = locate(posRef.current)
    if (loc.clip && !loc.gap) {
      posRef.current = loc.clip.day_offset + e.target.currentTime
      if (onTime) onTime(posRef.current)
    }
  }

  const loc = locate(posRef.current)
  return (
    <div className={`feed-wrap ${mini ? 'mini' : ''}`}>
      <video ref={vRef} className={`feed ${thermal ? 'thermal' : ''}`} muted={mini}
             autoPlay={!mini} playsInline preload="metadata"
             onEnded={handleEnded} onTimeUpdate={handleTime} />
      {!loc.clip && (
        <div className="nosignal">
          {loc.future ? '아직 녹화 전' : '신호 없음'}{mini ? '' : (loc.future ? ' (현재 시각 이후)' : ' (이 시각에 녹화 없음)')}
        </div>
      )}
    </div>
  )
}
