import { useEffect, useRef } from 'react'

// 포커스된 카메라의 큰 영상. 일반/열화상은 상단바에서 thermal prop 으로 받는다(토글은 Console).
// 열화상 실데이터 없으면 일반 영상에 CSS 필터(플레이스홀더). 실제 {id}__thermal.mp4 넣으면 그 영상 재생.
export default function Viewer({ videoId, cameraId, time, seekToken, live, thermal }) {
  const vRef = useRef(null)
  const posRef = useRef(0)            // 토글 시 현재 위치 유지
  const curId = useRef(null)
  const srcFor = () => `/video/${encodeURIComponent(videoId)}${thermal ? '?type=thermal' : ''}`

  // 외부 점프(카메라 선택/이력 클릭)
  useEffect(() => {
    const p = vRef.current
    if (!p || !videoId) return
    const changed = curId.current !== videoId
    curId.current = videoId
    if (changed) { p.src = srcFor(); p.load() }
    const go = () => { try { p.currentTime = time || 0 } catch (e) {} ; if (!live) p.play().catch(() => {}) }
    if (changed || p.readyState < 1) p.addEventListener('loadedmetadata', go, { once: true })
    else go()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoId, seekToken])

  // 일반/열화상 토글 → 현재 위치 유지하며 src 교체
  useEffect(() => {
    const p = vRef.current
    if (!p || !videoId) return
    const at = posRef.current || 0, playing = !p.paused
    p.src = srcFor(); p.load()
    const go = () => { try { p.currentTime = at } catch (e) {} ; if (playing) p.play().catch(() => {}) }
    p.addEventListener('loadedmetadata', go, { once: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [thermal])

  return (
    <div className="viewer">
      <div className="stage">
        <video ref={vRef} className={`feed ${thermal ? 'thermal' : ''}`}
               controls playsInline muted={live}
               onTimeUpdate={(e) => { posRef.current = e.target.currentTime }} />
        <div className="ovl">
          <span className="cam-badge">{cameraId || '—'}</span>
          {live && <span className="live-badge"><span className="b" />LIVE</span>}
        </div>
      </div>
    </div>
  )
}
