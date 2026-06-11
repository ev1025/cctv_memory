import { useState, useEffect } from 'react'
import Console from './components/Console'
import DateTimePicker from './components/DateTimePicker'
import { listCameras } from './api'

const EMPTY = { videoId: null, cameraId: null, time: 0, temp: null, level: null,
                seekToken: 0, alertId: null, live: false }

export default function App() {
  const [cameras, setCameras] = useState([])
  const [dates, setDates] = useState([])
  const [focus, setFocus] = useState(EMPTY)        // videoId 있으면 포커스 뷰, 없으면 그리드
  const [thermal, setThermal] = useState(false)    // 일반/열화상 (그리드 전체 / 포커스 뷰어 공통)
  const [dt, setDt] = useState({ date: '', time: '', live: true })  // 날짜+시간 (live=실시간 추적)

  const focusTo = (o) => setFocus(f => ({ ...f, ...o, seekToken: f.seekToken + 1 }))
  const onSelectCamera = (cam) => focusTo({
    videoId: cam.videos[0], cameraId: cam.camera_id, time: 0, live: true, temp: null, level: '정상',
  })
  const onBack = () => setFocus(EMPTY)
  const onSeekSeg = (videoId, start, seg) => focusTo({
    videoId, time: start, live: false,
    cameraId: (seg && seg.camera_id) || focus.cameraId,
    temp: seg && seg.temp != null ? seg.temp : focus.temp,
    level: (seg && seg.level) || focus.level,
  })
  const onGoLive = () => {
    const cam = cameras.find(c => c.camera_id === focus.cameraId)
    if (cam) onSelectCamera(cam)
  }

  // 날짜·시간 선택 → 이력 모드 / '지금' → 실시간 모드
  const onPickDt = (date, time) => setDt({ date, time, live: false })
  const onNowDt = () => setDt(prev => ({ ...prev, live: true }))

  useEffect(() => {
    listCameras().then(j => { setCameras(j.cameras || []); setDates(j.dates || []) })
  }, [])
  useEffect(() => {
    const tick = () => {
      const d = new Date(), p = (n) => String(n).padStart(2, '0')
      const date = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
      const time = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
      setDt(prev => prev.live ? { ...prev, date, time } : prev)   // 실시간 모드일 때만 현재로 갱신
    }
    tick(); const id = setInterval(tick, 1000); return () => clearInterval(id)
  }, [])

  const focused = !!focus.videoId
  const notLive = !dt.live || (focused && !focus.live)        // 실시간 아님(시점 선택 or 영상 점프)
  const backToLive = () => { onNowDt(); if (focused && !focus.live) onGoLive() }
  const goHome = () => { setFocus(EMPTY); onNowDt() }   // 브랜드 클릭 → 초기 화면(그리드+실시간)
  return (
    <div className="app">
      <div className="topbar">
        <div className="brand" onClick={goHome} role="button" tabIndex={0}
             onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && goHome()}
             title="처음 화면으로"><span className="dot" />CCTV 열화상 관제</div>
        <DateTimePicker date={dt.date} time={dt.time} live={dt.live} onPick={onPickDt} />
        {focused && <button className="back-btn" onClick={onBack}>전체 화면</button>}
        {notLive && <button className="live-btn" onClick={backToLive}><span className="b" />실시간 보기</button>}
        <div className="view-toggle">
          <button className={!thermal ? 'on' : ''} onClick={() => setThermal(false)}>일반</button>
          <button className={thermal ? 'on' : ''} onClick={() => setThermal(true)}>열화상</button>
        </div>
      </div>
      <Console cameras={cameras} date={dt.date} focus={focus} thermal={thermal}
               onSelectCamera={onSelectCamera} onSeekSeg={onSeekSeg} onGoLive={onGoLive} />
    </div>
  )
}
