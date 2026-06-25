import { useState, useEffect } from 'react'
import Console from './components/Console'
import DateTimePicker from './components/DateTimePicker'
import { listCameras, listAlerts } from './api'

const timeToSec = (t) => { const [h, m, s] = (t || '0:0:0').split(':').map(Number); return ((h || 0) * 3600 + (m || 0) * 60 + (s || 0)) % 86400 }
const secToTime = (sec) => { sec = ((Math.floor(sec) % 86400) + 86400) % 86400; const p = (n) => String(n).padStart(2, '0'); return `${p(Math.floor(sec / 3600))}:${p(Math.floor((sec % 3600) / 60))}:${p(sec % 60)}` }
const nowWall = () => { const d = new Date(); return d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() }
const todayStr = () => { const d = new Date(), p = (n) => String(n).padStart(2, '0'); return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}` }

export default function App() {
  const [cameras, setCameras] = useState([])
  const [dates, setDates] = useState([])
  const [focusCam, setFocusCam] = useState(null)             // camera_id | null(그리드)
  const [view, setView] = useState({ sec: nowWall(), token: 1 })  // 전 카메라 공통 재생 시각(24h초)
  const [thermal, setThermal] = useState(false)
  const [dateLabel, setDateLabel] = useState('')
  const [live, setLive] = useState(true)
  const [playing, setPlaying] = useState(false)   // 그리드 기본 정지(프레임, GPU 절약), 카메라 선택 시 재생
  const [marked, setMarked] = useState(null)      // 클릭한 이력/검색 결과 식별자 → 그 행 강조(재생 드리프트 무관)
  const [alertCams, setAlertCams] = useState({})  // camera_id → 'danger' | 'warning' (그리드 테두리 경보)

  // 오늘 날짜면 현재 시각까지만(이후는 아직 녹화 전), 과거 날짜면 하루 전체
  const maxSecOf = () => (dateLabel === todayStr() ? nowWall() : 86400)
  // 점프(token++로 전 카메라 재seek). soft=재생 진행 보고(슬라이더만).
  const seek = (sec, opts = {}) => {
    const wrapped = ((Math.floor(sec) % 86400) + 86400) % 86400
    const clamped = Math.min(wrapped, maxSecOf())   // 미래 시각으로는 점프 불가
    setView(v => ({ sec: clamped, token: v.token + 1 }))
    if (opts.cam !== undefined) setFocusCam(opts.cam)
    setLive(!!opts.live)
    if (!opts.keepMark) setMarked(null)            // 스크럽·실시간 등 직접 이동 시 강조 해제
  }
  const onFocusTime = (sec) => { if (!live) setView(v => ({ ...v, sec })) }  // 리뷰일 때만 영상이 시계 구동(실시간은 ticker)
  const playlistOf = (camId) => (cameras.find(c => c.camera_id === camId) || {}).playlist || []

  const onSelectCamera = (cam) => { setFocusCam(cam.camera_id); setPlaying(true) }  // 선택 시 재생
  const onBack = () => { setFocusCam(null); setPlaying(false) }                     // 그리드 복귀 → 정지
  const onScrub = (sec) => seek(sec)                          // 슬라이더 → 전 카메라 점프
  const onTogglePlay = () => setPlaying(p => !p)              // 재생/일시정지
  const onStop = () => setPlaying(false)                      // 정지
  const onStep = (d) => {                                        // ±10초
    if (d > 0 && live) return                                   // 실시간 = 현재 시각, 앞으로 없음
    const t = view.sec + d
    if (d > 0 && t >= maxSecOf()) seek(nowWall(), { live: true })   // 리뷰 중 현재시각 따라잡으면 실시간 전환
    else seek(t)
  }
  const onSeekSeg = (videoId, start, seg) => {              // 검색/이력 클릭 → 그 행 강조 + 점프
    seek(seg && seg.abs_s != null ? seg.abs_s : (start || 0),
         { cam: (seg && seg.camera_id) || focusCam, keepMark: true })
    setMarked(seg ? `${seg.video_id}:${Math.round(seg.start_s)}` : null)   // 그 구간 정확히 강조(이력=구간단위)
  }
  const onGoLive = () => { setDateLabel(todayStr()); seek(nowWall(), { live: true }) }   // 실시간 → 오늘·현재시각

  const onPickDt = (date, time) => { setDateLabel(date); seek(timeToSec(time)) }   // 헤더 시간 선택 → 점프

  useEffect(() => {
    listCameras().then(j => { setCameras(j.cameras || []); setDates(j.dates || []) })
    const d = new Date(), p = (n) => String(n).padStart(2, '0')
    setDateLabel(`${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`)
  }, [])

  // 실시간 재생 중: 시계를 실제 현재시각으로 진행(soft=재seek 없음) → 그리드·포커스 모두 시간이 흐름
  useEffect(() => {
    if (!live || !playing) return
    const id = setInterval(() => setView(v => ({ ...v, sec: nowWall() })), 1000)
    return () => clearInterval(id)
  }, [live, playing])

  // 경보 카메라 폴링: /alerts 의 위험/주의를 카메라별로 모아 그리드 테두리(빨강/노랑)로 표시
  useEffect(() => {
    let stop = false
    const sev = { '위험': 'danger', '주의': 'warning' }, rank = { warning: 1, danger: 2 }
    const pull = async () => {
      const j = await listAlerts(dateLabel)
      if (stop) return
      const map = {}
      for (const a of (j.alerts || [])) {
        const lv = sev[a.level]
        if (lv && (!map[a.camera_id] || rank[lv] > rank[map[a.camera_id]])) map[a.camera_id] = lv
      }
      setAlertCams(map)
    }
    if (dateLabel) { pull(); const id = setInterval(pull, 5000); return () => { stop = true; clearInterval(id) } }
  }, [dateLabel])

  const focused = !!focusCam
  const goHome = () => { setFocusCam(null); setDateLabel(todayStr()); seek(nowWall(), { live: true }) }
  return (
    <div className="app">
      <div className="topbar">
        <div className="brand" onClick={goHome} role="button" tabIndex={0}
             onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && goHome()}
             title="처음 화면으로"><span className="dot" />CCTV 열화상 관제</div>
        <DateTimePicker date={dateLabel} time={secToTime(view.sec)} live={live} onPick={onPickDt} />
        {focused && <button className="back-btn" onClick={onBack}>전체 화면</button>}
        <div className="view-toggle">
          <button className={!thermal ? 'on' : ''} onClick={() => setThermal(false)}>일반</button>
          <button className={thermal ? 'on' : ''} onClick={() => setThermal(true)}>열화상</button>
        </div>
      </div>
      <Console cameras={cameras} date={dateLabel} focusCam={focusCam} view={view} thermal={thermal}
               playlist={playlistOf(focusCam)} playing={playing} live={live} maxSec={maxSecOf()} marked={marked}
               alertCams={alertCams}
               onSelectCamera={onSelectCamera} onSeekSeg={onSeekSeg} onScrub={onScrub}
               onFocusTime={onFocusTime} onGoLive={onGoLive}
               onTogglePlay={onTogglePlay} onStop={onStop} onStep={onStep} />
    </div>
  )
}
