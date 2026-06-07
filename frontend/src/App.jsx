import { useState, useRef } from 'react'
import VideoList from './components/VideoList'
import Timeline from './components/Timeline'
import SearchPanel from './components/SearchPanel'
import { getSegments } from './api'

export default function App() {
  const [videoId, setVideoId] = useState(null)
  const [segments, setSegments] = useState([])
  const playerRef = useRef(null)

  // 라이브러리에서 영상 선택 → player 로드 + 구간 이력(타임라인) 로드
  const selectVideo = async (vid) => {
    setVideoId(vid)
    const p = playerRef.current
    if (p) { p.src = `/video/${encodeURIComponent(vid)}`; p.load() }
    const j = await getSegments(vid)
    setSegments(j.segments || [])
  }

  // 구간/검색 결과 클릭 → (필요시 해당 영상 로드 후) 그 시각으로 점프
  const seek = (vid, s) => {
    const p = playerRef.current
    if (!p || !vid) return
    const src = `/video/${encodeURIComponent(vid)}`
    if (!(p.src || '').endsWith(src)) {
      p.src = src
      setVideoId(vid)
      getSegments(vid).then(j => setSegments(j.segments || []))
      p.addEventListener('loadeddata', () => { p.currentTime = s; p.play() }, { once: true })
    } else {
      p.currentTime = s
      p.play()
    }
  }

  return (
    <div className="app">
      <header>
        <h1>🎥 Video Memory</h1>
        <span className="sub">색인된 영상 라이브러리 — 선택해 이력 보기 + 자연어 검색</span>
      </header>
      <div className="layout">
        <VideoList selectedId={videoId} onSelect={selectVideo} />
        <div className="center">
          <video ref={playerRef} controls src={videoId ? `/video/${encodeURIComponent(videoId)}` : undefined} />
          {!videoId && <div className="placeholder">← 왼쪽에서 영상을 선택하세요</div>}
        </div>
        <div className="right">
          <SearchPanel onSeek={seek} />
          <Timeline segments={segments} videoId={videoId} onSeek={seek} />
        </div>
      </div>
    </div>
  )
}
