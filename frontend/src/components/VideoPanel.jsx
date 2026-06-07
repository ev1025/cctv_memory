import { useState } from 'react'
import { indexVideo } from '../api'

export default function VideoPanel({ playerRef, videoId, onIndexed }) {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('')
  const [busy, setBusy] = useState(false)

  const handleIndex = async () => {
    if (!file) return
    setBusy(true)
    setStatus('색인 중… (구간 분할 → VLM 분석, 수 분 소요)')
    try {
      const j = await indexVideo(file)
      onIndexed(j.video_id)
      setStatus(`색인 완료: ${j.segments} 구간`)
    } catch (e) {
      setStatus('실패: ' + e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel">
      <div className="upload">
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleIndex} disabled={busy || !file}>
          {busy ? '색인 중…' : '색인'}
        </button>
        {status && <span className="status">{status}</span>}
      </div>
      <video
        ref={playerRef}
        controls
        src={videoId ? `/video/${encodeURIComponent(videoId)}` : undefined}
      />
      {!videoId && <div className="placeholder">영상을 업로드해 색인하면 여기 재생됩니다</div>}
    </div>
  )
}
