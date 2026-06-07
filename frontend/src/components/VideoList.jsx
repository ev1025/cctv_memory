import { useState, useEffect } from 'react'
import { listVideos } from '../api'

export default function VideoList({ selectedId, onSelect }) {
  const [videos, setVideos] = useState([])

  useEffect(() => { listVideos().then(j => setVideos(j.videos || [])) }, [])

  const riskText = (risks) => {
    const r = Object.entries(risks || {}).filter(([k]) => k !== 'none')
    return r.length ? r.map(([k, n]) => `${k} ${n}`).join(' · ') : '정상'
  }

  return (
    <div className="vlist">
      <div className="vlist-title">색인된 영상 ({videos.length})</div>
      {videos.map(v => (
        <div key={v.video_id}
             className={`vitem ${v.video_id === selectedId ? 'sel' : ''}`}
             onClick={() => onSelect(v.video_id)}>
          <div className="vname">{v.video_id}</div>
          <div className="vmeta">{v.segments}구간 · {riskText(v.risks)}</div>
        </div>
      ))}
      {!videos.length && <div className="hint">색인된 영상이 없습니다</div>}
    </div>
  )
}
