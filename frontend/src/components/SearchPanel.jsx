import { useState } from 'react'
import { queryVideo, fmt, eventLabel } from '../api'

// 자연어로 이력 검색 → RAG 답변 + 근거 구간(클릭 → 점프). 평소엔 비어 있고, 검색할 때만 결과 표시.
export default function SearchPanel({ onSeek, onGoLive }) {
  const [q, setQ] = useState('')
  const [answer, setAnswer] = useState('')
  const [segments, setSegments] = useState([])
  const [busy, setBusy] = useState(false)
  const [searched, setSearched] = useState(false)

  const search = async () => {
    if (!q.trim()) return
    setBusy(true); setSearched(true)
    try {
      const j = await queryVideo(q, false)
      setAnswer(j.answer || ''); setSegments(j.segments || [])
    } catch (e) {
      setAnswer('검색 실패: ' + e.message); setSegments([])
    } finally { setBusy(false) }
  }

  const clear = () => { setQ(''); setAnswer(''); setSegments([]); setSearched(false); onGoLive && onGoLive() }
  const hasResult = searched || segments.length > 0

  return (
    <div className="card-box searchbox">
      <div className="search">
        <input type="text" value={q} onChange={(e) => setQ(e.target.value)}
               onKeyDown={(e) => e.key === 'Enter' && search()}
               placeholder="행동 검색 — 예: 트렁크 여는 사람, 오래 서성인 사람" />
        <button onClick={search} disabled={busy}>{busy ? '검색 중…' : '검색'}</button>
        {hasResult && <button className="clear-btn" onClick={clear}>초기화</button>}
      </div>
      {answer && <div className="answer">{answer}</div>}
      {segments.map((s, i) => (
        <div key={i} className="scard" onClick={() => onSeek(s.video_id, s.start_s, s)}>
          <img src={`/thumb?path=${encodeURIComponent(s.thumb || '')}`}
               onError={(e) => (e.target.style.visibility = 'hidden')} alt="" />
          <div style={{ minWidth: 0 }}>
            <span className="ts">{fmt(s.start_s)}–{fmt(s.end_s)}</span>
            <span className={`tag ${s.event_type || 'normal'}`}>{eventLabel(s.event_type)}</span>
            {s.dwell_s > 0 && <span className="dwell">체류 {Math.round(s.dwell_s)}s</span>}
            <span className="score">score {s.score}</span>
            <div className="cap">{s.caption}</div>
          </div>
        </div>
      ))}
      {searched && !busy && segments.length === 0 && !answer &&
        <div className="hint">검색 결과 없음 (유사도 0.5 미만)</div>}
    </div>
  )
}
