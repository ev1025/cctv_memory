import { useState } from 'react'
import { queryVideo, fmt, eventLabel } from '../api'

// 타임라인 절대초 → HH:MM:SS (검색 결과를 영상/스크러버 시각과 동일 기준으로 표시)
const clock = (s) => { s = ((Math.floor(s) % 86400) + 86400) % 86400; const p = (n) => String(n).padStart(2, '0'); return `${p(Math.floor(s / 3600))}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}` }

// 자연어로 이력 검색 → RAG 답변 + 근거 구간(클릭 → 점프). 평소엔 비어 있고, 검색할 때만 결과 표시.
export default function SearchPanel({ onSeek, onGoLive, cam, maxSec = 86400 }) {
  const [q, setQ] = useState('')
  const [answer, setAnswer] = useState('')
  const [segments, setSegments] = useState([])
  const [busy, setBusy] = useState(false)
  const [searched, setSearched] = useState(false)
  const [sel, setSel] = useState(null)                                       // 클릭한 결과 카드 강조
  const shown = segments.filter(s => s.abs_s == null || s.abs_s <= maxSec)   // 오늘 현재시각 이후(미래) 결과 제외

  const search = async () => {
    if (!q.trim()) return
    setBusy(true); setSearched(true); setSel(null)
    try {
      const j = await queryVideo(q, false, cam)
      setAnswer(j.answer || ''); setSegments(j.segments || [])
    } catch (e) {
      setAnswer('검색 실패: ' + e.message); setSegments([])
    } finally { setBusy(false) }
  }

  const clear = () => { setQ(''); setAnswer(''); setSegments([]); setSearched(false); setSel(null); onGoLive && onGoLive() }
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
      {shown.map((s, i) => {
        const key = `${s.video_id}:${Math.round(s.start_s)}`
        return (
        <div key={i} className={`scard ${sel === key ? 'on' : ''}`}
             onClick={() => { setSel(key); onSeek(s.video_id, s.start_s, s) }}>
          <img src={`/thumb?path=${encodeURIComponent(s.thumb || '')}`}
               onError={(e) => (e.target.style.visibility = 'hidden')} alt="" />
          <div style={{ minWidth: 0 }}>
            <span className="ts">{s.abs_s != null ? clock(s.abs_s) : `${fmt(s.start_s)}–${fmt(s.end_s)}`}</span>
            <span className={`tag ${s.event_type || 'normal'}`}>{eventLabel(s.event_type)}</span>
            <span className="score">score {s.score}</span>
            <div className="cap">{s.caption}</div>
          </div>
        </div>
        )
      })}
      {searched && !busy && shown.length === 0 && !answer &&
        <div className="hint">검색 결과 없음 (이 카메라·현재시각 이전)</div>}
    </div>
  )
}
