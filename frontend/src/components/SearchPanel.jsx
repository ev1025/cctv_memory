import { useState } from 'react'
import { queryVideo, fmt } from '../api'

export default function SearchPanel({ onSeek, hasVideo, hasIndex }) {
  const [q, setQ] = useState('')
  const [fireOnly, setFireOnly] = useState(false)
  const [answer, setAnswer] = useState('')
  const [segments, setSegments] = useState([])
  const [busy, setBusy] = useState(false)
  const [searched, setSearched] = useState(false)

  const search = async () => {
    if (!q.trim()) return
    setBusy(true)
    setSearched(true)
    try {
      const j = await queryVideo(q, fireOnly)
      setAnswer(j.answer || '')
      setSegments(j.segments || [])
    } catch (e) {
      setAnswer('검색 실패: ' + e.message)
      setSegments([])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel">
      <div className="search">
        <input
          type="text"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          placeholder="예: 사람이 쓰러지는 장면 / 불꽃이 보이는 구간 / 기계가 넘어짐"
        />
        <label className="chk">
          <input type="checkbox" checked={fireOnly} onChange={(e) => setFireOnly(e.target.checked)} />
          화재만
        </label>
        <button onClick={search} disabled={busy}>{busy ? '검색 중…' : '검색'}</button>
      </div>

      {answer && <div className="answer">{answer}</div>}

      {segments.length > 0 ? (
        <>
          <div className="seg-title">근거 구간 · 클릭하면 영상이 그 시각으로 이동</div>
          {segments.map((s, i) => (
            <div key={i} className="card" onClick={() => onSeek(s.video_id, s.start_s)} title="클릭 → 해당 시각 재생">
              <img
                src={`/thumb?path=${encodeURIComponent(s.thumb || '')}`}
                onError={(e) => (e.target.style.visibility = 'hidden')}
                alt=""
              />
              <div className="meta">
                <span className="ts">{fmt(s.start_s)}–{fmt(s.end_s)}</span>
                <span className={`tag ${s.risk_type || 'none'}`}>{s.risk_type || 'none'}</span>
                <span className="score">score {s.score}</span>
                <div className="cap">{s.caption}</div>
              </div>
            </div>
          ))}
        </>
      ) : (
        <div className="hint">
          {busy ? '검색 중…'
            : searched ? '결과 없음.'
            : hasIndex ? '자연어로 검색하세요. 결과 구간을 클릭하면 영상이 점프합니다.'
            : '영상을 색인한 뒤 검색하세요.'}
        </div>
      )}
    </div>
  )
}
