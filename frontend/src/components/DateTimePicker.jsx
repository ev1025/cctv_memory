import { useState, useRef, useEffect, Fragment } from 'react'

const pad = (n) => String(n).padStart(2, '0')
const WD = ['일', '월', '화', '수', '목', '금', '토']

// 테마 일치 커스텀 날짜+시간 선택기. 다크 캘린더 + 시/분/초 스테퍼.
// live=실시간 추적(빨간 점), 특정 시점 선택 시 '이력'(앰버). '실시간으로'로 복귀.
export default function DateTimePicker({ date, time, live, onPick }) {
  const [open, setOpen] = useState(false)
  const [view, setView] = useState(null)        // {y, m(1-12)} 캘린더가 보는 달 (null=선택값 추종)
  const ref = useRef(null)

  useEffect(() => {
    const h = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])

  const [Y, M, D] = (date || '2026-01-01').split('-').map(Number)
  const [hh, mm, ss] = (time || '00:00:00').split(':').map(Number)
  const vy = view ? view.y : Y
  const vm = view ? view.m : M

  const pickDay = (d) => onPick(`${vy}-${pad(vm)}-${pad(d)}`, time)
  const shiftMonth = (delta) => {
    let y = vy, m = vm + delta
    if (m < 1) { m = 12; y-- } else if (m > 12) { m = 1; y++ }
    setView({ y, m })
  }
  const bump = (f, delta) => {
    let h = hh, mi = mm, s = ss
    if (f === 'h') h = (h + delta + 24) % 24
    if (f === 'm') mi = (mi + delta + 60) % 60
    if (f === 's') s = (s + delta + 60) % 60
    onPick(date, `${pad(h)}:${pad(mi)}:${pad(s)}`)
  }

  const firstWd = new Date(vy, vm - 1, 1).getDay()
  const days = new Date(vy, vm, 0).getDate()
  const cells = [...Array(firstWd).fill(null), ...Array.from({ length: days }, (_, i) => i + 1)]
  const T = { h: hh, m: mm, s: ss }

  return (
    <div className="dtpicker" ref={ref}>
      <button className={`dt-trigger ${live ? 'live' : 'hist'}`}
              onClick={() => setOpen(o => { if (!o) setView(null); return !o })} title="날짜·시간 선택">
        <svg className="dt-ico" width="14" height="14" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="12" cy="12" r="9" /><path d="M12 7.5v4.8l3 1.8" />
        </svg>
        <span className="dt-d">{date}</span>
        <span className="dt-t">{time}</span>
        <span className="dt-dot" />
      </button>
      {open && (
        <div className="dt-pop">
          <div className="dt-cal-head">
            <button type="button" onClick={() => shiftMonth(-1)} aria-label="이전 달">‹</button>
            <b>{vy}.{pad(vm)}</b>
            <button type="button" onClick={() => shiftMonth(1)} aria-label="다음 달">›</button>
          </div>
          <div className="dt-wd">{WD.map(w => <span key={w}>{w}</span>)}</div>
          <div className="dt-days">
            {cells.map((d, i) => d ? (
              <button type="button" key={i}
                      className={`dt-day ${d === D && vy === Y && vm === M ? 'on' : ''}`}
                      onClick={() => pickDay(d)}>{d}</button>
            ) : <span key={i} />)}
          </div>
          <div className="dt-time">
            {['h', 'm', 's'].map((f, i) => (
              <Fragment key={f}>
                {i > 0 && <span className="dt-colon">:</span>}
                <div className="dt-tf">
                  <button type="button" onClick={() => bump(f, -1)} aria-label="이전">▲</button>
                  <span>{pad(T[f])}</span>
                  <button type="button" onClick={() => bump(f, 1)} aria-label="다음">▼</button>
                </div>
              </Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
