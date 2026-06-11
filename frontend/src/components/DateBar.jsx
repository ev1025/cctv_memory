// 일자 선택 + CCTV 다중 선택(이력 탭 상단). 선택 변경 → 알림/이력 필터.
export default function DateBar({ dates, date, onDate, cameras, selCams, onToggleCam, onAllCams }) {
  const allOn = cameras.length > 0 && selCams.length === cameras.length
  return (
    <div className="ctxbar">
      <div className="grp">
        <label className="lbl">일자</label>
        <input type="date" value={date || ''} list="date-list" onChange={(e) => onDate(e.target.value)} />
        <datalist id="date-list">{dates.map(d => <option key={d} value={d} />)}</datalist>
      </div>
      <div className="grp">
        <label className="lbl">카메라</label>
        <div className="camchips">
          <span className={`camchip ${allOn ? 'on' : ''}`} onClick={onAllCams}>
            <span className="cdot" />전체
          </span>
          {cameras.map(c => (
            <span key={c.camera_id}
                  className={`camchip ${selCams.includes(c.camera_id) ? 'on' : ''}`}
                  onClick={() => onToggleCam(c.camera_id)}>
              <span className="cdot" />{c.camera_id}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
