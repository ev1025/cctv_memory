// 3×3 실시간 CCTV 월. 일반/열화상은 상단바(Console)에서 thermal prop 으로 받는다.
// 셀 클릭 → 해당 카메라 포커스. 색인된 카메라는 실시간 셀, 나머지는 '신호 없음'.
export default function CameraGrid({ cameras, thermal, onSelect }) {
  const cells = Array.from({ length: 9 }, (_, i) => cameras[i] || null)
  return (
    <div className={`cam-grid ${thermal ? 'thermal' : ''}`}>
      {cells.map((c, i) => c ? (
        <div className="cam-cell live" key={c.camera_id} onClick={() => onSelect(c)} title={`${c.camera_id} 선택`}>
          <video className={thermal ? 'thermal' : ''}
                 src={`/video/${encodeURIComponent(c.videos[0])}${thermal ? '?type=thermal' : ''}`}
                 autoPlay muted loop playsInline />
          <span className="cam-cell-label"><span className="ld" />{c.camera_id}</span>
        </div>
      ) : (
        <div className="cam-cell off" key={`off-${i}`}>
          <span className="off-text">신호 없음</span>
        </div>
      ))}
    </div>
  )
}
