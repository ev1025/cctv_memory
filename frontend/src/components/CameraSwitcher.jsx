import { levelClass } from '../api'

// 단일 포커스 카메라 전환 목록(좌). 카메라별 최고 온도/상태를 알림에서 파생해 표시.
export default function CameraSwitcher({ cameras, alerts, focusCam, onSelect }) {
  const byCam = {}
  for (const a of alerts || []) {
    const c = byCam[a.camera_id]
    if (!c || a.temp > c.temp) byCam[a.camera_id] = a
  }
  return (
    <div className="card-box">
      <div className="box-title">카메라 <span className="n">{cameras.length}</span></div>
      <div className="camswitch">
        {cameras.map(c => {
          const top = byCam[c.camera_id]
          const lc = top ? levelClass(top.level) : 'ok'
          return (
            <div key={c.camera_id}
                 className={`camrow ${c.camera_id === focusCam ? 'on' : ''}`}
                 onClick={() => onSelect(c, top)}>
              <span className={`feed ${lc}`} />
              <div style={{ minWidth: 0 }}>
                <div className="cname">{c.camera_id}</div>
                <div className="cmeta">{c.videos.length}개 영상 · {c.segments}구간</div>
              </div>
              <span className={`ctemp ${lc}`}>{top ? `${top.temp.toFixed(0)}℃` : '정상'}</span>
            </div>
          )
        })}
        {!cameras.length && <div className="hint">색인된 카메라가 없습니다</div>}
      </div>
    </div>
  )
}
