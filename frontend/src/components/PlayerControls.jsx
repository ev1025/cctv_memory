// 영상 재생 컨트롤 (NVR풍) — 24h 바 밑. 10초 뒤로 / 재생·일시정지 / 10초 앞으로. (실시간/LIVE는 시간바로 이동)
const Ico = ({ d }) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d={d} /></svg>
)
const REWIND = 'M11 6v12l-8.5-6L11 6zm9.5 0v12L12 12l8.5-6z'
const FORWARD = 'M13 6v12l8.5-6L13 6zM3.5 6v12L12 12 3.5 6z'
const PLAY = 'M8 5v14l11-7L8 5z'
const PAUSE = 'M6.5 5h3.5v14H6.5zM14 5h3.5v14H14z'

export default function PlayerControls({ playing, live, onBack10, onToggle, onFwd10, canFwd = true, onGoLive }) {
  return (
    <div className="pcontrols">
      <button className="pc-btn" onClick={onBack10} title="10초 뒤로"><Ico d={REWIND} /></button>
      <button className="pc-btn pc-main" onClick={onToggle} title={playing ? '일시정지' : '재생'}>
        <Ico d={playing ? PAUSE : PLAY} />
      </button>
      <button className="pc-btn" onClick={onFwd10} disabled={!canFwd}
              title={canFwd ? '10초 앞으로' : '현재 시각(실시간)'}><Ico d={FORWARD} /></button>
    </div>
  )
}
