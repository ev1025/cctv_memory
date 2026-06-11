import CameraGrid from './CameraGrid'
import Viewer from './Viewer'
import SearchPanel from './SearchPanel'
import HistoryTimeline from './HistoryTimeline'
import SituationLog from './SituationLog'

// 컨트롤은 헤더(App)로 통합. Console 은 본문만: [좌 컨텐츠 | 우(검색 + 로그)].
export default function Console({ cameras, date, focus, thermal, onSelectCamera, onSeekSeg, onGoLive }) {
  const focused = !!focus.videoId

  if (!focused) {
    // 전체 CCTV (그리드)
    return (
      <div className="layout focusview">
        <div className="col">
          <CameraGrid cameras={cameras} thermal={thermal} onSelect={onSelectCamera} />
        </div>
        <div className="col">
          <SearchPanel onSeek={onSeekSeg} onGoLive={onGoLive} />
          <SituationLog date={date} onSelect={(s) => onSeekSeg(s.video_id, s.start_s, s)} />
        </div>
      </div>
    )
  }

  // 개별 캠 (포커스)
  const activeKey = `${focus.videoId}:${Math.floor(focus.time)}`
  return (
    <div className="layout focusview">
      <div className="col">
        <Viewer videoId={focus.videoId} cameraId={focus.cameraId} time={focus.time}
                seekToken={focus.seekToken} live={focus.live} thermal={thermal} />
      </div>
      <div className="col">
        <SearchPanel onSeek={onSeekSeg} onGoLive={onGoLive} />
        <HistoryTimeline date={date} cams={[focus.cameraId]} cam={focus.cameraId}
                         activeKey={activeKey} onSeek={onSeekSeg} />
      </div>
    </div>
  )
}
