import CameraGrid from './CameraGrid'
import PlaylistPlayer from './PlaylistPlayer'
import TimeScrubber from './TimeScrubber'
import PlayerControls from './PlayerControls'
import SearchPanel from './SearchPanel'
import HistoryTimeline from './HistoryTimeline'
import SituationLog from './SituationLog'

// 본문: [좌(그리드 or 플레이어+24h바+컨트롤) | 우(검색 + 로그)]. 24h 바 = 유일한 시간 컨트롤.
export default function Console({ cameras, date, focusCam, view, thermal, playlist, playing, live, maxSec = 86400, marked, alertCams = {},
                                 onSelectCamera, onSeekSeg, onScrub, onFocusTime, onGoLive,
                                 onTogglePlay, onStop, onStep }) {
  const focused = !!focusCam

  if (!focused) {
    return (
      <div className="layout focusview">
        <div className="col player-col">
          <CameraGrid cameras={cameras} alertCams={alertCams} thermal={thermal} viewSec={view.sec} viewToken={view.token}
                      playing={playing} live={live} maxSec={maxSec} onSelect={onSelectCamera} />
          <TimeScrubber absSec={view.sec} maxSec={maxSec} live={live} onGoLive={onGoLive} onScrub={onScrub} />
          <PlayerControls playing={playing} live={live} onBack10={() => onStep(-10)}
                          onToggle={onTogglePlay} onFwd10={() => onStep(10)} canFwd={!live} onGoLive={onGoLive} />
        </div>
        <div className="col">
          <SearchPanel onSeek={onSeekSeg} onGoLive={onGoLive} maxSec={maxSec} />
          <SituationLog date={date} maxSec={maxSec} onSelect={(s) => onSeekSeg(s.video_id, s.start_s, s)} />
        </div>
      </div>
    )
  }

  return (
    <div className="layout focusview">
      <div className="col player-col">
        <div className="focus-feed">
          <PlaylistPlayer playlist={playlist} startSec={view.sec} seekToken={view.token}
                          thermal={thermal} playing={playing} live={live} maxSec={maxSec} onTime={onFocusTime} />
          <span className="cam-cell-label"><span className="ld" />{focusCam}</span>
        </div>
        <TimeScrubber absSec={view.sec} maxSec={maxSec} live={live} onGoLive={onGoLive} onScrub={onScrub} />
        <PlayerControls playing={playing} live={live} onBack10={() => onStep(-10)}
                        onToggle={onTogglePlay} onFwd10={() => onStep(10)} canFwd={!live} onGoLive={onGoLive} />
      </div>
      <div className="col">
        <SearchPanel onSeek={onSeekSeg} onGoLive={onGoLive} cam={focusCam} maxSec={maxSec} />
        <HistoryTimeline date={date} cams={[focusCam]} cam={focusCam}
                         marked={marked} maxSec={maxSec} onSeek={onSeekSeg} />
      </div>
    </div>
  )
}
