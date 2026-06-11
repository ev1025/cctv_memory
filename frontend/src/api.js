// FastAPI 백엔드 호출 래퍼 (같은 origin: build 시 FastAPI 서빙 / dev 시 vite proxy)

export async function listCameras() {
  try { const r = await fetch('/cameras'); return await r.json() }
  catch { return { cameras: [], dates: [] } }
}

export async function listAlerts(date, cams) {
  const qs = new URLSearchParams()
  if (date) qs.set('date', date)
  if (cams && cams.length) qs.set('cams', cams.join(','))
  try { const r = await fetch('/alerts?' + qs.toString()); return await r.json() }
  catch { return { alerts: [] } }
}

export async function listVideos() {
  try { const r = await fetch('/videos'); return await r.json() } catch { return { videos: [] } }
}

export async function getSegments(videoId) {
  const r = await fetch('/segments?video_id=' + encodeURIComponent(videoId))
  if (!r.ok) throw new Error('구간 로드 실패')
  return r.json()
}

export async function listHistory(date, cams) {
  const qs = new URLSearchParams()
  if (date) qs.set('date', date)
  if (cams && cams.length) qs.set('cams', cams.join(','))
  try { const r = await fetch('/history?' + qs.toString()); return await r.json() }
  catch { return { segments: [] } }
}

export async function queryVideo(question, specialOnly) {
  const fd = new FormData()
  fd.append('question', question)
  fd.append('special_only', specialOnly)
  const r = await fetch('/query', { method: 'POST', body: fd })
  if (!r.ok) throw new Error('검색 실패 (' + r.status + ')')
  return r.json()
}

// MM:SS (영상 내 타임코드)
export const fmt = (s) => {
  s = Math.floor(s || 0)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

// 위험도/온도 → 상태 클래스
export const levelClass = (level) => (level === '위험' ? 'crit' : level === '주의' ? 'warn' : 'ok')

// 이벤트 유형 → 한국어 라벨 (칩 표시용). 클래스는 `tag ${event_type}` 로 직접 사용.
export const EVENT_KO = {
  // AI-Hub GT 유형
  falldown: '쓰러짐', fight: '싸움', invasion: '침입', gathering: '군집',
  crowd: '인파밀집', flood: '침수',
  // (주차장 가정 유형 — 호환)
  fall: '낙상', loitering: '배회', vehicle_interaction: '차량접촉',
  smoking: '흡연', flammable: '인화물', intrusion: '침입',
  normal: '정상', unknown: '미상',
}
export const eventLabel = (t) => EVENT_KO[t] || t || '정상'
