// FastAPI 백엔드 호출 래퍼 (같은 origin: build 시 FastAPI 서빙 / dev 시 vite proxy)

export async function indexVideo(file) {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch('/index-video', { method: 'POST', body: fd })
  if (!r.ok) throw new Error('색인 실패 (' + r.status + ')')
  return r.json()
}

export async function queryVideo(question, fireOnly) {
  const fd = new FormData()
  fd.append('question', question)
  fd.append('fire_only', fireOnly)
  const r = await fetch('/query', { method: 'POST', body: fd })
  if (!r.ok) throw new Error('검색 실패 (' + r.status + ')')
  return r.json()
}

export async function memoryStatus() {
  try {
    const r = await fetch('/memory-status')
    return await r.json()
  } catch {
    return { count: 0 }
  }
}

export async function listVideos() {
  try { const r = await fetch('/videos'); return await r.json() } catch { return { videos: [] } }
}

export async function getSegments(videoId) {
  const r = await fetch('/segments?video_id=' + encodeURIComponent(videoId))
  if (!r.ok) throw new Error('구간 로드 실패')
  return r.json()
}

export const fmt = (s) => {
  s = Math.floor(s)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}
