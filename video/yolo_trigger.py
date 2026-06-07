"""
yolo_trigger.py — #1·2 1차 필터(YOLOv8) → 2차 분석(VLM) 융합 파이프라인

[왜] 선박 CCTV 는 24시간 돌아간다. 전 프레임을 무거운 VLM 에 넣으면 실시간성·VRAM 이 버티지 못한다.
     → 가벼운 YOLOv8 로 '사람(관심 객체)'이 잡힐 때만, 그 구간 프레임을 모아 VLM(multi-image)으로 분석.

[흐름]  영상 → sample_fps 로 균일 샘플 → YOLO 사람 감지(1차 트리거)
        → 감지 프레임 누적 → trigger_frames 장 모이면 VLM caption_frames(2차 시계열 분석)

[비고] YOLOv8n(COCO 사전학습)은 'person'(class 0)을 바로 감지. 가중치는 최초 1회 자동 다운로드.
       실제 도메인에선 person 외에 'fire/smoke' 커스텀 학습이나 별도 디텍터를 병행하면 된다.
"""
import os

import cv2
from PIL import Image

PERSON_CLASS = 0   # COCO: person


class YoloVlmPipeline:
    def __init__(self, vlm_backend="qwen2-vl", yolo_weights="yolov8n.pt", conf=0.4):
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_weights)             # COCO 사전학습(최초 1회 자동 다운로드)
        self.conf = conf
        self.vlm_backend = vlm_backend
        self._cap = None                            # VLM 은 첫 트리거 때 lazy 로드

    def _vlm(self):
        if self._cap is None:
            from image_to_text import VLMCaptioner
            self._cap = VLMCaptioner(self.vlm_backend).load()
        return self._cap

    def detect_persons(self, frame_bgr):
        """YOLO 로 사람 감지 → (감지 여부, 사람 수)."""
        res = self.yolo(frame_bgr, conf=self.conf, classes=[PERSON_CLASS], verbose=False)[0]
        return len(res.boxes) > 0, len(res.boxes)

    def process_video(self, video_path, sample_fps=1.0, trigger_frames=3, prompt=None, max_triggers=None):
        """영상 → 사람 감지 프레임만 샘플링 → trigger_frames 장 모이면 VLM 분석.

        max_triggers: 지정 시 그 횟수만큼만 분석하고 조기 종료(테스트/실시간 비용 제한).
        반환: [{"frames":[프레임 idx…], "persons":최대 사람수, "text":VLM 분석}] (트리거 구간별)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / sample_fps)))   # sample_fps 로 균일 샘플(초당 N장)
        idx, buffer, results = 0, [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                person, n = self.detect_persons(frame)
                if person:                              # ── 1차 필터 통과 ──
                    buffer.append((idx, n, frame))
                    if len(buffer) >= trigger_frames:
                        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for _, _, f in buffer]
                        text = self._vlm().caption_frames(imgs, prompt)   # ── 2차 VLM 시계열 ──
                        results.append({
                            "frames": [i for i, _, _ in buffer],
                            "persons": max(n for _, n, _ in buffer),
                            "text": text,
                        })
                        buffer = []
                        if max_triggers and len(results) >= max_triggers:
                            break
            idx += 1
        cap.release()
        return results

    def detect_image(self, image_path):
        """단일 이미지: 사람 감지만(트리거 테스트용)."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(image_path)
        person, n = self.detect_persons(frame)
        return {"image": os.path.basename(image_path), "person": person, "count": n}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python yolo_trigger.py <video_or_image_path> [vlm_backend]")
        sys.exit(0)
    path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "qwen2-vl"
    pipe = YoloVlmPipeline(vlm_backend=backend)
    if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        for r in pipe.process_video(path):
            print(r)
    else:
        print(pipe.detect_image(path))   # 이미지면 사람 감지만(VLM 미호출)
