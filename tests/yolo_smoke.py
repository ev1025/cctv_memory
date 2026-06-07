"""yolo_smoke.py — YOLOv8 사람 감지 동작 확인(ultralytics 샘플 bus.jpg, 사람 여럿 포함)."""
from ultralytics import YOLO


def main():
    yolo = YOLO("yolov8n.pt")                       # COCO 사전학습(최초 1회 자동 다운로드)
    res = yolo("https://ultralytics.com/images/bus.jpg",
               classes=[0], conf=0.4, verbose=False)[0]   # class 0 = person
    print(f"\n[YOLO] bus.jpg 사람 감지: {len(res.boxes)}명 (conf>0.4)")
    print("YOLO_SMOKE_DONE")


if __name__ == "__main__":
    main()
