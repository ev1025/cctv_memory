"""
make_sample_image.py — VLM 입력용 테스트 이미지(도형) 생성 (PIL)

실제 운영에선 선박 카메라 프레임이 입력되지만, PoC 검증에선 모델/네트워크와 무관한
'결정론적' 도형 이미지를 만든다. 여러 도형 '조합'을 만들어, VLM 이 단순히 항상
'셋 다 있다'고 답하는 게 아니라 진짜로 도형을 구분하는지까지 확인한다.
"""
import os
from PIL import Image, ImageDraw

COLORS = {"circle": "#e74c3c", "triangle": "#27ae60", "square": "#2980b9"}

# 입력 조합: key → 그릴 도형 목록
INPUT_SPECS = {
    "all3":        ["circle", "triangle", "square"],  # 셋 다
    "circle_only": ["circle"],                        # 원만
    "tri_sq":      ["triangle", "square"],            # 삼각형 + 사각형 (원 없음)
}


def _draw_shape(d, shape, box):
    x0, y0, x1, y1 = box
    color = COLORS[shape]
    if shape == "circle":
        d.ellipse(box, fill=color, outline="black", width=4)
    elif shape == "square":
        d.rectangle(box, fill=color, outline="black", width=4)
    elif shape == "triangle":
        d.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=color, outline="black")


def _render(shapes, size=(768, 512)):
    """도형을 '앞뒤로 겹치게' 그린다 — shapes[0] 이 맨 앞(가장 위).

    깊이감을 주기 위해: 위치를 좌→우로 조금씩 옮기며 겹치게 하고, 뒤 도형(목록 끝)
    부터 그려서 앞 도형(목록 앞)이 위에 올라오도록 한다. → VLM 이 앞뒤(occlusion)를 판단.
    """
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    n = len(shapes)
    r = 95                                  # 도형 반경
    step = 110                              # 중심 간격(2r 보다 작아 → 겹침)
    cx0 = size[0] // 2 - (n - 1) * step // 2
    cy = size[1] // 2
    boxes = [[cx0 + i * step - r, cy - r, cx0 + i * step + r, cy + r] for i in range(n)]
    # 뒤(목록 끝)부터 그려야 앞(목록 앞) 도형이 위에 겹쳐 그려진다
    for i in reversed(range(n)):
        _draw_shape(d, shapes[i], boxes[i])
    return img


def make_sample_images(out_dir):
    """INPUT_SPECS 의 모든 조합을 그려 저장하고 {key: path} 반환."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for key, shapes in INPUT_SPECS.items():
        p = os.path.join(out_dir, f"sample_{key}.png")
        _render(shapes).save(p)
        paths[key] = p
    return paths


def make_sample_image(path):
    """단일(all3) 이미지 — main.py 기본 입력용."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _render(INPUT_SPECS["all3"]).save(path)
    return path


if __name__ == "__main__":
    import config
    paths = make_sample_images(config.ASSETS_DIR)
    for k, p in paths.items():
        print(f"{k:12s} -> {p}")
