from PIL import Image, ImageDraw
import math

SIZES = [16, 24, 32, 48, 64, 128, 256]


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = size * 0.06
    r = size * 0.18
    # Fundo rounded rect azul escuro
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=r, fill=(22, 80, 150, 255)
    )

    cx = size / 2
    # Barras de forma de onda (5 barras verticais centralizadas)
    bar_count = 5
    heights = [0.35, 0.60, 0.80, 0.55, 0.30]
    bar_w = size * 0.07
    spacing = size * 0.11
    total_w = bar_count * bar_w + (bar_count - 1) * (spacing - bar_w)
    start_x = cx - total_w / 2

    for i in range(bar_count):
        bx = start_x + i * spacing
        bh = size * heights[i]
        by = (size - bh) / 2
        brightness = 180 + int(60 * heights[i])
        color = (brightness, brightness + 30, 255, 255)
        draw.rounded_rectangle(
            [bx, by, bx + bar_w, by + bh],
            radius=bar_w / 2,
            fill=color
        )

    # Microfone pequeno no canto inferior direito (decorativo)
    if size >= 48:
        mic_cx = size * 0.72
        mic_cy = size * 0.70
        mic_w = size * 0.10
        mic_h = size * 0.16
        draw.rounded_rectangle(
            [mic_cx - mic_w / 2, mic_cy - mic_h / 2,
             mic_cx + mic_w / 2, mic_cy + mic_h / 2],
            radius=mic_w / 2,
            fill=(255, 255, 255, 180)
        )

    return img


if __name__ == "__main__":
    frames = [make_icon(s) for s in SIZES]
    frames[0].save(
        "icon.ico",
        format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=frames[1:],
    )
    print("icon.ico gerado com sucesso!")
