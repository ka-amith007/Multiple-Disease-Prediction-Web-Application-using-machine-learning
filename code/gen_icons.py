from PIL import Image, ImageOps
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
SOURCE = os.path.join(ASSETS_DIR, 'logo.png')
OUT_192 = os.path.join(ASSETS_DIR, 'app-icon-192.png')
OUT_512 = os.path.join(ASSETS_DIR, 'app-icon-512.png')

# Background color (deep blue) for padding area.
BG = (14, 78, 166, 255)  # #0E4EA6

def make_icon(target_size):
    if not os.path.exists(SOURCE):
        raise FileNotFoundError(f"Logo not found at {SOURCE}")

    img = Image.open(SOURCE).convert('RGBA')

    # Ensure square by padding shorter side
    max_side = max(img.width, img.height)
    square_bg = Image.new('RGBA', (max_side, max_side), BG)
    x = (max_side - img.width) // 2
    y = (max_side - img.height) // 2
    square_bg.paste(img, (x, y), img)

    # Add safe-area padding (~10%) by shrinking content then centering on target canvas
    pad_ratio = 0.88  # content scale to leave ~12% padding
    content_size = int(target_size * pad_ratio)
    content = square_bg.resize((content_size, content_size), Image.LANCZOS)

    canvas = Image.new('RGBA', (target_size, target_size), BG)
    cx = (target_size - content_size) // 2
    cy = (target_size - content_size) // 2
    canvas.paste(content, (cx, cy), content)

    # Optional: slight rounding for aesthetics (kept square for compatibility)
    return canvas

def main():
    icon192 = make_icon(192)
    icon192.save(OUT_192, format='PNG')
    icon512 = make_icon(512)
    icon512.save(OUT_512, format='PNG')
    print(f"Generated: {OUT_192}\nGenerated: {OUT_512}")

if __name__ == '__main__':
    main()
