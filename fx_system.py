from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2
import math
import random


# ========================
# FX FUNKTIONEN
# ========================

def fx_filmgrain(img, intensity=0.12):
    """Analoger Filmgrain"""
    noise = np.random.normal(0, 255 * intensity, img.size[::-1] + (1,))
    noise = np.repeat(noise, 3, axis=2)
    noisy = np.clip(np.array(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy).filter(ImageFilter.GaussianBlur(0.3))


def fx_bw(img, contrast=1.3):
    """Schwarzweiß-Kontrast"""
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def fx_sepia(img, tone="#704214"):
    """Sepia-Tonung"""
    sepia = ImageOps.colorize(ImageOps.grayscale(img), tone, "#C0A080")
    return sepia


def fx_vignette(img, intensity=0.6, blur_radius=60):
    """Vignette mit weichem Rand (Gaussian Blur)"""
    width, height = img.size
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    radius = np.sqrt(xv**2 + yv**2)
    mask = np.clip(1 - radius, 0, 1)
    mask = np.power(mask, 3) * intensity  # stärkerer Falloff
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    vignette = np.uint8(255 * mask[..., np.newaxis])
    img_np = np.array(img, dtype=np.uint8)
    darkened = np.uint8(img_np * vignette / 255 + img_np * (1 - intensity / 2))
    return Image.fromarray(darkened)


def fx_softblur(img, radius=2.0):
    """Weicher globaler Blur (träumerisch)"""
    return img.filter(ImageFilter.GaussianBlur(radius))


def fx_glow(img, strength=1.3, blur_radius=10):
    """Sanfter Lichtglow (Bloom)"""
    glow = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return Image.blend(img, glow, alpha=min(strength * 0.3, 0.6))


def fx_scanlines(img, intensity=0.15):
    """CRT-Scanlines (Retro-Monitor)"""
    arr = np.array(img)
    h, w, _ = arr.shape
    for y in range(0, h, 2):
        arr[y:y+1, :, :] = (arr[y:y+1, :, :] * (1 - intensity)).astype(np.uint8)
    return Image.fromarray(arr)


def fx_chromatic(img, shift=3):
    """Chromatische Aberration (RGB-Shift)"""
    arr = np.array(img)
    b, g, r = cv2.split(arr)
    rows, cols = b.shape
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    r_shifted = cv2.warpAffine(r, M, (cols, rows))
    M = np.float32([[1, 0, -shift], [0, 1, 0]])
    b_shifted = cv2.warpAffine(b, M, (cols, rows))
    result = cv2.merge((b_shifted, g, r_shifted))
    return Image.fromarray(result)


def fx_posterize(img, bits=4):
    """Reduzierte Farbpalette (stilisierter Retro-Look)"""
    return ImageOps.posterize(img, bits)


# ========================
# FX CONTROLLER
# ========================

FX_MAP = {
    "filmgrain": fx_filmgrain,
    "bw": fx_bw,
    "sepia": fx_sepia,
    "vignette": fx_vignette,
    "softblur": fx_softblur,
    "glow": fx_glow,
    "scanlines": fx_scanlines,
    "chromatic": fx_chromatic,
    "posterize": fx_posterize
}


def apply_fx(image, fx_string="none"):
    """
    Kombiniert mehrere FX-Effekte.
    Syntax:
      "filmgrain+vignette+glow"
      "filmgrain:0.2+vignette:0.8"
    """
    if not fx_string or fx_string.lower() == "none":
        return image

    fx_list = fx_string.split("+")
    for fx in fx_list:
        if ":" in fx:
            name, value = fx.split(":")
            try:
                value = float(value)
            except:
                value = None
        else:
            name, value = fx, None

        name = name.strip().lower()
        func = FX_MAP.get(name)
        if func:
            print(f"🎬 FX → {name} ({value if value else 'default'})")
            try:
                if value:
                    image = func(image, value)
                else:
                    image = func(image)
            except Exception as e:
                print(f"⚠️ Fehler bei FX {name}: {e}")
        else:
            print(f"⚠️ Unbekannter FX: {name}")

    return image
