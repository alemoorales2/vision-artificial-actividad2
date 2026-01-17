#!/usr/bin/env python3
"""
Script para regenerar figuras con las nuevas imágenes EuroSAT (I07, I08, I09).
Las imágenes satelitales ahora son:
  - I07.jpg: River (río)
  - I08.jpg: Highway (carretera)
  - I09.jpg: Industrial (puerto/industrial)
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from skimage import color, exposure
from skimage.filters import gaussian, median, sobel, laplace
from skimage.feature import canny
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops_table

# Rutas
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
IMG_DIR = BASE_DIR / "images"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("REGENERANDO FIGURAS - Nuevas imágenes EuroSAT")
print("=" * 70)
print(f"Directorio de imágenes: {IMG_DIR}")
print(f"Directorio de figuras: {FIG_DIR}")

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def load_gray(path, max_side=512):
    """Carga imagen en escala de grises y la normaliza."""
    img = Image.open(path)
    arr = np.asarray(img.convert("L"))
    x = arr.astype(np.float32) / 255.0
    x = exposure.rescale_intensity(x, in_range="image", out_range=(0, 1))
    h, w = x.shape
    scale = max(h, w) / max_side
    if scale > 1.0:
        new_h, new_w = int(h/scale), int(w/scale)
        x = resize(x, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
    return x

def spatial_filters(x):
    """Aplica filtros espaciales."""
    return {
        "gaussian_s1": gaussian(x, 1.0, preserve_range=True),
        "gaussian_s2": gaussian(x, 2.0, preserve_range=True),
        "median_r2": median(x, footprint=disk(2)),
        "median_r4": median(x, footprint=disk(4)),
        "unsharp_r1_a1": np.clip(x + (x - gaussian(x, 1.0))*1.0, 0, 1),
        "sobel": exposure.rescale_intensity(sobel(x), out_range=(0,1)),
        "laplace": exposure.rescale_intensity(np.abs(laplace(x)), out_range=(0,1)),
        "canny_s12": canny(x, 1.2).astype(np.float32),
    }

def morph_filters(x, r):
    """Aplica operaciones morfológicas."""
    se = disk(r)
    return {
        f"opening_r{r}": opening(x, se),
        f"closing_r{r}": closing(x, se),
        f"tophat_r{r}": white_tophat(x, se),
        f"erosion_r{r}": erosion(x, se),
        f"dilation_r{r}": dilation(x, se),
    }

def pipelines(x):
    """Pipelines combinados."""
    den = median(x, footprint=disk(3))
    p1 = np.clip(den + (den - gaussian(den, 1.5))*1.2, 0, 1)
    p2 = opening(closing(x, disk(3)), disk(3))
    return {
        "P1_med3+unsharp": p1,
        "P2_close+open": p2,
    }

def sigma_hat_fast(x):
    """Estimación rápida del ruido."""
    hp = x - gaussian(x, 1.0, preserve_range=True)
    med = np.median(hp)
    mad = np.median(np.abs(hp - med))
    return float(1.4826 * mad)

# ============================================================
# ENCONTRAR IMÁGENES
# ============================================================

# Buscar todas las imágenes I01-I09 (pueden ser .png o .jpg)
img_paths = []
for i in range(1, 10):
    for ext in ['.png', '.jpg', '.jpeg']:
        p = IMG_DIR / f"I{i:02d}{ext}"
        if p.exists():
            img_paths.append(p)
            break

print(f"\nImágenes encontradas: {len(img_paths)}")
for p in img_paths:
    print(f"  - {p.name}")

# Información de las imágenes (actualizado con las nuevas clases)
IMG_INFO = {
    "I01": ("Médica", "Open-I", "Radiografía tórax"),
    "I02": ("Médica", "Open-I", "Radiografía tórax"),
    "I03": ("Médica", "Open-I", "Radiografía tórax"),
    "I04": ("Industrial", "MVTec AD", "Grid (defecto)"),
    "I05": ("Industrial", "MVTec AD", "Leather (defecto)"),
    "I06": ("Industrial", "MVTec AD", "Metal nut"),
    "I07": ("Satélite", "EuroSAT RGB", "River"),
    "I08": ("Satélite", "EuroSAT RGB", "Highway"),
    "I09": ("Satélite", "EuroSAT RGB", "Industrial"),
}

# ============================================================
# 1. GRID DE ORIGINALES (3x3)
# ============================================================
print("\n[1/4] Generando originals_grid.png...")

if len(img_paths) >= 9:
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for i, (ax, p) in enumerate(zip(axes.ravel(), img_paths[:9])):
        x = load_gray(p, max_side=400)
        ax.imshow(x, cmap="gray", vmin=0, vmax=1)
        
        img_id = p.stem
        domain, source, desc = IMG_INFO.get(img_id, ("", "", ""))
        ax.set_title(f"{img_id}: {desc}\n({domain})", fontsize=9)
        ax.axis("off")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "originals_grid.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ originals_grid.png guardado")

# ============================================================
# 2. MONTAJES POR IMAGEN (especialmente I07)
# ============================================================
print("\n[2/4] Generando montajes...")

# Generar montaje para todas, pero especialmente I01, I04, I07
montage_targets = ["I01", "I04", "I07"]

for p in img_paths:
    img_id = p.stem
    if img_id not in montage_targets:
        continue
        
    print(f"  → Generando {img_id}_montage.png...")
    
    x = load_gray(p, max_side=512)
    S = spatial_filters(x)
    M = morph_filters(x, 4)
    P = pipelines(x)
    
    picks = [
        ("original", x),
        ("gaussian_s1", S["gaussian_s1"]),
        ("median_r2", S["median_r2"]),
        ("unsharp", S["unsharp_r1_a1"]),
        ("sobel", S["sobel"]),
        ("closing_r4", M["closing_r4"]),
        ("opening_r4", M["opening_r4"]),
        ("P1_med+unsharp", P["P1_med3+unsharp"]),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for ax, (lab, im) in zip(axes, picks):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(lab, fontsize=10)
        ax.axis("off")
    
    domain, source, desc = IMG_INFO.get(img_id, ("", "", ""))
    fig.suptitle(f"Comparativa de filtros: {img_id} - {desc} ({domain})", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{img_id}_montage.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {img_id}_montage.png guardado")

# ============================================================
# 3. HISTOGRAMAS
# ============================================================
print("\n[3/4] Generando histogramas...")

hist_targets = ["I01", "I04", "I07"]

for p in img_paths:
    img_id = p.stem
    if img_id not in hist_targets:
        continue
        
    print(f"  → Generando {img_id}_hist.png...")
    
    x = load_gray(p, max_side=512)
    S = spatial_filters(x)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 257)
    
    h_orig, _ = np.histogram(x.ravel(), bins=bins, density=True)
    h_gauss, _ = np.histogram(S["gaussian_s1"].ravel(), bins=bins, density=True)
    h_unsharp, _ = np.histogram(S["unsharp_r1_a1"].ravel(), bins=bins, density=True)
    
    ax.plot(bins[:-1], h_orig, label="original", linewidth=1.5, alpha=0.8)
    ax.plot(bins[:-1], h_gauss, label="gaussian_s1", linewidth=1.5, alpha=0.8)
    ax.plot(bins[:-1], h_unsharp, label="unsharp", linewidth=1.5, alpha=0.8)
    
    ax.legend(fontsize=10)
    domain, source, desc = IMG_INFO.get(img_id, ("", "", ""))
    ax.set_title(f"Histogramas normalizados: {img_id} - {desc}")
    ax.set_xlabel("Intensidad (0-1)")
    ax.set_ylabel("Densidad")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{img_id}_hist.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {img_id}_hist.png guardado")

# ============================================================
# 4. CALCULAR MÉTRICAS Y MOSTRAR RESUMEN
# ============================================================
print("\n[4/4] Calculando métricas para las imágenes satelitales...")

print("\n" + "=" * 70)
print("MÉTRICAS PARA IMÁGENES SATELITALES (I07-I09)")
print("=" * 70)

for p in img_paths:
    img_id = p.stem
    if img_id not in ["I07", "I08", "I09"]:
        continue
    
    x = load_gray(p, max_side=512)
    S = spatial_filters(x)
    
    domain, source, desc = IMG_INFO.get(img_id, ("", "", ""))
    print(f"\n{img_id}: {desc} ({domain})")
    print("-" * 50)
    
    # Calcular métricas para cada filtro
    results = []
    for name, y in S.items():
        if name in ["sobel", "laplace", "canny_s12"]:
            continue  # Skip edge detectors
        ssim_val = ssim(x, y, data_range=1.0)
        sigma_val = sigma_hat_fast(y)
        contrast = float(np.std(y))
        results.append((name, ssim_val, sigma_val, contrast))
    
    # Ordenar por SSIM
    results.sort(key=lambda t: t[1], reverse=True)
    
    print(f"{'Filtro':<20} {'SSIM':>8} {'σ_hat':>8} {'Contraste':>10}")
    print("-" * 50)
    for name, ssim_val, sigma_val, contrast in results:
        print(f"{name:<20} {ssim_val:>8.3f} {sigma_val:>8.3f} {contrast:>10.3f}")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("¡COMPLETADO!")
print("=" * 70)

print("\n📊 IMÁGENES SATELITALES ACTUALIZADAS (EuroSAT RGB):")
print("-" * 70)
print(f"{'ID':<6} {'Clase':<15} {'Descripción':<40}")
print("-" * 70)
print(f"{'I07':<6} {'River':<15} {'Río con vegetación':<40}")
print(f"{'I08':<6} {'Highway':<15} {'Carretera/autopista':<40}")
print(f"{'I09':<6} {'Industrial':<15} {'Zona industrial/puerto con barcos':<40}")

print(f"\n📁 Figuras generadas en: {FIG_DIR}")
print("\nArchivos creados/actualizados:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  • {f.name}")
