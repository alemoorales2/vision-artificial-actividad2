# Exploración de Filtros Espaciales y Morfológicos en Escenarios Reales

**Máster en Inteligencia Artificial - UNIR**  
**Asignatura:** Visión Artificial  
**Curso:** 2025-2026

## 👥 Autores

- Alejandro Morales Miranda
- Paula Polo Cabas
- Miguel Fernández Llamas

## 📝 Descripción

Este trabajo analiza el efecto de filtros espaciales y operaciones morfológicas sobre imágenes reales de tres dominios:

| Dominio | Dataset | Descripción |
|---------|---------|-------------|
| 🏥 Médico | [Open-I (Indiana CXR)](https://openi.nlm.nih.gov/) | Radiografías de tórax |
| 🏭 Industrial | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Detección de defectos superficiales |
| 🛰️ Satelital | [EuroSAT](https://github.com/phelber/EuroSAT) | Clasificación de uso del suelo |

## 🔬 Metodología

### Filtros espaciales evaluados
- **Suavizado Gaussiano** (σ = 1, 2)
- **Filtro Mediana** (r = 2, 4)
- **Unsharp Masking** (realce de contraste)
- **Detectores de bordes**: Sobel, Laplace, Canny

### Operaciones morfológicas
- Erosión, Dilatación
- Apertura, Cierre
- Top-hat, Black-hat

### Métricas de evaluación
- **SSIM**: Similitud estructural
- **σ̂**: Estimación robusta de ruido
- **Contraste RMS**
- **Densidad de bordes**

## 📁 Estructura del proyecto

```
├── main.tex              # Documento principal LaTeX
├── main.pdf              # PDF compilado
├── portada.tex           # Portada del documento
├── referencias.bib       # Bibliografía
├── generar_figuras.py    # Script para generar figuras
├── images/               # Imágenes originales (I01-I09)
├── *_montage.png         # Comparativas visuales
├── *_hist.png            # Histogramas
└── originals_grid.png    # Grid de imágenes originales
```

## 🚀 Cómo compilar

### Requisitos
- LaTeX (TeX Live, MiKTeX o similar)
- Python 3.x con: `numpy`, `matplotlib`, `scikit-image`, `Pillow`

### Compilar el documento
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Regenerar figuras
```bash
python generar_figuras.py
```

## 📊 Resultados principales

- El **filtro Gaussiano** (σ bajo) ofrece el mejor compromiso entre reducción de ruido y preservación estructural (SSIM > 0.90)
- El **filtro Mediana** preserva mejor los bordes en texturas industriales
- El **Unsharp masking** es eficaz para resaltar defectos finos, aunque amplifica ruido
- La **apertura morfológica** reduce fragmentación; el **cierre** conecta estructuras discontinuas

## 📄 Licencia

Proyecto académico - Máster en Inteligencia Artificial, UNIR.
