# -*- coding: utf-8 -*-
"""
============================================================================
 TP Procesamiento de Imagenes - Diagnostico de Retinopatia Diabetica
============================================================================
 Clasifica imagenes de fondo de ojo en GRADO 0 (sano) vs GRADO 4
 (proliferativo) mediante procesamiento de imagenes, segmentacion de estructuras y ANALISIS DE TEXTURA.

 Etapas:
   1) Contextualizacion .... carga del dataset y mascara del campo de vision
   2) Pre-procesamiento ..... seleccion del canal de mayor contraste +
                              filtro pasa-bajos Gaussiano + realce (CLAHE)
   3) Segmentacion .......... vasos (Black-Top-Hat), disco optico,
                              exudados (White-Top-Hat) y hemorragias
   4) Diagnostico ........... criterio de TEXTURA (entropia local) con
                              umbral por Otsu (no supervisado)

 -------------------------------------------------------------------------
 CRITERIO DE DECISION
 -------------------------------------------------------------------------
 Una retina sana (Grado 0) tiene una textura mas IRREGULAR (mayor entropia
 local), porque sus vasos y detalles estan nitidos. Una Grado 4 suele ser
 mas de bajo contraste, lo que SUAVIZA la textura (menor entropia).

 Regla:   GRADO 4   si   entropia_local <= umbral
 El umbral sale de la propia distribucion
 de entropias del dataset mediante el metodo de Otsu (no supervisado), igual
 que el umbral que ya usamos para segmentar los vasos.

============================================================================
"""

import os
import glob
import random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.filters.rank import entropy as _rank_entropy
    from skimage.filters import threshold_otsu as _threshold_otsu
    from skimage.morphology import disk as _disk
    _SKIMAGE_OK = True
    _DISK5 = _disk(5)
except Exception:
    _SKIMAGE_OK = False

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
CLASES = {0: "Grado 0", 4: "Grado 4"}
RESIZE_W = 768          
N_DEMO = 5              # cuantas imagenes al azar mostrar en la demo
random.seed(7)


# ===========================================================================
#  ETAPA 1 - CONTEXTUALIZACION / CARGA
# ===========================================================================
def cargar_y_redimensionar(path):
    """Lee la imagen en RGB y la lleva a un ancho estandar."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    esc = RESIZE_W / w
    return cv2.resize(img, (RESIZE_W, int(h * esc)), interpolation=cv2.INTER_AREA)


def mascara_fov(img_rgb):
    """Mascara binaria del campo de vision (el circulo de la retina)."""
    gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gris, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return (m > 0).astype(np.uint8) * 255


# ===========================================================================
#  ETAPA 2 - PRE-PROCESAMIENTO
# ===========================================================================
def seleccionar_canal(img_rgb, fov):
    """Selecciona el canal RGB con mayor contraste dentro del FOV (mayor std)."""
    nombres = ["R", "G", "B"]
    px = fov > 0
    stds = [float(img_rgb[:, :, i][px].std()) for i in range(3)]
    idx = int(np.argmax(stds))
    return nombres[idx], img_rgb[:, :, idx].copy()


def preprocesar(canal, fov):
    """Filtro pasa-bajos Gaussiano + CLAHE (realce de contraste local)."""
    suave = cv2.GaussianBlur(canal, (5, 5), 1.2)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    realzada = clahe.apply(suave)
    realzada = cv2.bitwise_and(realzada, realzada, mask=fov)
    return suave, realzada


# ===========================================================================
#  ETAPA 3 - SEGMENTACION
# ===========================================================================
def segmentar_vasos(canal, fov):
    """Vasos: estructuras OSCURAS y finas. Black-Top-Hat + Otsu."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bth = cv2.morphologyEx(canal, cv2.MORPH_BLACKHAT, k)
    bth = cv2.bitwise_and(bth, bth, mask=fov)
    _, vasos = cv2.threshold(bth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vasos = cv2.morphologyEx(vasos, cv2.MORPH_OPEN, k2)
    return vasos


def detectar_disco_optico(canal, fov):
    """Disco optico = region mas brillante. Mascara circular para excluirlo."""
    desenfoque = cv2.GaussianBlur(canal, (0, 0), sigmaX=15)
    desenfoque = cv2.bitwise_and(desenfoque, desenfoque, mask=fov)
    _, _, _, maxloc = cv2.minMaxLoc(desenfoque)
    r = int(RESIZE_W * 0.11)
    mascara = np.zeros_like(canal)
    cv2.circle(mascara, maxloc, r, 255, -1)
    return mascara


def segmentar_exudados(canal_realzado, fov, mascara_od):
    """Exudados brillantes: manchas CLARAS y chicas. White-Top-Hat + umbral."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    wth = cv2.morphologyEx(canal_realzado, cv2.MORPH_TOPHAT, k)
    wth = cv2.bitwise_and(wth, wth, mask=fov)
    _, exud = cv2.threshold(wth, 40, 255, cv2.THRESH_BINARY)   # umbral ABSOLUTO
    exud[mascara_od > 0] = 0                                   # excluir disco optico
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    exud = cv2.morphologyEx(exud, cv2.MORPH_OPEN, k2)
    return exud


def segmentar_hemorragias(canal, fov, vasos):
    """Hemorragias: manchas OSCURAS compactas (no finas como los vasos)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bth = cv2.morphologyEx(canal, cv2.MORPH_BLACKHAT, k)
    bth = cv2.bitwise_and(bth, bth, mask=fov)
    _, oscuras = cv2.threshold(bth, 30, 255, cv2.THRESH_BINARY)  # umbral ABSOLUTO
    vd = cv2.dilate(vasos, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cand = cv2.bitwise_and(oscuras, cv2.bitwise_not(vd))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((cand > 0).astype(np.uint8))
    hemo = np.zeros_like(cand)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 8:
            continue
        aspecto = max(w_, h_) / (min(w_, h_) + 1e-6)
        relleno = area / (w_ * h_ + 1e-6)
        if aspecto <= 4.0 and relleno >= 0.30:
            hemo[lab == i] = 255
    return hemo


# ===========================================================================
#  ANALISIS DE TEXTURA (criterio de diagnostico)
# ===========================================================================
def mapa_entropia(canal_realzado, fov):
    """Mapa de entropia local (textura) con vecindario de disco r=5."""
    if not _SKIMAGE_OK:
        return None
    le = _rank_entropy(canal_realzado, _DISK5, mask=fov)
    return le.astype(np.float32)


def calcular_texturas(canal_realzado, fov, mapa_ent=None):
    """
    Descriptores de textura dentro del FOV:
      - ent_local : entropia local media (mide irregularidad de la textura).
      - glcm_homog: homogeneidad de la matriz de co-ocurrencia (GLCM).
    Grado 0 (nitida) -> mas textura (mayor entropia). Devuelve NaN sin skimage.
    """
    if not _SKIMAGE_OK:
        return dict(ent_local=float("nan"), glcm_homog=float("nan"))
    from skimage.feature import graycomatrix, graycoprops
    px = fov > 0
    if mapa_ent is None:
        mapa_ent = mapa_entropia(canal_realzado, fov)
    ent_local = float(mapa_ent[px].mean())
    q = (canal_realzado.astype(np.float32) / 256 * 32).astype(np.uint8)
    q[~px] = 0
    glcm = graycomatrix(q, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    glcm_homog = float(graycoprops(glcm, "homogeneity").mean())
    return dict(ent_local=ent_local, glcm_homog=glcm_homog)


# ===========================================================================
#  PIPELINE COMPLETO -> FEATURES
# ===========================================================================
def procesar(path, devolver_etapas=False, con_textura=False):
    """
    Corre el pipeline sobre una imagen y devuelve sus features.
    con_textura=True agrega entropia local y homogeneidad GLCM (necesarias
    para el diagnostico; un poco mas lento).
    """
    img = cargar_y_redimensionar(path)
    fov = mascara_fov(img)
    nom, canal = seleccionar_canal(img, fov)
    suave, realzada = preprocesar(canal, fov)
    vasos = segmentar_vasos(realzada, fov)
    od_mask = detectar_disco_optico(realzada, fov)
    exud = segmentar_exudados(realzada, fov, od_mask)
    hemo = segmentar_hemorragias(realzada, fov, vasos)

    area_fov = max(1, int((fov > 0).sum()))
    feats = dict(
        canal=nom,
        f_exud=(exud > 0).sum() / area_fov,
        f_hemo=(hemo > 0).sum() / area_fov,
        f_vasos=(vasos > 0).sum() / area_fov,
        n_exud=cv2.connectedComponents((exud > 0).astype(np.uint8))[0] - 1,
        n_hemo=cv2.connectedComponents((hemo > 0).astype(np.uint8))[0] - 1,
    )
    mapa_ent = None
    if con_textura:
        mapa_ent = mapa_entropia(realzada, fov)
        feats.update(calcular_texturas(realzada, fov, mapa_ent))
    if devolver_etapas:
        etapas = dict(img=img, fov=fov, canal=canal, suave=suave,
                      realzada=realzada, vasos=vasos, exud=exud,
                      hemo=hemo, mapa_ent=mapa_ent)
        return feats, etapas
    return feats


# ===========================================================================
#  ETAPA 4 - DIAGNOSTICO (criterio de textura, umbral por Otsu)
# ===========================================================================
def umbral_textura(valores):
    """
    Umbral de decision para la entropia local, obtenido por el metodo de Otsu
    sobre la distribucion de entropias del dataset.
    """
    v = np.asarray(valores, dtype=float)
    return float(_threshold_otsu(v))


def clasificar(feats, umbral):
    """GRADO 4 si la textura es poco irregular (entropia local <= umbral)."""
    return 4 if feats["ent_local"] <= umbral else 0


# ===========================================================================
#  VISUALIZACION PASO A PASO
# ===========================================================================
def overlay_lesiones(img, exud, hemo):
    """Marca exudados y hemorragias sobre la imagen original."""
    out = img.copy()
    out[exud > 0] = [255, 255, 0]
    out[hemo > 0] = [0, 255, 255]
    return out


def mostrar_proceso(path, grado_real, umbral):
    """Figura con la imagen tras cada etapa + el diagnostico por textura."""
    feats, e = procesar(path, devolver_etapas=True, con_textura=True)
    pred = clasificar(feats, umbral)
    overlay = overlay_lesiones(e["img"], e["exud"], e["hemo"])

    paneles = [
        (e["img"],      "1. Original (RGB)",                                None),
        (e["fov"],      "2. Mascara campo de vision",                       "gray"),
        (e["canal"],    f"3. Canal de mayor contraste ({feats['canal']})",  "gray"),
        (e["suave"],    "4. Filtro pasa-bajos Gaussiano",                   "gray"),
        (e["realzada"], "5. Realce de contraste (CLAHE)",                   "gray"),
        (e["vasos"],    f"6. Vasos (Black-Top-Hat)  {feats['f_vasos']*100:.1f}%", "gray"),
        (e["exud"],     f"7. Exudados brillantes  n={feats['n_exud']}",     "gray"),
        (e["mapa_ent"], f"8. Entropia local (textura)",                     "magma"),
        (overlay,       "9. Lesiones detectadas",                           None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (im, titulo, cmap) in zip(axes.ravel(), paneles):
        ax.imshow(im, cmap=cmap)
        ax.set_title(titulo, fontsize=10)
        ax.axis("off")
    ok = "CORRECTO" if pred == grado_real else "ERROR"
    fig.suptitle(
        f"{os.path.basename(path)}   |   Real: Grado {grado_real}   "
        f"Prediccion: Grado {pred}   [{ok}]\n"
        f"entropia local = {feats['ent_local']:.3f} (umbral Otsu <= {umbral:.3f})   "
        f"homogeneidad GLCM = {feats['glcm_homog']:.3f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return pred


# ===========================================================================
#  MODOS DE EJECUCION
# ===========================================================================
def listar_imagenes():
    items = []
    for grado, sub in CLASES.items():
        for p in sorted(glob.glob(os.path.join(BASE, sub, "*.jpeg"))):
            items.append((p, grado))
    return items


def demo():
    """
    Calcula el umbral de textura (Otsu) sobre TODO el dataset y luego muestra
    el proceso completo de N_DEMO imagenes al azar con su diagnostico.
    """
    if not _SKIMAGE_OK:
        print("Falta scikit-image:  pip install scikit-image")
        return
    items = listar_imagenes()
    if not items:
        print("No se encontraron imagenes en 'Grado 0' / 'Grado 4'.")
        return
    print(f"Calculando entropias del dataset ({len(items)} imagenes) para el umbral Otsu...")
    ents = []
    for i, (p, _) in enumerate(items):
        ents.append(procesar(p, con_textura=True)["ent_local"])
        print(f"  {i+1}/{len(items)}", end="\r", flush=True)
    umbral = umbral_textura(ents)
    print(f"\nUmbral de textura (Otsu, no supervisado): entropia local <= {umbral:.3f}\n")

    muestra = random.sample(items, min(N_DEMO, len(items)))
    aciertos = 0
    for path, grado in muestra:
        pred = mostrar_proceso(path, grado, umbral)
        aciertos += int(pred == grado)
        print(f"  {os.path.basename(path):16s}  real=Grado {grado}  ->  pred=Grado {pred}")
    print(f"\nAciertos en la muestra: {aciertos}/{len(muestra)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Diagnostico de Retinopatia Diabetica (Grado 0 vs 4)").parse_args()
    demo()
