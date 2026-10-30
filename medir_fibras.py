#!/usr/bin/env python3
"""
Medición automática del diámetro de fibras en imágenes de microscopía electrónica (SEM).

Uso:
    python medir_fibras.py IMAGEN_O_CARPETA [IMAGEN_O_CARPETA ...] [opciones]

Ejemplos:
    python medir_fibras.py amostra_533.tif
    python medir_fibras.py fotos/ --aumentos 5 10 --salida resultados/
    python medir_fibras.py amostra_533.tif amostra_534.tif -n 150
    python medir_fibras.py otra.png --escala-um 2       # imagen sin metadatos

Qué hace en cada imagen:
  1. Separa la barra de información inferior del microscopio.
  2. Escala µm/píxel: la lee de los metadatos del TIFF de ZEISS ("Pixel Size").
     Si la imagen no los tiene, mide la barra verde y usa el valor de --escala-um.
  3. Detecta las fibras como crestas brillantes (filtro de Hessiano multiescala).
  4. En puntos al azar, separados entre sí, mide el diámetro de borde a borde,
     siempre en perpendicular a la fibra: corrige la orientación siguiendo el eje
     de la fibra y se queda con el ángulo de ancho mínimo. Descarta cruces,
     extremos y tramos irregulares.
  5. Sólo acepta fibras de primer plano:
       - no tapadas: siguiendo la fibra a lo largo de su eje, sus bordes se ven
         continuos; si otra fibra pasa por encima, quedan interrumpidos (--continuidad);
       - enfocadas y claras: brillo, nitidez de borde y contraste comparables a los
         de las mejores fibras de la imagen (--capa, --nitidez).
     Las fibras de detrás salen desenfocadas y su medida estaría distorsionada.

Archivos que genera (en --salida, o junto a cada imagen):
    <nombre>_medidas.png      imagen (x2) con cada medida señalada y numerada
    <nombre>_histograma.png   histograma de diámetros (µm)
    <nombre>_deteccion.png    control: aceptadas (verde) y descartadas por ser de detrás (rojo)
    <nombre>_medidas.csv      tabla con todas las medidas
  y, si se analiza más de una imagen:
    conjunto_histograma.png   histograma de todas las medidas juntas
    conjunto_medidas.csv      todas las medidas, con la imagen de la que vienen
    conjunto_resumen.csv      una fila por imagen (aumento, n, media, desviación, mediana)

Requisitos:  pip install numpy scipy scikit-image opencv-python matplotlib pillow
"""

import argparse
import csv
import os
import re
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology


# --------------------------------------------------------------------------- #
# Lectura y escala
# --------------------------------------------------------------------------- #
def leer_imagen(ruta):
    """Devuelve (imagen RGB uint8, imagen gris uint8). 'ruta' puede ser una ruta o un archivo en memoria."""
    img = Image.open(ruta)
    rgb = np.array(img.convert("RGB"))
    gris = np.array(img.convert("L"))
    return rgb, gris


def separar_barra_info(gris):
    """Encuentra la fila donde empieza la barra de información del microscopio.

    En las imágenes ZEISS hay una línea negra seguida de una franja casi blanca.
    Si no se encuentra, se usa la imagen completa.
    """
    h = gris.shape[0]
    medias = gris.mean(axis=1)
    for y in range(int(h * 0.6), h - 1):
        if medias[y] < 15 and medias[y + 1] > 200:
            return y
    return h


def escala_metadatos(ruta):
    """Lee el tamaño de píxel guardado por el microscopio (TIFF de ZEISS, etiqueta 34118).

    Devuelve µm/píxel o None si la imagen no lo tiene.
    """
    try:
        img = Image.open(ruta)
        texto = str(img.tag_v2.get(34118, ""))
    except Exception:
        return None
    m = re.search(r"Pixel Size\s*=\s*([\d.]+)\s*(pm|nm|µm|um|mm)", texto)
    if not m:
        return None
    factor = {"pm": 1e-6, "nm": 1e-3, "µm": 1.0, "um": 1.0, "mm": 1e3}[m.group(2)]
    return float(m.group(1)) * factor


def aumento_metadatos(ruta):
    """Aumento del microscopio (en X) leído de los metadatos ZEISS, o None."""
    try:
        texto = str(Image.open(ruta).tag_v2.get(34118, ""))
    except Exception:
        return None
    m = re.search(r"Mag\s*=\s*([\d.]+)\s*(K?)\s*X", texto)
    if not m:
        return None
    return float(m.group(1)) * (1000 if m.group(2) else 1)


def detectar_barra_escala(rgb, y_inicio):
    """Busca la barra verde de escala debajo de y_inicio. Devuelve su longitud en píxeles."""
    zona = rgb[y_inicio:].astype(int)
    r, g, b = zona[..., 0], zona[..., 1], zona[..., 2]
    verde = (g > 150) & (r < 100) & (b < 100)
    if verde.sum() == 0:
        return None
    # La barra horizontal es la fila con más píxeles verdes
    fila = np.argmax(verde.sum(axis=1))
    xs = np.nonzero(verde[fila])[0]
    return int(xs.max() - xs.min() + 1)


# --------------------------------------------------------------------------- #
# Detección de fibras (crestas brillantes)
# --------------------------------------------------------------------------- #
SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.5, 7.5, 10.0, 13.0, 17.0, 22.0, 29.0, 38.0)


def normalizar(img):
    lo, hi = np.percentile(img, (0.5, 99.5))
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def preparar(gris):
    """Quita ruido y corrige la iluminación.

    Devuelve dos imágenes float en [0, 1]:
      - deteccion: con el fondo local restado, para encontrar las fibras.
      - medida: sólo corregida de iluminación general; conserva el brillo real de
        cada fibra (las de delante son más claras) y sus bordes sin alterar.
    """
    suave = cv2.fastNlMeansDenoising(gris, None, h=10, templateWindowSize=7, searchWindowSize=21)
    img = suave.astype(np.float32)
    deteccion = normalizar(img - cv2.GaussianBlur(img, (0, 0), sigmaX=60))
    medida = normalizar(img - cv2.GaussianBlur(img, (0, 0), sigmaX=300) + img.mean())
    return deteccion, medida


def mapa_crestas(img):
    """Filtro de Hessiano multiescala para fibras brillantes sobre fondo oscuro.

    Devuelve (respuesta, sigma óptima, nx, ny) donde (nx, ny) es la dirección
    perpendicular a la fibra en cada píxel.
    """
    resp = np.zeros(img.shape, np.float32)
    sig = np.zeros(img.shape, np.float32)
    nx = np.zeros(img.shape, np.float32)
    ny = np.zeros(img.shape, np.float32)
    for s in SIGMAS:
        g = cv2.GaussianBlur(img, (0, 0), sigmaX=s)
        hyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        hxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        hxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4.0
        tmp = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
        l2 = 0.5 * (hxx + hyy - tmp)  # autovalor más negativo
        l1 = 0.5 * (hxx + hyy + tmp)
        r = (s ** 2) * np.maximum(0, -l2 - np.abs(l1))  # cresta brillante y alargada
        # Dirección a través de la fibra (autovector de l2), en forma angular para que
        # no degenere en fibras horizontales o verticales
        theta = 0.5 * np.arctan2(2 * hxy, hxx - hyy)  # autovector de l1 = a lo largo de la fibra
        mejor = r > resp
        resp[mejor] = r[mejor]
        sig[mejor] = s
        nx[mejor] = -np.sin(theta)[mejor]
        ny[mejor] = np.cos(theta)[mejor]
    return resp, sig, nx, ny


def lineas_centrales(resp, sensibilidad):
    """Umbraliza la respuesta de cresta y la reduce a líneas de 1 píxel."""
    umbral = filters.threshold_otsu(resp) * (1.0 - sensibilidad)
    crestas = resp > umbral
    crestas = morphology.remove_small_objects(crestas, max_size=40)
    return morphology.skeletonize(crestas)


# --------------------------------------------------------------------------- #
# Medida por perfil de intensidad (de borde a borde)
# --------------------------------------------------------------------------- #
def perfil(img, x, y, nx, ny, L, ancho_banda):
    """Perfil de intensidad perpendicular a la fibra, promediado en una banda paralela."""
    t = np.arange(-L, L + 1e-6, 0.25)
    dx, dy = -ny, nx  # dirección a lo largo de la fibra
    offs = np.linspace(-ancho_banda, ancho_banda, 5)
    xs = x + t[None, :] * nx + offs[:, None] * dx
    ys = y + t[None, :] * ny + offs[:, None] * dy
    vals = ndi.map_coordinates(img, [ys, xs], order=1, mode="nearest")
    return t, vals.mean(0)


def bordes(t, p, suavizado):
    """Localiza los dos bordes de la fibra en un perfil centrado (t=0 dentro de la fibra).

    A cada lado, saliendo desde el centro, el borde es el primer máximo de la caída
    de intensidad que supere el 40 % de la mayor caída de ese lado. Así se mide de
    borde a borde aunque alrededor haya otras fibras en lugar de fondo oscuro.
    Devuelve (ancho, t_izq, t_der, contraste, pendiente) o None. 'pendiente' es la
    caída de intensidad por píxel en el borde más débil: alta = borde nítido (enfocado).
    """
    p = ndi.gaussian_filter1d(p, max(1.0, suavizado / 0.25))
    g = np.gradient(p, 0.25)
    c = len(t) // 2
    res = []
    for lado in (-1, 1):
        d = g[c::-1] if lado < 0 else -g[c:]  # caída de intensidad hacia fuera
        dmax = d.max()
        if dmax <= 0:
            return None
        k = next((j for j in range(1, len(d) - 1)
                  if d[j] >= 0.4 * dmax and d[j] >= d[j - 1] and d[j] >= d[j + 1]), None)
        if k is None or k >= len(d) - 3:  # borde fuera de la ventana
            return None
        den = d[k - 1] - 2 * d[k] + d[k + 1]
        off = 0.5 * (d[k - 1] - d[k + 1]) / den if den != 0 else 0.0
        res.append((lado * (k + off) * 0.25, d[k], k))
    (t_izq, g_izq, k_izq), (t_der, g_der, k_der) = res
    ancho = t_der - t_izq
    if ancho <= 0:
        return None
    # Contraste: interior de la fibra frente al exterior cercano a cada borde
    interior = p[c - k_izq:c + k_der + 1].mean()
    fuera = max(1, int(0.5 * ancho / 0.25) + 8)
    ext_izq = p[max(0, c - k_izq - fuera):c - k_izq + 1].min()
    ext_der = p[c + k_der:c + k_der + fuera + 1].min()
    contraste = interior - max(ext_izq, ext_der)
    return ancho, t_izq, t_der, contraste, min(g_izq, g_der)


def medir_en(img, x, y, nx, ny, L, banda, suavizado):
    """Mide de borde a borde en (x, y) a lo largo de la normal (nx, ny). Devuelve dict o None."""
    t, p = perfil(img, x, y, nx, ny, L, banda)
    r = bordes(t, p, suavizado)
    if r is None:
        return None
    ancho, t1, t2, contraste, pendiente = r
    tc = (t1 + t2) / 2
    interior = p[(t >= t1) & (t <= t2)]
    return dict(ancho=ancho, xc=x + tc * nx, yc=y + tc * ny, desvio=tc, contraste=contraste,
                pendiente=pendiente, brillo=float(interior.mean()) if len(interior) else 0.0)


def girar(nx, ny, grados):
    a = np.deg2rad(grados)
    c, s_ = np.cos(a), np.sin(a)
    return nx * c - ny * s_, nx * s_ + ny * c


def medir_punto(img, x, y, nx, ny, sigma):
    """Mide el diámetro en (x, y) perpendicular a la fibra.

    1. Localiza los bordes y el eje de la fibra.
    2. Corrige la orientación siguiendo el eje a ambos lados del punto
       (así la medida no queda en diagonal).
    3. Ajuste fino: entre ángulos cercanos se queda con el de ancho mínimo,
       que es el perpendicular.
    4. Comprueba que la fibra sea recta y de grosor constante alrededor.
    """
    L = max(10.0, 3.5 * sigma + 8)
    banda = max(1.0, 0.5 * sigma)
    suav = max(0.5, 0.08 * sigma)
    m = medir_en(img, x, y, nx, ny, L, banda, suav)
    if m is None or m["ancho"] < 1.5 or m["contraste"] < 0.08:
        return None
    nx0, ny0 = nx, ny
    xc, yc = m["xc"], m["yc"]

    # 2. Corregir la dirección con dos puntos del eje, a un lado y otro
    for _ in range(3):
        paso = max(6.0, 1.5 * m["ancho"])
        dx, dy = -ny, nx
        eje = []
        for sgn in (-1, 1):
            mm = medir_en(img, xc + sgn * paso * dx, yc + sgn * paso * dy, nx, ny, L, banda, suav)
            if mm is None:
                return None
            eje.append((mm["xc"], mm["yc"]))
        ddx, ddy = eje[1][0] - eje[0][0], eje[1][1] - eje[0][1]
        norma = np.hypot(ddx, ddy)
        nnx, nny = ddy / norma, -ddx / norma
        if nnx * nx + nny * ny < 0:
            nnx, nny = -nnx, -nny
        cambio = np.degrees(np.arccos(np.clip(nnx * nx + nny * ny, -1, 1)))
        nx, ny = nnx, nny
        m = medir_en(img, xc, yc, nx, ny, L, banda, suav)
        if m is None:
            return None
        xc, yc = m["xc"], m["yc"]
        if cambio < 1.0:
            break
    # Si la dirección se ha ido muy lejos de la inicial, probablemente es un cruce
    if abs(nx * nx0 + ny * ny0) < np.cos(np.deg2rad(35)):
        return None

    # 3. Ajuste fino: ángulo de ancho mínimo
    mejor = (m["ancho"], nx, ny, m)
    for ang in (-8, -6, -4, -2, 2, 4, 6, 8):
        gx, gy = girar(nx, ny, ang)
        mm = medir_en(img, xc, yc, gx, gy, L, banda, suav)
        if mm is not None and mm["ancho"] < mejor[0]:
            mejor = (mm["ancho"], gx, gy, mm)
    _, nx, ny, m = mejor
    xc, yc = m["xc"], m["yc"]

    # 4. Comprobar a lo largo de la fibra (descarta cruces, ramas y extremos)
    ancho = m["ancho"]
    paso = max(5.0, 1.2 * ancho)
    dx, dy = -ny, nx
    anchos = [ancho]
    for sgn in (-1, 1):
        mm = medir_en(img, xc + sgn * paso * dx, yc + sgn * paso * dy, nx, ny, L, banda, suav)
        if mm is None or abs(mm["desvio"]) > 0.2 * ancho + 0.75:  # la fibra no sigue recta
            return None
        anchos.append(mm["ancho"])
    anchos = np.array(anchos)
    if anchos.max() - anchos.min() > 0.2 * np.median(anchos) + 0.75:
        return None
    a = float(np.median(anchos))
    p1 = (xc - a / 2 * nx, yc - a / 2 * ny)
    p2 = (xc + a / 2 * nx, yc + a / 2 * ny)
    return dict(x=xc, y=yc, ancho_px=a, p1=p1, p2=p2, nx=nx, ny=ny, sigma=sigma,
                contraste=m["contraste"])


def profundidad(img, m):
    """Indicadores de si la fibra está delante: brillo interior y nitidez de los bordes.

    La nitidez es la mayor pendiente de intensidad en cada borde, con un suavizado fijo
    de 1 px, igual para fibras finas y gruesas. Una fibra de detrás está desenfocada:
    sus bordes son más suaves y es más oscura.
    """
    a = m["ancho_px"]
    margen = max(4.0, 0.2 * a)
    t, p = perfil(img, m["x"], m["y"], m["nx"], m["ny"], a / 2 + margen + 2, 1.0)
    p = ndi.gaussian_filter1d(p, 4)  # 1 px
    g = np.abs(np.gradient(p, 0.25))
    pend = []
    for borde in (-a / 2, a / 2):
        zona = np.abs(t - borde) <= margen
        pend.append(g[zona].max())
    interior = p[np.abs(t) <= max(1.0, 0.3 * a)]
    m["nitidez"] = float(np.mean(pend))
    m["brillo"] = float(interior.mean())


def fuerza_bordes(img, x, y, nx, ny, a):
    """Caída de intensidad en cada uno de los dos bordes esperados (a ±a/2 del eje)."""
    margen = max(2.0, 0.2 * a)
    t, p = perfil(img, x, y, nx, ny, a / 2 + margen + 2, 1.0)
    p = ndi.gaussian_filter1d(p, 4)  # 1 px
    g = np.gradient(p, 0.25)
    izq = g[np.abs(t + a / 2) <= margen].max()     # sube al entrar en la fibra
    der = -g[np.abs(t - a / 2) <= margen].min()    # baja al salir
    return izq, der


def continuidad(img, m):
    """Fracción del recorrido en que la fibra se ve entera (sin que nada la tape).

    Se sigue la fibra a lo largo de su eje hacia ambos lados y, cada medio diámetro,
    se comprueba que sus dos bordes siguen visibles en su sitio. Si otra fibra pasa
    por encima, los bordes desaparecen en el cruce; si es la fibra la que pasa por
    encima de otra, sus bordes se siguen viendo. Así, una fibra de delante se ve
    continua en todo el recorrido y una de detrás queda interrumpida.
    """
    a = m["ancho_px"]
    sigma = m["sigma"]
    L = max(10.0, 3.5 * sigma + 8, 0.5 * a + 10)
    banda = max(1.0, 0.5 * sigma)
    suav = max(0.5, 0.08 * sigma)
    paso = 2.0  # fino, para no saltarse fibras finas que crucen por encima
    recorrido = max(60.0, 4.0 * a)
    ref = min(fuerza_bordes(img, m["x"], m["y"], m["nx"], m["ny"], a))
    if ref <= 0:
        return 0.0, 99
    h, w = img.shape
    validos = total = huecos = 0
    for sgn in (-1, 1):
        x, y, nx, ny = m["x"], m["y"], m["nx"], m["ny"]
        ax, ay = x, y  # último punto del eje confirmado (para seguir la curvatura)
        seguidos = 0
        for k in range(int(recorrido / paso)):
            x, y = x - ny * sgn * paso, y + nx * sgn * paso
            if not (a <= x < w - a and a <= y < h - a):
                break
            # Recentrar sobre el eje y actualizar la dirección, por si la fibra se curva
            if k % 3 == 2:
                mm = medir_en(img, x, y, nx, ny, L, banda, suav)
                if mm is not None and abs(mm["ancho"] - a) < 0.3 * a + 1 and abs(mm["desvio"]) < 0.3 * a + 1:
                    x, y = mm["xc"], mm["yc"]
                    dx, dy = (x - ax) * sgn, (y - ay) * sgn
                    d = np.hypot(dx, dy)
                    if d > 0.5 * a:
                        nnx, nny = dy / d, -dx / d
                        if nnx * nx + nny * ny < 0:
                            nnx, nny = -nnx, -nny
                        nx, ny = 0.5 * (nx + nnx), 0.5 * (ny + nny)
                        nn = np.hypot(nx, ny)
                        nx, ny = nx / nn, ny / nn
                        ax, ay = x, y
            total += 1
            if min(fuerza_bordes(img, x, y, nx, ny, a)) >= 0.3 * ref:
                validos += 1
                seguidos = 0
            else:
                seguidos += 1
                if seguidos == 2:  # interrupción real (no un píxel de ruido)
                    huecos += 1
    return (validos / total if total else 0.0), huecos


def medir(img, img_medida, n_medidas, sep_min, capa=0.7, sensibilidad=0.0, nitidez=0.35,
          min_continuidad=0.7, max_huecos=2, semilla=0):
    """Toma medidas sólo en las fibras de primer plano.

    Una fibra se considera de delante si se ve continua a lo largo de su eje, sin que
    otra fibra la tape (--continuidad), y además es de las más brillantes (--capa),
    con bordes nítidos (--nitidez) y buen contraste. Las de detrás quedan tapadas en
    los cruces, salen más oscuras y desenfocadas, y su medida estaría distorsionada.
    """
    resp, sig, nx, ny = mapa_crestas(img)
    centros = lineas_centrales(resp, sensibilidad)
    ys, xs = np.nonzero(centros)
    orden = np.random.default_rng(semilla).permutation(len(ys))
    h, w = img.shape
    ocupado = np.zeros(img.shape, np.uint8)   # zonas ya medidas (separación mínima)
    probado = np.zeros(img.shape, np.uint8)   # zonas ya probadas y descartadas

    def siguiente_candidato(it):
        """Mide el siguiente punto válido del esqueleto (o None si no quedan)."""
        for i in it:
            y, x = int(ys[i]), int(xs[i])
            if ocupado[y, x] or probado[y, x]:
                continue
            m = medir_punto(img_medida, x, y, nx[y, x], ny[y, x], sig[y, x])
            cv2.circle(probado, (x, y), 3, 1, -1)
            if m is None or m["ancho_px"] < 0.8 * sig[y, x]:  # ancho absurdo para la escala
                continue
            margen = m["ancho_px"]
            if not (margen <= m["x"] < w - margen and margen <= m["y"] < h - margen):
                continue
            if ocupado[int(m["y"]), int(m["x"])]:
                continue
            profundidad(img_medida, m)
            return m
        return None

    def es_de_delante(m, ref):
        rapido = (m["brillo"] >= ref["brillo"] and m["nitidez"] >= nitidez * ref["nitidez"]
                  and m["contraste"] >= 0.4 * ref["contraste"])
        if not rapido:
            return False
        if "continuidad" not in m:
            m["continuidad"], m["huecos"] = continuidad(img_medida, m)
        return m["continuidad"] >= min_continuidad and m["huecos"] <= max_huecos

    # 1. Calibración: con una muestra de puntos se fija qué es "brillante", "nítido" y
    #    "contrastado" en esta imagen (comparado con las mejores fibras)
    it = iter(orden)
    muestra = []
    while len(muestra) < 200:
        m = siguiente_candidato(it)
        if m is None:
            break
        muestra.append(m)
    if not muestra:
        return [], []
    ref = dict(brillo=np.quantile([m["brillo"] for m in muestra], 1.0 - capa),
               nitidez=np.quantile([m["nitidez"] for m in muestra], 0.9),
               contraste=np.quantile([m["contraste"] for m in muestra], 0.9))

    # 2. Selección: primer plano = no tapada por ninguna otra fibra (continuidad) y
    #    brillante, nítida y bien contrastada. Se sigue buscando por toda la imagen
    #    hasta reunir las medidas pedidas.
    candidatos, medidas = [], []

    def evaluar(m):
        m["delante"] = es_de_delante(m, ref)
        if m["delante"] and ocupado[int(m["y"]), int(m["x"])]:
            m["delante"] = False
        candidatos.append(m)
        if m["delante"]:
            medidas.append(m)
            cv2.circle(ocupado, (int(m["x"]), int(m["y"])), int(max(sep_min, 2 * m["ancho_px"])), 1, -1)

    for m in muestra:
        if len(medidas) < n_medidas:
            evaluar(m)
    while len(medidas) < n_medidas:
        m = siguiente_candidato(it)
        if m is None:
            break
        evaluar(m)
    return medidas, candidatos


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #
def imagen_medidas(rgb, medidas, factor=2):
    """Imagen (ampliada x2 para que se lea bien) con cada medida como un segmento rojo
    con topes, numerado."""
    img = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    f = lambda p: (int(round(p[0] * factor)), int(round(p[1] * factor)))
    for k, m in enumerate(medidas, 1):
        p1, p2 = np.array(m["p1"]), np.array(m["p2"])
        d = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)
        tope = np.array([-d[1], d[0]]) * max(3, 0.4 * m["ancho_px"])
        cv2.line(img, f(p1), f(p2), (255, 40, 40), 2, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.line(img, f(p - tope), f(p + tope), (255, 40, 40), 2, cv2.LINE_AA)
        # Número fuera de la marca, en la prolongación de la medida
        texto = str(k)
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        centro = np.array(p2) * factor + d * (6 + 0.6 * np.hypot(tw, th))
        pos = (int(centro[0] - tw / 2), int(centro[1] + th / 2))
        for color, grosor in (((0, 0, 0), 4), ((255, 255, 0), 1)):
            cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, grosor, cv2.LINE_AA)
    return img


def imagen_deteccion(rgb, candidatos, factor=2):
    """Imagen de control: medidas válidas en fibras de delante (verde) y descartadas
    por pertenecer a fibras de detrás (rojo)."""
    img = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    f = lambda p: (int(round(p[0] * factor)), int(round(p[1] * factor)))
    # Las descartadas se dibujan espaciadas para que la imagen no se sature
    libre = np.ones(rgb.shape[:2], np.uint8)
    for m in sorted(candidatos, key=lambda m: not m["delante"]):
        x, y = int(m["x"]), int(m["y"])
        if not m["delante"] and not libre[y, x]:
            continue
        cv2.circle(libre, (x, y), int(max(12, m["ancho_px"])), 0, -1)
        color = (60, 255, 60) if m["delante"] else (255, 60, 60)
        cv2.line(img, f(m["p1"]), f(m["p2"]), color, 2, cv2.LINE_AA)
    return img


def estadisticas(diam):
    """Resumen estadístico de una lista de diámetros (µm)."""
    diam = np.asarray(diam, float)
    desv = diam.std(ddof=1) if len(diam) > 1 else 0.0
    return dict(n=len(diam), media=diam.mean(), desv=desv,
                cv=100 * desv / diam.mean(),  # coeficiente de variación, %
                mediana=float(np.median(diam)), minimo=diam.min(), maximo=diam.max())


def figura_histograma(diam, titulo):
    """Figura de matplotlib con el histograma de diámetros, la media, ± desv. y el CV."""
    e = estadisticas(diam)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    bins = np.histogram_bin_edges(diam, bins="auto")
    ax.hist(diam, bins=bins, color="#3b6fb6", edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(e["media"], color="#d1495b", lw=1.8, zorder=3, label=f"media = {e['media']:.3f} µm")
    ax.axvspan(e["media"] - e["desv"], e["media"] + e["desv"], color="#d1495b", alpha=0.10,
               zorder=1, label=f"± desv. est. = {e['desv']:.3f} µm")
    # Coeficiente de variación (sólo como texto en la leyenda)
    ax.plot([], [], " ", label=f"CV = {e['cv']:.1f} %")
    ax.set_xlabel("Diámetro (µm)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{titulo}  (n = {e['n']})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def dibujar_medidas(rgb, medidas, ruta, factor=2):
    Image.fromarray(imagen_medidas(rgb, medidas, factor)).save(ruta)


def guardar_deteccion(rgb, candidatos, ruta, factor=2):
    Image.fromarray(imagen_deteccion(rgb, candidatos, factor)).save(ruta)


def dibujar_histograma(diam, titulo, ruta):
    fig = figura_histograma(diam, titulo)
    fig.savefig(ruta)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Análisis completo de una imagen
# --------------------------------------------------------------------------- #
def analizar(rgb, gris, um_px, n_medidas=100, separacion=25, capa=0.7, sensibilidad=0.0,
             nitidez=0.35, continuidad=0.7, max_huecos=2, semilla=0):
    """Analiza una imagen ya leída. No escribe nada en disco.

    Devuelve un dict con:
      medidas      lista de medidas aceptadas (cada una con 'diametro_um')
      candidatos   todas las medidas probadas (con 'delante' True/False)
      atipicas     número de medidas descartadas por atípicas
      rgb_util     la imagen sin la barra de información
    """
    y_info = separar_barra_info(gris)
    rgb_util, gris_util = rgb[:y_info], gris[:y_info]
    img, img_medida = preparar(gris_util)
    medidas, candidatos = medir(img, img_medida, n_medidas, separacion,
                                capa=capa, sensibilidad=sensibilidad, nitidez=nitidez,
                                min_continuidad=continuidad, max_huecos=max_huecos,
                                semilla=semilla)
    # Valores atípicos (p. ej. zonas desenfocadas o grumos tomados por fibra)
    n_atipicas = 0
    anchos = np.array([m["ancho_px"] for m in medidas])
    if len(anchos) >= 8:
        q1, q3 = np.percentile(anchos, (25, 75))
        limite = q3 + 3 * (q3 - q1)
        atipicas = [m for m in medidas if m["ancho_px"] > limite]
        n_atipicas = len(atipicas)
        medidas = [m for m in medidas if m["ancho_px"] <= limite]
        for m in atipicas:
            m["delante"] = False
    for m in medidas:
        m["diametro_um"] = m["ancho_px"] * um_px
    return dict(medidas=medidas, candidatos=candidatos, atipicas=n_atipicas, rgb_util=rgb_util)


# --------------------------------------------------------------------------- #
# Uso desde la línea de comandos
# --------------------------------------------------------------------------- #
EXTENSIONES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def listar_imagenes(entradas):
    rutas = []
    for e in entradas:
        if os.path.isdir(e):
            rutas += sorted(os.path.join(e, f) for f in os.listdir(e)
                            if f.lower().endswith(EXTENSIONES))
        elif os.path.isfile(e):
            rutas.append(e)
        else:
            print(f"Aviso: no existe {e}")
    return rutas


def escala_de(ruta, rgb, y_info, args):
    """µm/píxel de la imagen y de dónde sale, o (None, motivo)."""
    if args.um_por_pixel:
        return args.um_por_pixel, "fijada a mano"
    um_meta = escala_metadatos(ruta)
    if args.escala_um is None and um_meta:
        return um_meta, "metadatos del microscopio"
    largo = detectar_barra_escala(rgb, y_info)
    if not largo:
        return None, "no se encontró la barra de escala verde (usa --um-por-pixel)"
    if args.escala_um is None:
        return None, (f"sin tamaño de píxel en los metadatos; barra verde de {largo} px: "
                      f"indica cuánto mide con --escala-um")
    return args.escala_um / largo, f"barra verde de {largo} px = {args.escala_um} µm"


def analizar_imagen(ruta, args):
    """Analiza una imagen, guarda sus resultados y devuelve (diámetros en µm, resumen)."""
    base = os.path.splitext(os.path.basename(ruta))[0]
    rgb, gris = leer_imagen(ruta)
    y_info = separar_barra_info(gris)
    um_px, origen = escala_de(ruta, rgb, y_info, args)
    if um_px is None:
        print(f"  {base}: {origen}. Se omite.")
        return None
    print(f"  Escala: {um_px * 1000:.2f} nm/px ({origen})")

    r = analizar(rgb, gris, um_px, args.num_medidas, args.separacion, capa=args.capa,
                 sensibilidad=args.sensibilidad, nitidez=args.nitidez,
                 continuidad=args.continuidad, max_huecos=args.max_huecos, semilla=args.semilla)
    medidas, candidatos, rgb_util = r["medidas"], r["candidatos"], r["rgb_util"]
    if r["atipicas"]:
        print(f"  Descartadas {r['atipicas']} medidas atípicas.")
    if not medidas:
        print("  No se pudo tomar ninguna medida. Prueba a bajar --continuidad o --nitidez.")
        return None
    if len(medidas) < args.num_medidas:
        print(f"  Aviso: sólo hay {len(medidas)} puntos válidos en fibras de delante "
              f"({args.num_medidas} pedidos).")
    diam = np.array([m["diametro_um"] for m in medidas])

    carpeta = args.salida or os.path.dirname(os.path.abspath(ruta))
    os.makedirs(carpeta, exist_ok=True)
    pref = os.path.join(carpeta, base)
    dibujar_medidas(rgb_util, medidas, pref + "_medidas.png")
    aumento = aumento_metadatos(ruta)
    titulo = f"{base} · {aumento / 1000:g} KX" if aumento else base
    dibujar_histograma(diam, titulo, pref + "_histograma.png")
    guardar_deteccion(rgb_util, candidatos, pref + "_deteccion.png")
    with open(pref + "_medidas.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["n", "x_px", "y_px", "diametro_px", "diametro_um"])
        for k, m in enumerate(medidas, 1):
            wr.writerow([k, f"{m['x']:.1f}", f"{m['y']:.1f}", f"{m['ancho_px']:.2f}",
                         f"{m['diametro_um']:.4f}"])

    e = estadisticas(diam)
    print(f"  Medidas: {e['n']}   diámetro medio {e['media']:.3f} ± {e['desv']:.3f} µm   "
          f"CV {e['cv']:.1f} %   mediana {e['mediana']:.3f} µm   "
          f"(mín {e['minimo']:.3f}, máx {e['maximo']:.3f})")
    resumen = dict(imagen=base, aumento=f"{aumento / 1000:g} KX" if aumento else "",
                   nm_px=f"{um_px * 1000:.2f}", n=e["n"], media_um=f"{e['media']:.4f}",
                   desv_um=f"{e['desv']:.4f}", cv_pct=f"{e['cv']:.1f}",
                   mediana_um=f"{e['mediana']:.4f}", min_um=f"{e['minimo']:.4f}",
                   max_um=f"{e['maximo']:.4f}")
    return diam, resumen


def main():
    ap = argparse.ArgumentParser(
        description="Mide el diámetro de las fibras de primer plano en imágenes SEM.")
    ap.add_argument("imagenes", nargs="+",
                    help="Imágenes (.tif, .png, .jpg...) o carpetas con imágenes")
    ap.add_argument("-n", "--num-medidas", type=int, default=100,
                    help="Número de medidas por imagen (por defecto 100)")
    ap.add_argument("--aumentos", type=float, nargs="+", default=None,
                    help="Analizar sólo imágenes con estos aumentos, en KX (p.ej. --aumentos 5 10). "
                         "Se lee de los metadatos")
    ap.add_argument("--escala-um", type=float, default=None,
                    help="Valor en µm de la barra de escala verde. Sólo hace falta si la "
                         "imagen no trae el tamaño de píxel en sus metadatos")
    ap.add_argument("--um-por-pixel", type=float, default=None,
                    help="Fija la escala a mano (µm/píxel) en lugar de leerla")
    ap.add_argument("--continuidad", type=float, default=0.7,
                    help="Fracción mínima del recorrido en que los bordes de la fibra deben verse "
                         "sin que otra la tape (0-1). Mayor = sólo las de más arriba "
                         "(por defecto 0.7)")
    ap.add_argument("--max-huecos", type=int, default=2,
                    help="Máximo de cruces en que la fibra puede quedar tapada por otra "
                         "(por defecto 2; 0 = sólo la capa superior)")
    ap.add_argument("--capa", type=float, default=0.7,
                    help="Fracción de las fibras más brillantes que se aceptan: "
                         "0.3 = sólo las más claras, 1.0 = todas (por defecto 0.7)")
    ap.add_argument("--nitidez", type=float, default=0.35,
                    help="Nitidez mínima del borde, relativa a las fibras más nítidas de la imagen "
                         "(0-1). Mayor = sólo fibras muy enfocadas; 0 = sin filtro "
                         "(por defecto 0.35)")
    ap.add_argument("--sensibilidad", type=float, default=0.0,
                    help="Ajuste del detector de fibras: >0 detecta fibras más tenues, "
                         "<0 sólo las más marcadas (por defecto 0)")
    ap.add_argument("--separacion", type=int, default=25,
                    help="Distancia mínima en píxeles entre puntos de medida (por defecto 25)")
    ap.add_argument("--semilla", type=int, default=0, help="Semilla aleatoria (reproducible)")
    ap.add_argument("--salida", default=None,
                    help="Carpeta de salida (por defecto, la misma de cada imagen)")
    args = ap.parse_args()

    rutas = listar_imagenes(args.imagenes)
    if args.aumentos:
        elegidas = []
        for r in rutas:
            aum = aumento_metadatos(r)
            if aum is not None and any(abs(aum - k * 1000) < 1 for k in args.aumentos):
                elegidas.append(r)
            else:
                print(f"Se omite {os.path.basename(r)} (aumento "
                      f"{f'{aum / 1000:g} KX' if aum else 'desconocido'})")
        rutas = elegidas
    if not rutas:
        sys.exit("No hay imágenes que analizar.")

    todas, resumenes = [], []
    for ruta in rutas:
        print(f"== {os.path.basename(ruta)}")
        r = analizar_imagen(ruta, args)
        if r is None:
            continue
        diam, resumen = r
        resumenes.append(resumen)
        todas += [(resumen["imagen"], resumen["aumento"], d) for d in diam]

    if len(resumenes) > 1:
        carpeta = args.salida or os.path.dirname(os.path.abspath(rutas[0]))
        with open(os.path.join(carpeta, "conjunto_medidas.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["imagen", "aumento", "diametro_um"])
            wr.writerows([(i, a, f"{d:.4f}") for i, a, d in todas])
        diam = np.array([d for _, _, d in todas])
        dc = diam.std(ddof=1)
        total = dict(imagen="CONJUNTO", aumento="", nm_px="", n=len(diam),
                     media_um=f"{diam.mean():.4f}", desv_um=f"{dc:.4f}",
                     cv_pct=f"{100 * dc / diam.mean():.1f}", mediana_um=f"{np.median(diam):.4f}",
                     min_um=f"{diam.min():.4f}", max_um=f"{diam.max():.4f}")
        with open(os.path.join(carpeta, "conjunto_resumen.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(resumenes[0]))
            wr.writeheader()
            wr.writerows(resumenes + [total])
        dibujar_histograma(diam, f"Conjunto de {len(resumenes)} imágenes",
                           os.path.join(carpeta, "conjunto_histograma.png"))
        desv = diam.std(ddof=1)
        print(f"\n== Conjunto ({len(resumenes)} imágenes): {len(diam)} medidas   "
              f"diámetro medio {diam.mean():.3f} ± {desv:.3f} µm   "
              f"CV {100 * desv / diam.mean():.1f} %   mediana {np.median(diam):.3f} µm")
        print(f"Resultados en: {carpeta}")


if __name__ == "__main__":
    main()
