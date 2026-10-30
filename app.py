"""
Medidor de diámetro de fibras (SEM) — aplicación web con Streamlit.

Ejecutar en local:
    pip install -r requirements.txt
    streamlit run app.py

Toda la parte de análisis está en medir_fibras.py; aquí sólo va la interfaz.
"""

import io
import os
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import medir_fibras as mf

st.set_page_config(page_title="Medidor de fibras SEM", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def a_png(arr_o_fig):
    """Convierte una imagen (array) o una figura de matplotlib a bytes PNG."""
    buf = io.BytesIO()
    if isinstance(arr_o_fig, np.ndarray):
        Image.fromarray(arr_o_fig).save(buf, format="PNG")
    else:
        arr_o_fig.savefig(buf, format="PNG")
        plt.close(arr_o_fig)
    return buf.getvalue()


def info_archivo(datos):
    """Aumento y escala que trae la imagen en sus metadatos (sin analizarla)."""
    aumento = mf.aumento_metadatos(io.BytesIO(datos))
    um_px = mf.escala_metadatos(io.BytesIO(datos))
    return aumento, um_px


def texto_aumento(aumento):
    return f"{aumento / 1000:g} KX" if aumento else "—"


@st.cache_data(show_spinner=False, max_entries=30)
def procesar(datos, nombre, modo_escala, valor_escala, parametros):
    """Analiza una imagen subida. Devuelve un dict con resultados e imágenes en PNG,
    o un dict con 'error'. Se guarda en caché: misma imagen + mismos ajustes = instantáneo."""
    rgb, gris = mf.leer_imagen(io.BytesIO(datos))
    aumento = mf.aumento_metadatos(io.BytesIO(datos))
    y_info = mf.separar_barra_info(gris)

    # Escala µm/píxel
    if modo_escala == "manual":
        um_px, origen = valor_escala, "fijada a mano"
    else:
        um_meta = mf.escala_metadatos(io.BytesIO(datos))
        if modo_escala == "auto" and um_meta:
            um_px, origen = um_meta, "metadatos del microscopio"
        else:
            largo = mf.detectar_barra_escala(rgb, y_info)
            if modo_escala == "auto":
                return dict(error="La imagen no trae el tamaño de píxel en sus metadatos. "
                                  "En el panel lateral, en «Escala», elige «Barra de escala» "
                                  "e indica cuánto mide la barra, o fija los µm por píxel.")
            if not largo:
                return dict(error="No se encontró la barra de escala verde. En el panel lateral, "
                                  "fija los µm por píxel a mano.")
            um_px, origen = valor_escala / largo, f"barra verde de {largo} px = {valor_escala:g} µm"

    r = mf.analizar(rgb, gris, um_px, **parametros)
    medidas = r["medidas"]
    if not medidas:
        return dict(error="No se encontró ninguna fibra de primer plano que medir. Prueba a "
                          "bajar «Continuidad» o «Nitidez» en las opciones avanzadas.")

    diam = np.array([m["diametro_um"] for m in medidas])
    base = os.path.splitext(nombre)[0]
    titulo = f"{base} · {texto_aumento(aumento)}" if aumento else base
    tabla = pd.DataFrame({
        "n": np.arange(1, len(medidas) + 1),
        "x (px)": [round(float(m["x"]), 1) for m in medidas],
        "y (px)": [round(float(m["y"]), 1) for m in medidas],
        "Diámetro (px)": [round(float(m["ancho_px"]), 2) for m in medidas],
        "Diámetro (µm)": [round(float(m["diametro_um"]), 4) for m in medidas],
    })
    return dict(
        base=base, aumento=aumento, um_px=um_px, origen=origen,
        estadisticas=mf.estadisticas(diam), diametros=diam, tabla=tabla,
        atipicas=r["atipicas"], pedidas=parametros["n_medidas"],
        png_medidas=a_png(mf.imagen_medidas(r["rgb_util"], medidas)),
        png_deteccion=a_png(mf.imagen_deteccion(r["rgb_util"], r["candidatos"])),
        png_histograma=a_png(mf.figura_histograma(diam, titulo)),
    )


def fila_resumen(nombre, res):
    e = res["estadisticas"]
    return {
        "Imagen": res["base"], "Aumento": texto_aumento(res["aumento"]),
        "Escala (nm/px)": round(res["um_px"] * 1000, 2), "N.º medidas": e["n"],
        "Media (µm)": round(e["media"], 4), "Desv. est. (µm)": round(e["desv"], 4),
        "CV (%)": round(e["cv"], 1), "Mediana (µm)": round(e["mediana"], 4),
        "Mín (µm)": round(e["minimo"], 4), "Máx (µm)": round(e["maximo"], 4),
    }


def excel_resultados(resumen, medidas):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="Resumen", index=False)
        medidas.to_excel(xw, sheet_name="Medidas", index=False)
        for hoja, df in (("Resumen", resumen), ("Medidas", medidas)):
            ws = xw.sheets[hoja]
            for i, col in enumerate(df.columns, 1):
                ancho = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) + 2
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(ancho, 40)
    return buf.getvalue()


def zip_resultados(resultados, resumen, medidas, png_conjunto):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("resumen.csv", resumen.to_csv(index=False))
        z.writestr("todas_las_medidas.csv", medidas.to_csv(index=False))
        z.writestr("resultados.xlsx", excel_resultados(resumen, medidas))
        if png_conjunto:
            z.writestr("conjunto_histograma.png", png_conjunto)
        for res in resultados.values():
            b = res["base"]
            z.writestr(f"{b}/{b}_medidas.png", res["png_medidas"])
            z.writestr(f"{b}/{b}_histograma.png", res["png_histograma"])
            z.writestr(f"{b}/{b}_deteccion.png", res["png_deteccion"])
            z.writestr(f"{b}/{b}_medidas.csv", res["tabla"].to_csv(index=False))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Panel lateral: ajustes
# --------------------------------------------------------------------------- #
AJUSTES = ("n_medidas", "continuidad", "max_huecos", "nitidez", "capa", "separacion")
with st.sidebar:
    st.header("⚙️ Ajustes")
    n_medidas = st.slider("Medidas por imagen", 20, 300, 100, step=10, key="n_medidas",
                          help="Cuántos diámetros intentar medir en cada imagen. Si no hay "
                               "suficientes fibras de primer plano, se toman las que haya.")

    st.subheader("Escala")
    modo = st.radio("¿De dónde sale la escala?",
                    ["Automática (metadatos)", "Barra de escala", "µm por píxel"],
                    help="Las imágenes TIFF del microscopio ZEISS traen el tamaño de píxel "
                         "guardado y no hace falta tocar nada. Para otras imágenes, indica "
                         "cuánto mide la barra verde o la escala directamente.")
    modo_escala, valor_escala = "auto", 0.0
    if modo == "Barra de escala":
        modo_escala = "barra"
        valor_escala = st.number_input("La barra verde mide (µm)", min_value=0.001,
                                       value=1.0, step=0.5, format="%.3f")
    elif modo == "µm por píxel":
        modo_escala = "manual"
        valor_escala = st.number_input("µm por píxel", min_value=0.0001, value=0.0276,
                                       step=0.001, format="%.4f")

    with st.expander("Opciones avanzadas"):
        st.caption("Controlan qué fibras se consideran «de primer plano». Los valores por "
                   "defecto funcionan bien con imágenes de 5 KX y 10 KX.")
        continuidad = st.slider("Continuidad", 0.0, 1.0, 0.7, 0.05, key="continuidad",
                                help="Fracción del recorrido en que los bordes de la fibra deben "
                                     "verse sin que otra la tape. Más alto = sólo las de más arriba.")
        max_huecos = st.slider("Cruces tapados permitidos", 0, 6, 2, key="max_huecos",
                               help="Cuántas veces puede quedar la fibra tapada por otra. "
                                    "0 = sólo la capa superior.")
        nitidez = st.slider("Nitidez mínima", 0.0, 1.0, 0.35, 0.05, key="nitidez",
                            help="Nitidez del borde comparada con las fibras más nítidas de la "
                                 "imagen. Más alto = sólo fibras muy enfocadas.")
        capa = st.slider("Fracción de fibras más claras", 0.1, 1.0, 0.7, 0.05, key="capa",
                         help="Sólo se aceptan fibras entre las más claras de la imagen. "
                              "1.0 = sin filtro de brillo.")
        separacion = st.slider("Separación entre medidas (px)", 5, 100, 25, key="separacion",
                               help="Distancia mínima entre dos puntos de medida.")
        if st.button("Restablecer valores por defecto", width="stretch"):
            for k in AJUSTES:
                st.session_state.pop(k, None)
            st.rerun()

parametros = dict(n_medidas=n_medidas, separacion=separacion, capa=capa, nitidez=nitidez,
                  continuidad=continuidad, max_huecos=max_huecos)


# --------------------------------------------------------------------------- #
# Página principal
# --------------------------------------------------------------------------- #
st.title("🔬 Medidor de diámetro de fibras")
st.markdown(
    "Sube tus imágenes de microscopía electrónica (SEM) y obtén el **diámetro de las fibras de "
    "primer plano**, con cada medida señalada en la imagen, su **histograma** y una **tabla "
    "resumen** lista para exportar."
)

with st.expander("ℹ️ ¿Cómo funciona?"):
    st.markdown(
        """
1. **Sube** una o varias imágenes (TIFF del microscopio, PNG o JPG).
2. Pulsa **Analizar**. Cada imagen tarda entre 20 segundos y 1 minuto.
3. Revisa los resultados y **descarga** el Excel o el ZIP con todo.

**Qué hace el análisis**
- Lee la escala (nm por píxel) de los metadatos del microscopio.
- Detecta las fibras y mide su diámetro **de borde a borde y en perpendicular** a la fibra.
- Sólo mide fibras de **primer plano**: las que no están tapadas por otras y están enfocadas.
  Las de detrás salen borrosas y su medida estaría distorsionada.
- Calcula media, desviación estándar, **coeficiente de variación (CV)**, mediana, mínimo y máximo.

**Imagen de control:** en verde, las medidas aceptadas; en rojo, las descartadas por
pertenecer a fibras de detrás. Sirve para comprobar que el filtro funciona bien.
"""
    )

archivos = st.file_uploader("Sube las imágenes", type=["tif", "tiff", "png", "jpg", "jpeg"],
                            accept_multiple_files=True,
                            help="Puedes arrastrar varias imágenes a la vez.")

if not archivos:
    st.info("👆 Sube una o varias imágenes para empezar.")
    st.stop()

# Vista previa de lo subido: aumento y escala detectados
filas = []
for a in archivos:
    aumento, um_px = info_archivo(a.getvalue())
    filas.append({"Imagen": a.name, "Aumento": texto_aumento(aumento),
                  "Escala": f"{um_px * 1000:.2f} nm/px" if um_px else "sin metadatos",
                  "Tamaño": f"{a.size / 1e6:.1f} MB"})
st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
sin_escala = [f["Imagen"] for f in filas if f["Escala"] == "sin metadatos"]
if sin_escala and modo_escala == "auto":
    st.warning("Estas imágenes no traen la escala en sus metadatos: "
               f"**{', '.join(sin_escala)}**. Indica la escala en el panel lateral "
               "(«Barra de escala» o «µm por píxel»); si no, se omitirán.")

clave_actual = (tuple((a.name, a.size) for a in archivos), modo_escala, valor_escala,
                tuple(sorted(parametros.items())))
etiqueta = "1 imagen" if len(archivos) == 1 else f"{len(archivos)} imágenes"
if st.button(f"▶️ Analizar {etiqueta}",
             type="primary", width="stretch"):
    resultados, errores = {}, {}
    barra = st.progress(0.0, text="Preparando…")
    for i, a in enumerate(archivos):
        barra.progress(i / len(archivos),
                       text=f"Analizando {a.name} ({i + 1} de {len(archivos)})…")
        try:
            res = procesar(a.getvalue(), a.name, modo_escala, valor_escala, parametros)
        except Exception as ex:  # una imagen dañada no debe tumbar el resto
            res = dict(error=f"No se pudo procesar la imagen ({ex}).")
        if "error" in res:
            errores[a.name] = res["error"]
        else:
            resultados[a.name] = res
    barra.progress(1.0, text="¡Listo!")
    barra.empty()
    st.session_state["resultados"] = resultados
    st.session_state["errores"] = errores
    st.session_state["clave"] = clave_actual

if "resultados" not in st.session_state:
    st.stop()
if st.session_state.get("clave") != clave_actual:
    st.info("Has cambiado las imágenes o los ajustes. Pulsa **Analizar** para actualizar "
            "los resultados.")

resultados, errores = st.session_state["resultados"], st.session_state["errores"]
for nombre, err in errores.items():
    st.error(f"**{nombre}**: {err}")
if not resultados:
    st.stop()

# --------------------------------------------------------------------------- #
# Resumen general
# --------------------------------------------------------------------------- #
st.divider()
st.header("📊 Resultados")

resumen = pd.DataFrame([fila_resumen(n, r) for n, r in resultados.items()])
todas = pd.concat([r["tabla"].assign(Imagen=r["base"], Aumento=texto_aumento(r["aumento"]))
                   [["Imagen", "Aumento", "n", "x (px)", "y (px)", "Diámetro (px)", "Diámetro (µm)"]]
                   for r in resultados.values()], ignore_index=True)
diam_total = todas["Diámetro (µm)"].to_numpy()
eg = mf.estadisticas(diam_total)
png_conjunto = None
if len(resultados) > 1:
    resumen = pd.concat([resumen, pd.DataFrame([{
        "Imagen": "CONJUNTO", "Aumento": "", "Escala (nm/px)": np.nan, "N.º medidas": eg["n"],
        "Media (µm)": round(eg["media"], 4), "Desv. est. (µm)": round(eg["desv"], 4),
        "CV (%)": round(eg["cv"], 1), "Mediana (µm)": round(eg["mediana"], 4),
        "Mín (µm)": round(eg["minimo"], 4), "Máx (µm)": round(eg["maximo"], 4)}])],
        ignore_index=True)
    png_conjunto = a_png(mf.figura_histograma(diam_total, f"Conjunto de {len(resultados)} imágenes"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Imágenes analizadas", len(resultados), border=True)
c2.metric("Medidas totales", eg["n"], border=True)
c3.metric("Diámetro medio", f"{eg['media']:.3f} µm", border=True,
          help=f"± {eg['desv']:.3f} µm (desviación estándar). Mediana {eg['mediana']:.3f} µm.")
c4.metric("Coef. de variación", f"{eg['cv']:.1f} %", border=True,
          help="Desviación estándar / media × 100. Indica cuán homogéneo es el grosor.")

st.subheader("Tabla resumen")
NUM = st.column_config.NumberColumn
# Para mostrar: la escala como texto, para que la fila del conjunto quede en blanco
vista = resumen.assign(**{"Escala (nm/px)": resumen["Escala (nm/px)"].map(
    lambda v: "" if pd.isna(v) else f"{v:.2f}")})
st.dataframe(vista, hide_index=True, width="stretch",
             column_config={"Media (µm)": NUM(format="%.4f"), "Desv. est. (µm)": NUM(format="%.4f"),
                            "CV (%)": NUM(format="%.1f %%"), "Mediana (µm)": NUM(format="%.4f"),
                            "Mín (µm)": NUM(format="%.4f"), "Máx (µm)": NUM(format="%.4f")})

marca = datetime.now().strftime("%Y%m%d_%H%M")
d1, d2, d3 = st.columns(3)
d1.download_button("⬇️ Descargar Excel (resumen + medidas)",
                   excel_resultados(resumen, todas), f"fibras_{marca}.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   on_click="ignore", type="primary", width="stretch")
d2.download_button("⬇️ Descargar todo (ZIP con imágenes)",
                   zip_resultados(resultados, resumen, todas, png_conjunto),
                   f"fibras_{marca}.zip", "application/zip", on_click="ignore", width="stretch",
                   help="Incluye, por cada imagen, la imagen con las medidas, el histograma, "
                        "la imagen de control y su tabla; además del resumen y el Excel.")
d3.download_button("⬇️ Descargar resumen (CSV)", resumen.to_csv(index=False).encode("utf-8-sig"),
                   f"resumen_fibras_{marca}.csv", "text/csv", on_click="ignore", width="stretch")

if png_conjunto:
    st.subheader("Histograma del conjunto")
    izq, _ = st.columns([3, 2])
    izq.image(png_conjunto, width="stretch")

# --------------------------------------------------------------------------- #
# Detalle por imagen
# --------------------------------------------------------------------------- #
st.divider()
st.header("🖼️ Detalle por imagen")
pestanas = st.tabs([f"{r['base']} · {texto_aumento(r['aumento'])}" for r in resultados.values()])
for pestana, (nombre, res) in zip(pestanas, resultados.items()):
    with pestana:
        e = res["estadisticas"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Medidas", e["n"])
        m2.metric("Media", f"{e['media']:.3f} µm")
        m3.metric("Desv. est.", f"{e['desv']:.3f} µm")
        m4.metric("CV", f"{e['cv']:.1f} %")
        m5.metric("Mediana", f"{e['mediana']:.3f} µm")
        notas = [f"Escala: {res['um_px'] * 1000:.2f} nm/px ({res['origen']})."]
        if e["n"] < res["pedidas"]:
            notas.append(f"Sólo se encontraron {e['n']} puntos válidos en fibras de primer plano "
                         f"de {res['pedidas']} pedidos.")
        if res["atipicas"]:
            notas.append(f"Se descartaron {res['atipicas']} medidas atípicas (zonas "
                         "desenfocadas o grumos).")
        st.caption(" ".join(notas))

        izq, der = st.columns([3, 2])
        with izq:
            st.markdown("**Medidas en la imagen** (cada número corresponde a la tabla)")
            st.image(res["png_medidas"], width="stretch")
            st.caption("💡 Pasa el ratón por encima de la imagen y pulsa el botón de pantalla "
                       "completa (arriba a la derecha) para verla en grande.")
        with der:
            st.markdown("**Histograma**")
            st.image(res["png_histograma"], width="stretch")
            st.markdown("**Tabla de medidas**")
            st.dataframe(res["tabla"], hide_index=True, width="stretch", height=260,
                         column_config={"x (px)": NUM(format="%.1f"), "y (px)": NUM(format="%.1f"),
                                        "Diámetro (px)": NUM(format="%.2f"),
                                        "Diámetro (µm)": NUM(format="%.4f")})

        with st.expander("Imagen de control: fibras de primer plano (verde) y descartadas (rojo)"):
            st.image(res["png_deteccion"], width="stretch")

        b = res["base"]
        k1, k2, k3 = st.columns(3)
        k1.download_button("⬇️ Imagen con medidas", res["png_medidas"], f"{b}_medidas.png",
                           "image/png", on_click="ignore", width="stretch", key=f"im_{nombre}")
        k2.download_button("⬇️ Histograma", res["png_histograma"], f"{b}_histograma.png",
                           "image/png", on_click="ignore", width="stretch", key=f"hi_{nombre}")
        k3.download_button("⬇️ Medidas (CSV)", res["tabla"].to_csv(index=False).encode("utf-8-sig"),
                           f"{b}_medidas.csv", "text/csv", on_click="ignore", width="stretch",
                           key=f"cs_{nombre}")

st.caption("Medidor de fibras SEM · análisis automático de diámetros de fibras de primer plano.")
