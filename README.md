# Medidor de diámetro de fibras (SEM)

Aplicación web para medir el diámetro de las fibras de primer plano en imágenes de
microscopía electrónica de barrido. Se suben una o varias imágenes y devuelve, para cada una:

- la imagen con cada medida señalada y numerada,
- el histograma de diámetros (media, desviación estándar y coeficiente de variación),
- la tabla de medidas,

además de una tabla resumen de todas las imágenes, el histograma del conjunto y botones
para exportar (Excel, ZIP con todo, CSV).

## Archivos

| Archivo | Qué es |
|---|---|
| `app.py` | La aplicación web (Streamlit) |
| `medir_fibras.py` | El análisis de imagen. También se puede usar solo, desde la terminal |
| `requirements.txt` | Librerías necesarias |
| `.streamlit/config.toml` | Colores de la app y tamaño máximo de subida (200 MB) |

## Publicarla en Streamlit Community Cloud (gratis)

1. Crea un repositorio en GitHub y sube **todo el contenido de esta carpeta**, incluida la
   carpeta oculta `.streamlit`.
2. Entra en <https://share.streamlit.io> con tu cuenta de GitHub.
3. Pulsa **Create app** → **Deploy a public app from GitHub**.
4. Elige el repositorio y la rama; en **Main file path** escribe `app.py`.
5. En **Advanced settings**, elige **Python 3.12** (o superior).
6. Pulsa **Deploy**. La primera vez tarda unos minutos en instalar las librerías.

Cada vez que subas cambios al repositorio, la app se actualiza sola.

## Probarla en tu ordenador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador en <http://localhost:8501>.

## Usar el análisis desde la terminal (sin la web)

```bash
python medir_fibras.py carpeta_con_fotos/ --aumentos 5 10 --salida resultados/
```

`python medir_fibras.py --help` muestra todas las opciones.

## Notas

- **Escala**: los TIFF del microscopio ZEISS traen el tamaño de píxel en sus metadatos y la
  app lo lee sola. Si subes PNG/JPG (sin metadatos), indica en el panel lateral cuánto mide la
  barra de escala verde. Es menos precisa que los metadatos (la barra mide pocos píxeles), así
  que conviene usar siempre los TIFF originales.
- **Tiempo**: cada imagen tarda entre 20 segundos y 1 minuto. Si repites el análisis con la
  misma imagen y los mismos ajustes, el resultado sale al instante.
- **Fibras de primer plano**: sólo se miden fibras no tapadas por otras y enfocadas. La
  «Imagen de control» de cada resultado muestra en verde las medidas aceptadas y en rojo las
  descartadas por pertenecer a fibras de detrás. Los criterios se ajustan en
  «Opciones avanzadas».
# app_fibras
