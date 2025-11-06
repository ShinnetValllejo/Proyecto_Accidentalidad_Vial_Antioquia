# AppDashboard.py
# DASHBOARD INTERACTIVO — Proyecto Accidentabilidad Vial Antioquia
# Requiere: streamlit, streamlit-elements, pillow, pandas

import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import base64
from streamlit_elements import elements, mui, html, dashboard

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Dashboard — Accidentabilidad Vial Valle de Aburrá",
                   layout="wide", page_icon="🚦")

BASE_DIR = Path(__file__).parent.resolve()
GRAF_DIR = BASE_DIR / "Graficas_Salida"
MODEL_DIR = BASE_DIR / "Modelo_Predict"
IMG_PATH = BASE_DIR / "Diseño" / "Map_portada.png"

# -----------------------
# UTIL: base64 para imágenes
# -----------------------
def img_to_base64(path: Path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def pil_to_base64_img_tag(pil_img: Image.Image, fmt="PNG", max_w=None):
    # convierte PIL a data URI para usar en html.img()
    from io import BytesIO
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{data}"

# -----------------------
# CARGA DE RECURSOS
# -----------------------
@st.cache_data
def load_images():
    out = {}
    for name, fname in [
        ("gravedad", "Accidentes_Gravedad_SVA.jpg"),
        ("jornada", "Accidentes_Jornada_SVA.jpg"),
        ("clase", "Accidentes_Clase_SVA.jpg"),
        ("comuna", "Accidentes_Comuna_SVA.jpg"),
        ("roc", "Curva_ROC_SVA.jpg"),
        ("matriz", "Matriz_Confusion_SVA.jpg")
    ]:
        p = GRAF_DIR / fname
        if p.exists():
            out[name] = Image.open(p).convert("RGBA")
        else:
            out[name] = None
    return out

@st.cache_data
def load_dataframes():
    out = {}
    try:
        out["importancia"] = pd.read_csv(MODEL_DIR / "Importancia_Variables_RF.csv")
    except Exception:
        out["importancia"] = None
    try:
        out["predicciones"] = pd.read_csv(MODEL_DIR / "Predicciones_Nuevos_Accidentes.csv")
    except Exception:
        out["predicciones"] = None
    try:
        out["resumen"] = (MODEL_DIR / "Resumen_Ejecutivo_Modelo.txt").read_text(encoding="utf-8")
    except Exception:
        out["resumen"] = ""
    return out

IMAGES = load_images()
DATA = load_dataframes()
PORTADA_BG = img_to_base64(IMG_PATH)

# -----------------------
# ESTILOS GLOBALES (forzar fondo blanco base)
# -----------------------
st.markdown(
    """
    <style>
    .stApp { background-color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# NAVEGACIÓN
# -----------------------
if "vista" not in st.session_state:
    st.session_state["vista"] = "portada"

def go(v):
    st.session_state["vista"] = v

# -----------------------
# PORTADA (completa en streamlit-elements)
# -----------------------
def mostrar_portada():
    # embedea imagen de portada en CSS background (base64)
    bg_data = PORTADA_BG or ""
    st.markdown(
        f"""
        <style>
        .portada {{
            width: 100%;
            height: 78vh;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-image: url("data:image/png;base64,{bg_data}");
            background-position: center;
            background-size: cover;
            background-repeat: no-repeat;
            padding: 3rem;
            box-sizing: border-box;
        }}
        .portada_box {{
            background: rgba(255,255,255,0.88);
            padding: 2rem 2.8rem;
            border-radius: 14px;
            max-width: 55%;
            margin-left: auto;
            margin-right: 4%;
            box-shadow: 0 8px 30px rgba(0,0,0,0.18);
            display:flex;
            flex-direction:column;
            justify-content:center;
        }}
        .titulo {{
            font-family: "Times New Roman", serif;
            font-weight: 700;
            font-size: 34px;
            color: #0f1720;
            text-align: right;
            line-height: 1.35;
            white-space: pre-line;
        }}
        .button_row {{
            display:flex;
            justify-content:center;
            gap:14px;
            margin-top:18px;
        }}
        .stButton>button {{
            min-width:160px;
            height:44px;
            border-radius:10px;
            background-color:#2e7d32;
            color:white;
            font-weight:600;
        }}
        .stButton>button:hover {{ background-color:#43a047; }}
        </style>
        """, unsafe_allow_html=True
    )

    # markup container
    st.markdown("<div class='portada'>", unsafe_allow_html=True)
    # left side can be empty or show small logo / whitespace
    st.markdown("<div style='width:28%;'></div>", unsafe_allow_html=True)

    # right panel with title and buttons
    st.markdown("<div class='portada_box'>", unsafe_allow_html=True)
    # Title (multiline, right aligned)
    st.markdown(
        "<div class='titulo'>{}</div>".format(
            "Análisis de datos y modelado predictivo\nsobre la accidentalidad vial\nen el Valle de Aburrá\n(2015–2019)"
        ),
        unsafe_allow_html=True
    )

    # buttons (use streamlit buttons centered)
    st.markdown("<div class='button_row'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1,1,1,1], gap="small")
    with c1:
        if st.button("📊 Análisis Exploratorio"):
            go("dashboard")
    with c2:
        if st.button("🤖 Modelo Predictivo"):
            go("modelo")
    with c3:
        if st.button("📈 Curvas y Resultados"):
            go("curvas")
    with c4:
        if st.button("🧾 Resumen Ejecutivo"):
            go("resumen")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # cierre portada_box
    st.markdown("</div>", unsafe_allow_html=True)  # cierre portada

# -----------------------
# DASHBOARD INTERACTIVO (streamlit-elements grid)
# -----------------------
def mostrar_dashboard_interactivo():
    st.markdown("<h2 style='margin-bottom:8px;'>Panel interactivo — Diseñe su layout</h2>", unsafe_allow_html=True)

    # layout inicial (4 celdas)
    layout = [
        dashboard.Item("g1", 0, 0, 6, 5),
        dashboard.Item("g2", 6, 0, 6, 5),
        dashboard.Item("g3", 0, 5, 6, 5),
        dashboard.Item("g4", 6, 5, 6, 5),
    ]

    # helper: obtener data-uri de PIL image
    def pil_to_data_uri(img):
        if img is None:
            return ""
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    with elements("dashboard"):
        with dashboard.Grid(layout, draggableHandle=".draggable", rowHeight=90):
            # Tarjeta 1
            with mui.Card(key="g1", sx={"p":2, "height":"100%"}):
                mui.Typography("Distribución por Gravedad", variant="h6")
                src = pil_to_data_uri(IMAGES.get("gravedad"))
                if src:
                    html.img(src=src, style={"width":"100%", "borderRadius":"8px", "marginTop":"8px"})
                else:
                    mui.Typography("Gráfica no disponible", variant="body2")
            # Tarjeta 2
            with mui.Card(key="g2", sx={"p":2, "height":"100%"}):
                mui.Typography("Accidentes por Jornada", variant="h6")
                src = pil_to_data_uri(IMAGES.get("jornada"))
                if src:
                    html.img(src=src, style={"width":"100%", "borderRadius":"8px", "marginTop":"8px"})
                else:
                    mui.Typography("Gráfica no disponible", variant="body2")
            # Tarjeta 3
            with mui.Card(key="g3", sx={"p":2, "height":"100%"}):
                mui.Typography("Tipos de Accidente", variant="h6")
                src = pil_to_data_uri(IMAGES.get("clase"))
                if src:
                    html.img(src=src, style={"width":"100%", "borderRadius":"8px", "marginTop":"8px"})
                else:
                    mui.Typography("Gráfica no disponible", variant="body2")
            # Tarjeta 4
            with mui.Card(key="g4", sx={"p":2, "height":"100%"}):
                mui.Typography("Top 10 Comunas", variant="h6")
                src = pil_to_data_uri(IMAGES.get("comuna"))
                if src:
                    html.img(src=src, style={"width":"100%", "borderRadius":"8px", "marginTop":"8px"})
                else:
                    mui.Typography("Gráfica no disponible", variant="body2")

    st.markdown("<div style='margin-top:14px; display:flex; gap:12px'>", unsafe_allow_html=True)
    if st.button("🏠 Volver a Portada"):
        go("portada")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# VISTA MODELO PREDICTIVO (estructura con cards)
# -----------------------
def mostrar_modelo():
    st.header("Modelo Predictivo — Random Forest")
    if DATA["importancia"] is not None:
        st.subheader("Importancia de Variables")
        st.dataframe(DATA["importancia"].head(30), use_container_width=True)
    else:
        st.info("No se encontró archivo de importancia de variables.")

    st.divider()
    st.subheader("Predicciones simuladas")
    if DATA["predicciones"] is not None:
        st.dataframe(DATA["predicciones"], use_container_width=True)
    else:
        st.info("No se encontró archivo de predicciones.")

    st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
    if st.button("🏠 Volver a Portada"):
        go("portada")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# VISTA CURVAS Y MATRICES
# -----------------------
def mostrar_curvas():
    st.header("Curvas y Matrices — Evaluación del modelo")
    col1, col2 = st.columns(2)
    with col1:
        if IMAGES.get("roc") is not None:
            st.image(IMAGES["roc"], caption="Curva ROC", use_column_width=True)
        else:
            st.info("Curva ROC no disponible.")
    with col2:
        if IMAGES.get("matriz") is not None:
            st.image(IMAGES["matriz"], caption="Matriz de Confusión", use_column_width=True)
        else:
            st.info("Matriz de confusión no disponible.")

    st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
    if st.button("🏠 Volver a Portada"):
        go("portada")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# VISTA RESUMEN EJECUTIVO
# -----------------------
def mostrar_resumen():
    st.header("Resumen Ejecutivo")
    st.markdown(DATA["resumen"] or "Resumen no disponible.")
    st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
    if st.button("🏠 Volver a Portada"):
        go("portada")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# CONTROLADOR PRINCIPAL
# -----------------------
def main():
    v = st.session_state.get("vista", "portada")
    if v == "portada":
        mostrar_portada()
    elif v == "dashboard":
        mostrar_dashboard_interactivo()
    elif v == "modelo":
        mostrar_modelo()
    elif v == "curvas":
        mostrar_curvas()
    elif v == "resumen":
        mostrar_resumen()
    else:
        mostrar_portada()

if __name__ == "__main__":
    main()
