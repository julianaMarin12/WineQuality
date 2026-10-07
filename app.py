import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    FEATURES,
    LIMITS,
    TARGET,
    contributions,
    find_training_csv,
    load_artifacts,
    predict,
    train_artifacts,
)

st.set_page_config(
    page_title="Calidad del Vino · Predictor SVR",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Paleta y estilos
# ---------------------------------------------------------------------------
INK = "#2A1219"
MERLOT = "#6E1A33"
BORDEAUX = "#3E0D1E"
VINE = "#6F7D4E"
CORK = "#B48A5A"
ROSE = "#F5EEF0"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Karla:wght@400;500;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, label, li {{
    font-family: 'Karla', 'Helvetica Neue', Arial, sans-serif;
    color: {INK};
}}
h1, h2, h3, .serif {{
    font-family: 'Cormorant Garamond', Georgia, 'Times New Roman', serif !important;
    color: {BORDEAUX};
    letter-spacing: 0.01em;
}}
.block-container {{ padding-top: 3.5rem; max-width: 1200px; }}

/* Etiqueta de botella como cabecera */
.label {{
    background: {BORDEAUX};
    color: #F3E6E9;
    border-radius: 6px;
    padding: 2.2rem 2.6rem 2rem;
    position: relative;
    margin-bottom: 1.6rem;
    box-shadow: inset 0 0 0 1px {CORK}55, inset 0 0 0 8px {BORDEAUX}, inset 0 0 0 9px {CORK}88;
}}
.label h1 {{
    color: #F8EDEF !important;
    font-size: clamp(2.2rem, 4.6vw, 3.6rem);
    font-weight: 600;
    line-height: 1.05;
    margin: 0 0 .6rem 0;
    padding: 0;
}}
.label p {{
    color: #E5CFD5;
    font-size: 1.05rem;
    max-width: 62ch;
    margin: 0;
    line-height: 1.55;
}}
.label .vintage {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-style: italic;
    color: {CORK};
    font-size: 1.15rem;
    margin-bottom: .4rem;
}}

/* Pestañas */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {MERLOT}33; }}
.stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.25rem; font-weight: 600; color: {INK}aa; padding: .4rem 0;
}}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p {{ color: {MERLOT} !important; }}

/* Grupos de variables */
.group-title {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.35rem; font-weight: 600; color: {MERLOT};
    border-bottom: 1px solid {CORK}66; padding-bottom: .2rem; margin: .2rem 0 .6rem;
}}

/* Casillas numéricas */
.stNumberInput [data-baseweb="input"] {{
    background: #fff; border: 1px solid {CORK}66; border-radius: 4px;
}}
.stNumberInput [data-baseweb="input"]:focus-within {{ border-color: {MERLOT}; }}
.stNumberInput input {{ font-variant-numeric: tabular-nums; font-weight: 500; }}
.stNumberInput button {{ color: {MERLOT}; }}

/* Botones */
.stButton > button, .stDownloadButton > button {{
    background: {MERLOT}; color: #fff; border: none; border-radius: 4px;
    font-weight: 700; padding: .6rem 1.4rem;
}}
.stButton > button p, .stDownloadButton > button p {{ color: #fff !important; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {BORDEAUX}; color: #fff; }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
    outline: 3px solid {CORK}; outline-offset: 2px;
}}

/* Resultado */
.result {{ text-align: center; }}
.result .score {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 4.6rem; font-weight: 700; color: {BORDEAUX}; line-height: 1;
}}
.result .score small {{ font-size: 1.6rem; color: {INK}88; font-weight: 500; }}
.result .verdict {{
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-style: italic; font-size: 1.6rem; color: {MERLOT}; margin-top: .2rem;
}}
.result .err {{ font-size: .92rem; color: {INK}99; margin-top: .4rem; }}

.note {{ font-size: .95rem; color: {INK}cc; line-height: 1.6; max-width: 70ch; }}

/* Copa: único momento animado */
@keyframes fill {{ from {{ transform: translateY(100%); }} to {{ transform: translateY(0); }} }}
.wine-level {{ animation: fill 1.4s cubic-bezier(.3,.7,.2,1) both; transform-box: fill-box; }}
@media (prefers-reduced-motion: reduce) {{ .wine-level {{ animation: none; }} }}

footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Metadatos de las variables
# ---------------------------------------------------------------------------
META = {
    "fixed_acidity": ("Acidez fija", "g/L", 0.1, "Ácidos no volátiles (tartárico, málico). Dan frescura."),
    "volatile_acidity": ("Acidez volátil", "g/L", 0.01, "Ácido acético. En exceso da notas a vinagre."),
    "citric_acid": ("Ácido cítrico", "g/L", 0.01, "Aporta frescura y sensación frutal."),
    "residual_sugar": ("Azúcar residual", "g/L", 0.1, "Azúcar que queda tras la fermentación."),
    "chlorides": ("Cloruros", "g/L", 0.001, "Cantidad de sal en el vino."),
    "free_sulfur_dioxide": ("SO₂ libre", "mg/L", 0.5, "Protege contra oxidación y microbios."),
    "density": ("Densidad", "g/cm³", 0.0001, "Depende del alcohol y del azúcar."),
    "pH": ("pH", "", 0.01, "Qué tan ácido es el vino (la mayoría entre 3 y 4)."),
    "alcohol": ("Alcohol", "% vol", 0.1, "Grado alcohólico del vino."),
}
GROUPS = {
    "Acidez": ["fixed_acidity", "volatile_acidity", "citric_acid", "pH"],
    "Azúcar y sales": ["residual_sugar", "chlorides", "free_sulfur_dioxide"],
    "Cuerpo": ["alcohol", "density"],
}
DEFAULTS = {
    "fixed_acidity": 6.9, "volatile_acidity": 0.33, "citric_acid": 0.31,
    "residual_sugar": 3.0, "chlorides": 0.05, "free_sulfur_dioxide": 30.0,
    "density": 0.9945, "pH": 3.20, "alcohol": 10.5,
}
PRESETS = {
    "Perfil promedio": DEFAULTS,
    "Blanco fresco y ligero": {
        "fixed_acidity": 6.6, "volatile_acidity": 0.25, "citric_acid": 0.36,
        "residual_sugar": 1.8, "chlorides": 0.035, "free_sulfur_dioxide": 34.0,
        "density": 0.9910, "pH": 3.18, "alcohol": 12.4,
    },
    "Tinto con cuerpo": {
        "fixed_acidity": 8.0, "volatile_acidity": 0.40, "citric_acid": 0.30,
        "residual_sugar": 2.2, "chlorides": 0.075, "free_sulfur_dioxide": 14.0,
        "density": 0.9955, "pH": 3.35, "alcohol": 13.2,
    },
    "Vino con defectos": {
        "fixed_acidity": 7.8, "volatile_acidity": 0.95, "citric_acid": 0.02,
        "residual_sugar": 2.0, "chlorides": 0.12, "free_sulfur_dioxide": 6.0,
        "density": 0.9978, "pH": 3.55, "alcohol": 9.2,
    },
}

PRESET_STYLE = {
    "Perfil promedio": "Tinto",
    "Blanco fresco y ligero": "Blanco",
    "Tinto con cuerpo": "Tinto",
    "Vino con defectos": "Tinto",
}

# Métricas del notebook (5-Fold CV, modelos optimizados)
CV_RESULTS = pd.DataFrame(
    [
        ("SVM (SVR lineal)", 0.6397, 0.1579, 0.7989, 0.2560),
        ("Stacking", 0.6419, 0.1597, 0.7975, 0.2589),
        ("Regresión Lineal", 0.6416, 0.1595, 0.7974, 0.2588),
        ("Random Forest", 0.6515, 0.1622, 0.8109, 0.2342),
        ("Gradient Boosting", 0.6583, 0.1643, 0.8149, 0.2268),
        ("CNN 1D", 0.6750, 0.1666, 0.8330, 0.1917),
        ("Árbol de Decisión", 0.6677, 0.1653, 0.8373, 0.1835),
        ("KNN", 0.7499, 0.1873, 0.9160, 0.0231),
    ],
    columns=["Modelo", "MAE", "MAPE", "RMSE", "R²"],
)
MAE_SVM = 0.6397


def verdict(q: float) -> str:
    if q < 4:
        return "Calidad baja"
    if q < 5:
        return "Calidad aceptable"
    if q < 6:
        return "Buena calidad"
    if q < 7:
        return "Muy buena calidad"
    return "Calidad excelente"


# Colores de cada estilo: (superficie, fondo). Solo cambian la copa, no la predicción.
WINE_STYLES = {
    "Tinto": ("#9B2343", "#3E0D1E"),
    "Rosado": ("#F08A9B", "#C2475E"),
    "Blanco": ("#F1DC8C", "#C9A646"),
}
MURKY = ("#8A6E52", "#4A3A2C")   # tono turbio de un vino con defectos


def _mix(c1: str, c2: str, t: float) -> str:
    """Mezcla dos colores hex: t=0 -> c1, t=1 -> c2."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def wine_glass_svg(score: float, style: str = "Tinto", vmax: float = 10.0) -> str:
    """Copa: el nivel sube con la calidad, el color depende del estilo y
    se vuelve turbio cuando la calidad es baja."""
    level = float(np.clip(score / vmax, 0.04, 1.0))
    clarity = float(np.clip((score - 3.5) / 1.5, 0.0, 1.0))   # 3.5 -> turbio, 5+ -> limpio
    light, dark = WINE_STYLES.get(style, WINE_STYLES["Tinto"])
    c_top = _mix(MURKY[0], light, clarity)
    c_bottom = _mix(MURKY[1], dark, clarity)
    top, bottom = 30, 190          # límites verticales del cáliz
    y_level = bottom - (bottom - top) * level
    sparkle = ""
    if clarity > 0.75:             # brillo en la superficie para los vinos buenos
        sparkle = (f'<ellipse class="wine-level" cx="80" cy="{y_level + 1:.1f}" rx="18" ry="1.6" '
                   f'fill="#fff" fill-opacity="{0.25 + 0.35 * (clarity - 0.75) / 0.25:.2f}"/>')
    return f"""
<svg viewBox="0 0 200 330" width="190" role="img"
     aria-label="Copa de vino {style.lower()} llena al {level*100:.0f}%">
  <defs>
    <clipPath id="bowl">
      <path d="M40 30 L160 30 C164 110 150 175 100 192 C50 175 36 110 40 30 Z"/>
    </clipPath>
    <linearGradient id="wine" x1="0" y1="0" x2="0.4" y2="1">
      <stop offset="0" stop-color="{c_top}"/>
      <stop offset="1" stop-color="{c_bottom}"/>
    </linearGradient>
  </defs>
  <g clip-path="url(#bowl)">
    <rect class="wine-level" x="0" y="{y_level:.1f}" width="200" height="{bottom - y_level + 10:.1f}" fill="url(#wine)"/>
    <ellipse class="wine-level" cx="100" cy="{y_level:.1f}" rx="70" ry="5" fill="{_mix(c_top, '#ffffff', 0.15)}"/>
    {sparkle}
  </g>
  <path d="M40 30 L160 30 C164 110 150 175 100 192 C50 175 36 110 40 30 Z"
        fill="none" stroke="{INK}" stroke-width="2.5"/>
  <path d="M52 44 C50 90 56 130 70 160" fill="none" stroke="#ffffff" stroke-opacity=".55"
        stroke-width="4" stroke-linecap="round"/>
  <line x1="100" y1="192" x2="100" y2="292" stroke="{INK}" stroke-width="3"/>
  <ellipse cx="100" cy="300" rx="52" ry="9" fill="none" stroke="{INK}" stroke-width="2.5"/>
</svg>"""


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Descorchando el modelo…")
def get_artifacts():
    try:
        art = load_artifacts()
        art["source"] = "Artefactos .joblib exportados desde Colab"
        return art, None
    except Exception as load_err:  # archivos ausentes o versión incompatible
        csv = find_training_csv()
        if csv is not None:
            art = train_artifacts(pd.read_csv(csv))
            art["source"] = f"Entrenado al iniciar con {csv.name}"
            return art, None
        return None, load_err


art, load_error = get_artifacts()

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="label">
  <div class="vintage">Cosecha de datos · modelo SVR optimizado</div>
  <h1>¿Qué tan bueno es este vino?</h1>
  <p>Describe su perfil fisicoquímico y el modelo estima la calificación de calidad
  que le daría un panel de catadores.</p>
</div>
""",
    unsafe_allow_html=True,
)

if art is None:
    st.error(
        "No se encontró el modelo. Sube a la carpeta `models/` los cuatro archivos que "
        "genera el notebook (`knn_imputer.joblib`, `iqr_bounds.joblib`, "
        "`standard_scaler.joblib`, `best_svm_model.joblib`) o coloca "
        "`base procesada.csv` en la carpeta `data/` para entrenarlo al iniciar."
    )
    st.caption(f"Detalle técnico: {load_error}")
    st.stop()

tab_one, tab_batch, tab_model = st.tabs(
    ["Analizar un vino", "Analizar un lote (CSV)", "Cómo funciona el modelo"]
)

# ---------------------------------------------------------------------------
# Pestaña 1: un vino
# ---------------------------------------------------------------------------
with tab_one:
    for f in FEATURES:
        st.session_state.setdefault(f, DEFAULTS[f])

    st.session_state.setdefault("style", "Tinto")

    def apply_preset():
        for f, v in PRESETS[st.session_state["preset"]].items():
            st.session_state[f] = v
        st.session_state["style"] = PRESET_STYLE[st.session_state["preset"]]

    st.selectbox(
        "Partir de un perfil de ejemplo",
        list(PRESETS),
        key="preset",
        on_change=apply_preset,
    )

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        cols = st.columns(3, gap="medium")
        for col, (group, feats) in zip(cols, GROUPS.items()):
            with col:
                st.markdown(f'<div class="group-title">{group}</div>', unsafe_allow_html=True)
                for f in feats:
                    name, unit, step, help_txt = META[f]
                    lo, hi = LIMITS[f]
                    label = f"{name} ({unit})" if unit else name
                    fmt = "%.4f" if f == "density" else ("%.3f" if f == "chlorides" else "%.2f")
                    st.number_input(label, float(lo), float(hi), step=float(step),
                                    key=f, help=help_txt, format=fmt)

    sample = pd.DataFrame([{f: st.session_state[f] for f in FEATURES}])
    q = float(predict(sample, art)[0])

    with right:
        st.segmented_control(
            "Estilo del vino", list(WINE_STYLES), key="style",
            help="Solo cambia el color de la copa; el modelo no usa el tipo de vino.",
        )
        st.markdown(
            f"""
<div class="result">
  {wine_glass_svg(q, st.session_state["style"] or "Tinto")}
  <div class="score">{q:.2f}<small> / 10</small></div>
  <div class="verdict">{verdict(q)}</div>
  <div class="err">Margen típico de error: ± {MAE_SVM:.2f} puntos (MAE en validación cruzada)</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### Qué está moviendo la calificación")
    st.markdown(
        '<p class="note">Como el SVR usa un kernel lineal, cada variable suma o resta una '
        "cantidad concreta respecto al vino promedio de la base de entrenamiento.</p>",
        unsafe_allow_html=True,
    )
    contrib = contributions(sample, art).sort_values()
    labels = [META[f][0] for f in contrib.index]
    fig = go.Figure(
        go.Bar(
            x=contrib.values,
            y=labels,
            orientation="h",
            marker_color=[MERLOT if v >= 0 else VINE for v in contrib.values],
            text=[f"{v:+.2f}" for v in contrib.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.3f} puntos<extra></extra>",
        )
    )
    span = max(0.3, float(np.abs(contrib.values).max()) * 1.35)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Karla, Arial, sans-serif", color=INK, size=14),
        xaxis=dict(range=[-span, span], zeroline=True, zerolinecolor=INK,
                   gridcolor="#E3D3D8", title="Puntos de calidad"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Pestaña 2: lote
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown(
        '<p class="note">Sube un CSV con las nueve variables del modelo. Si incluye la '
        "columna <code>quality</code>, también verás qué tan cerca quedó la predicción. "
        "Las celdas vacías o fuera de rango se imputan igual que en el notebook.</p>",
        unsafe_allow_html=True,
    )
    template = pd.DataFrame([DEFAULTS, PRESETS["Blanco fresco y ligero"], PRESETS["Tinto con cuerpo"]])
    st.download_button(
        "Descargar plantilla CSV",
        template.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_vinos.csv",
        mime="text/csv",
    )

    up = st.file_uploader("Archivo CSV", type=["csv"])
    if up is not None:
        try:
            df_up = pd.read_csv(up, sep=None, engine="python")
        except Exception as e:
            st.error(f"No se pudo leer el archivo como CSV: {e}")
            df_up = None

    if up is not None and df_up is not None:
        missing = [f for f in FEATURES if f not in df_up.columns]
        if len(missing) == len(FEATURES):
            st.error(
                "El archivo no tiene ninguna de las columnas esperadas. "
                "Usa estos nombres: " + ", ".join(FEATURES)
            )
        else:
            if missing:
                st.warning("Columnas ausentes que se imputarán: " + ", ".join(missing))

            preds = predict(df_up, art)
            out = df_up.copy()
            out["prediccion_calidad"] = np.round(preds, 3)
            out["valoracion"] = [verdict(p) for p in preds]

            c1, c2, c3 = st.columns(3)
            c1.metric("Vinos analizados", f"{len(out)}")
            c2.metric("Calidad promedio estimada", f"{preds.mean():.2f}")

            if TARGET in df_up.columns and df_up[TARGET].notna().any():
                real = pd.to_numeric(df_up[TARGET], errors="coerce")
                mask = real.notna()
                out["diferencia_absoluta"] = np.round(np.abs(real - preds), 3)
                c3.metric("Error medio en este lote (MAE)", f"{np.abs(real[mask] - preds[mask]).mean():.3f}")

                lo_ = float(min(real[mask].min(), preds.min())) - 0.3
                hi_ = float(max(real[mask].max(), preds.max())) + 0.3
                sc = go.Figure()
                sc.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines",
                                        line=dict(color=CORK, dash="dot"), name="Predicción perfecta"))
                sc.add_trace(go.Scatter(x=real[mask], y=preds[mask.values], mode="markers",
                                        marker=dict(color=MERLOT, size=10, opacity=.8),
                                        name="Vinos",
                                        hovertemplate="Real %{x}<br>Predicha %{y:.2f}<extra></extra>"))
                sc.update_layout(
                    height=380, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Karla, Arial, sans-serif", color=INK),
                    xaxis=dict(title="Calidad real", gridcolor="#E3D3D8"),
                    yaxis=dict(title="Calidad predicha", gridcolor="#E3D3D8"),
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(sc, width="stretch", config={"displayModeBar": False})

            st.dataframe(out, width="stretch", hide_index=True)
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Descargar resultados", buf.getvalue().encode("utf-8"),
                               file_name="predicciones_vino.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Pestaña 3: modelo
# ---------------------------------------------------------------------------
with tab_model:
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        st.markdown("### El recorrido de cada muestra")
        st.markdown(
            """
<div class="note">
<ol>
<li><b>Limpieza.</b> Los ceros imposibles y los valores fuera de los límites físicos se marcan como faltantes.</li>
<li><b>Imputación.</b> Un KNNImputer (k = 5) rellena los faltantes con vinos parecidos.</li>
<li><b>Recorte de atípicos.</b> Cada variable se limita al rango Q1 − 1.5·IQR a Q3 + 1.5·IQR del entrenamiento.</li>
<li><b>Escalado.</b> StandardScaler con la media y desviación del entrenamiento.</li>
<li><b>Predicción.</b> SVR con kernel lineal, C = 1.0 y ε = 0.01.</li>
</ol>
<p>Se eligió el SVR porque obtuvo el menor MAE (0.640) y MAPE (15.8 %) de los ocho
modelos evaluados con validación cruzada de 5 pliegues. Su R² de 0.256 indica que las
variables fisicoquímicas explican una parte de la calidad, no toda: las predicciones
tienden a acercarse a la media y los vinos extremos se subestiman o sobrestiman.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(f"Origen del modelo: {art['source']}")

    with b:
        st.markdown("### Comparación de modelos (5-Fold CV)")
        ordered = CV_RESULTS.sort_values("MAE", ascending=False)
        bar = go.Figure(
            go.Bar(
                x=ordered["MAE"], y=ordered["Modelo"], orientation="h",
                marker_color=[MERLOT if m.startswith("SVM") else "#D9C3CA" for m in ordered["Modelo"]],
                text=[f"{v:.3f}" for v in ordered["MAE"]], textposition="outside",
                hovertemplate="%{y}: MAE %{x:.4f}<extra></extra>",
            )
        )
        bar.update_layout(
            height=340, margin=dict(l=10, r=30, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Karla, Arial, sans-serif", color=INK, size=13),
            xaxis=dict(title="MAE (menor es mejor)", range=[0.55, 0.8], gridcolor="#E3D3D8"),
        )
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
        st.dataframe(
            CV_RESULTS.style.format({"MAE": "{:.4f}", "MAPE": "{:.2%}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
            width="stretch", hide_index=True,
        )

    model = art["model"]
    if hasattr(model, "coef_"):
        st.markdown("### Peso de cada variable en el modelo")
        coefs = pd.Series(np.ravel(model.coef_), index=[META[f][0] for f in FEATURES]).sort_values()
        cf = go.Figure(go.Bar(
            x=coefs.values, y=coefs.index, orientation="h",
            marker_color=[MERLOT if v >= 0 else VINE for v in coefs.values],
            hovertemplate="%{y}: %{x:+.3f} por desviación estándar<extra></extra>",
        ))
        cf.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Karla, Arial, sans-serif", color=INK, size=13),
            xaxis=dict(title="Cambio en la calidad por cada desviación estándar", gridcolor="#E3D3D8",
                       zeroline=True, zerolinecolor=INK),
        )
        st.plotly_chart(cf, width="stretch", config={"displayModeBar": False})
