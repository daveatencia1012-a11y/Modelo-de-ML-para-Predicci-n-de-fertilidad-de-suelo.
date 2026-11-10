import io
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
COMPONENTES = {
    'N': 'Nitrógeno (N)', 'P': 'Fósforo (P)', 'K': 'Potasio (K)',
    'pH': 'Acidez o alcalinidad (pH)', 'EC': 'Conductividad eléctrica (EC)',
    'OC': 'Carbono orgánico (OC)', 'S': 'Azufre (S)', 'Zn': 'Zinc (Zn)',
    'Fe': 'Hierro (Fe)', 'Cu': 'Cobre (Cu)', 'Mn': 'Manganeso (Mn)', 'B': 'Boro (B)',
}
CLASIFICACIONES = {
    0: {'nombre': 'SUELO MENOS FÉRTIL', 'color': '#DC3545', 'emoji': '🔴',
        'descripcion': 'La muestra presenta una fertilidad relativamente baja según el modelo.'},
    1: {'nombre': 'SUELO FÉRTIL', 'color': '#EAB308', 'emoji': '🟡',
        'descripcion': 'La muestra presenta un nivel de fertilidad favorable según el modelo.'},
    2: {'nombre': 'SUELO ALTAMENTE FÉRTIL', 'color': '#16A34A', 'emoji': '🟢',
        'descripcion': 'La muestra pertenece al nivel de fertilidad más alto identificado en el dataset.'},
}


def numero_visible(valor):
    """Número corto para la etiqueta; el modelo recibe el float64 original."""
    return f'{float(valor):.4f}'.rstrip('0').rstrip('.')


def numero_exacto(valor):
    """Representación decimal mínima que recupera el mismo float64."""
    return str(float(valor))


def limpiar_prediccion():
    for clave in ('ultima_muestra', 'ultima_pred', 'ultima_proba'):
        st.session_state.pop(clave, None)


def validar_dataset(datos):
    encabezado = next(csv.reader(io.StringIO(datos.decode('utf-8-sig'))), [])
    esperadas = [*COMPONENTES, 'Output']
    if len(encabezado) != len(set(encabezado)) or set(encabezado) != set(esperadas):
        raise ValueError('El CSV debe contener exactamente estas columnas, sin duplicados: ' + ', '.join(esperadas))
    df = pd.read_csv(io.BytesIO(datos))
    if df.empty or df.isna().any().any():
        raise ValueError('El CSV está vacío o contiene valores faltantes.')
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in esperadas):
        raise ValueError('Todos los componentes y Output deben ser numéricos.')
    if not np.isfinite(df[esperadas].to_numpy(dtype=float)).all():
        raise ValueError('El CSV contiene valores infinitos o fuera del rango numérico.')
    if set(df['Output'].unique()) != {0, 1, 2}:
        raise ValueError('Output debe incluir las tres clases originales: 0, 1 y 2.')
    if df['Output'].value_counts().min() < 3:
        raise ValueError('Se necesitan al menos tres registros por clase para dividir y balancear los datos.')
    # Un único contrato float64 evita truncar percentiles fraccionarios de columnas enteras.
    df = df[esperadas].astype({c: 'float64' for c in COMPONENTES})
    df['Output'] = df['Output'].astype('int64')
    return df


def cargar_clases(dataset_id):
    """Una revisión documental solo autoriza los bytes del CSV revisado."""
    ruta = BASE_DIR / 'clases.json'
    if not ruta.exists():
        return {}
    try:
        config = json.loads(ruta.read_text(encoding='utf-8'))
        if config.get('dataset_sha256') != dataset_id or dataset_id is None:
            st.warning('La documentación de clases no corresponde al CSV actual. Se usan clases neutrales.')
            return {}
        if not config['fuente'].strip() or not config['evidencia'].strip():
            raise ValueError('Falta la fuente o la evidencia de comparación de valores.')
        if config['correspondencia'] != {'0': 'menos fértil', '1': 'fértil', '2': 'altamente fértil'}:
            raise ValueError('La correspondencia no coincide con la presentación documentada.')
        return {str(c): info for c, info in CLASIFICACIONES.items()}
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        st.warning(f'No se pueden verificar los significados de las clases: {exc}')
        return {}


def etiqueta_clase(clase):
    info = CLASES.get(str(int(clase)))
    return f"{info['nombre']} (clase {int(clase)})" if info else f'Clase {int(clase)}'

st.set_page_config(
    page_title='FertiIA | Fertilidad del suelo',
    page_icon='🌱',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --primary:#24543a;
        --primary-dark:#193d2b;
        --olive:#556329;
        --accent:#eef1e5;
        --surface:#ffffff;
        --background:#f4f6f3;
        --text:#1f2d26;
        --muted:#54645c;
        --border:#dce8df;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    /* Streamlit puede asignar colores propios a los elementos internos. */
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stText"],
    .stApp [data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stCaptionContainer"] * {
        color: inherit;
    }

    .block-container {
        max-width: 1260px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #f0f6f2;
        color: var(--text);
        border-right: 1px solid #dce8df;
    }

    [data-testid="stHeader"] {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stCaptionContainer"],
    [data-testid="stExpander"] summary {
        color: var(--text) !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: var(--primary-dark);
    }

    .stApp a {
        color: var(--olive);
        text-decoration: underline;
    }

    .hero {
        background: linear-gradient(120deg, #193d2b 0%, #556329 100%);
        border-radius: 24px;
        padding: 2rem 2.2rem;
        color: white;
        box-shadow: 0 10px 26px rgba(31,95,65,.18);
        margin-bottom: 1.1rem;
    }

    .stApp .hero h1 {
        margin: 0 0 .45rem 0;
        color: white;
        font-size: 2.35rem;
        line-height: 1.1;
    }

    .hero p {
        margin: 0;
        color: #edf8f1;
        font-size: 1.02rem;
    }

    .section-tag {
        color: var(--primary-dark);
        letter-spacing: .08em;
        font-weight: 800;
        font-size: .85rem;
        margin-top: 1rem;
        margin-bottom: .45rem;
    }

    .soft-card {
        background: white;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 8px 18px rgba(31,95,65,.05);
        height: 100%;
    }

    .soft-card h4 {
        margin: 0 0 .35rem 0;
        color: var(--primary-dark);
    }

    .soft-card p, .soft-card li {
        color: var(--muted);
        margin-bottom: .2rem;
    }

    [data-testid="stMetric"],
    div[data-testid="metric-container"] {
        background: var(--surface) !important;
        color: var(--primary-dark) !important;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        min-height: 125px;
        box-shadow: 0 6px 14px rgba(31,95,65,.05);
        width: 100%;
        box-sizing: border-box;
    }

    [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] * {
        color: var(--muted) !important;
        font-weight: 600;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] * {
        color: var(--primary-dark) !important;
        font-size: clamp(1.8rem, 3vw, 2.6rem);
        font-weight: 750;
        line-height: 1.2;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .45rem;
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: var(--accent);
        padding: .5rem 1rem;
        height: auto;
        color: var(--primary-dark) !important;
        flex-shrink: 0;
    }

    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }

    .stApp .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: var(--olive);
    }

    [data-baseweb="input"] {
        border-radius: 12px !important;
        border: 1px solid #d8e4db !important;
        background: white !important;
    }

    [data-baseweb="input"] input {
        color: #1f2d26 !important;
        -webkit-text-fill-color: #1f2d26;
        background: white;
    }

    .stApp input::placeholder, .stApp textarea::placeholder {
        color: var(--muted);
        -webkit-text-fill-color: var(--muted);
        opacity: 1;
    }

    [data-baseweb="input"]:focus-within {
        border-color: var(--olive) !important;
        box-shadow: 0 0 0 2px var(--accent);
    }

    [data-testid="stForm"] {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.2rem;
    }

    [data-testid="stFileUploaderDropzone"],
    [data-testid="stExpander"] {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    .stApp [data-testid="stFileUploaderDropzone"] small,
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--muted) !important;
    }

    [data-testid="stTooltipContent"],
    [data-baseweb="tooltip"],
    [data-testid="stAlert"] {
        background: var(--accent) !important;
        color: var(--text) !important;
    }

    [data-testid="stTooltipContent"] *,
    [data-baseweb="tooltip"] *,
    .stApp [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        color: inherit !important;
    }

    [data-testid="stNumberInput"] button,
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--accent);
        color: var(--primary-dark);
        border-color: var(--border);
    }

    .stApp [data-testid="stFileUploaderDropzone"] button *,
    .stApp [data-testid="stNumberInput"] button * {
        color: inherit;
    }

    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        border-radius: 12px;
        border: 1px solid var(--primary);
        background: var(--primary);
        color: white !important;
        font-weight: 600;
    }

    .stApp .stButton button *, .stApp .stDownloadButton button *,
    .stApp .stFormSubmitButton button * {
        color: inherit !important;
    }

    .stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
        border-color: var(--primary-dark);
        background: var(--primary-dark);
    }

    .stApp button:focus-visible, .stApp input:focus-visible {
        outline: 3px solid var(--olive);
        outline-offset: 3px;
    }

    .stButton button:disabled, .stDownloadButton button:disabled,
    .stFormSubmitButton button:disabled {
        background: #e3e7df;
        color: #54645c !important;
        border-color: #b5bfb3;
        opacity: 1;
    }

    .tip-box {
        background: #f8fbf9;
        border: 1px dashed #b7d1c0;
        border-radius: 16px;
        padding: .95rem 1rem;
        color: #425a4d;
    }

    .small-note {
        color: #5e6d65;
        font-size: .92rem;
    }

    .footer-note {
        color: var(--muted);
        font-size: .9rem;
        margin-top: 1.5rem;
    }

    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 calc(50% - 1rem) !important;
            min-width: 0 !important;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    @media (max-width: 600px) {
        .block-container {
            padding: 1rem .85rem 2rem;
        }

        .hero {
            padding: 1.4rem 1.2rem;
            border-radius: 18px;
        }

        .stApp .hero h1 {
            font-size: 1.8rem;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
        }

        .stTabs [data-baseweb="tab"] {
            flex-shrink: 0;
        }

        [data-testid="stForm"] {
            padding: .85rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner='Entrenando el modelo con tu dataset...')
def entrenar(datos: bytes):
    df = validar_dataset(datos)
    if 'Output' not in df.columns:
        raise ValueError('El archivo debe incluir una columna llamada Output.')

    X = df.drop(columns=['Output'])
    y = df['Output']

    if X.empty:
        raise ValueError('No hay variables de entrada para entrenar el modelo.')

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise ValueError('Todas las variables predictoras deben ser numéricas.')

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError('El dataset contiene valores vacíos. Debes limpiarlo antes de usarlo.')

    if y.nunique() < 2:
        raise ValueError('Se necesitan al menos dos clases distintas en Output.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    conteos = y_train.value_counts()
    if conteos.min() < 2:
        raise ValueError('SMOTE necesita al menos dos muestras por clase en entrenamiento.')

    smote = SMOTE(random_state=42, k_neighbors=min(5, int(conteos.min()) - 1))
    X_res, y_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_res_scaled, y_res)
    y_pred = clf.predict(X_test_scaled)

    importancias = pd.DataFrame({
        'Variable': X.columns,
        'Importancia': clf.feature_importances_
    }).sort_values('Importancia', ascending=False)

    resumen = {
        'df': df,
        'X': X,
        'y': y,
        'cols': list(X.columns),
        'scaler': scaler,
        'clf': clf,
        'y_test': y_test,
        'preds': y_pred,
        'acc': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'reporte': classification_report(y_test, y_pred, labels=clf.classes_, output_dict=True, zero_division=0),
        'importancias': importancias,
    }
    return resumen


def cargar_datos_bytes():
    with st.sidebar:
        st.title('🌿 Panel de control')
        st.markdown('Usa el dataset del proyecto o sube uno nuevo con la misma estructura.')
        archivo = st.file_uploader(
            'Sube dataset1.csv', type=['csv'], help='Debe contener las mismas variables y la columna Output.'
        )

        st.markdown('### Guía rápida')
        st.markdown(
            '- **1.** Selecciona las cantidades de los componentes del suelo.\n'
            '- **2.** Presiona **Analizar fertilidad**.\n'
            '- **3.** Revisa la predicción y las gráficas.'
        )
        if st.button('🔄 Volver a entrenar', use_container_width=True):
            entrenar.clear()
            limpiar_prediccion()
            st.rerun()

        st.markdown('---')
        st.caption('Proyecto académico · Sistemas Inteligentes')

    if archivo is not None:
        return archivo.getvalue()

    try:
        return (BASE_DIR / 'dataset1.csv').read_bytes()
    except OSError as exc:
        st.error(f'No se pudo cargar dataset1.csv desde la carpeta de la aplicación: {exc}')

    return None


def fig_distribucion_clases(df):
    conteo = df['Output'].value_counts().sort_index().reset_index()
    conteo.columns = ['Clase', 'Cantidad']
    conteo['Clase'] = conteo['Clase'].map(etiqueta_clase)
    fig = px.bar(
        conteo,
        x='Clase',
        y='Cantidad',
        text='Cantidad',
        title='Distribución de clases de fertilidad en el dataset',
        color_discrete_sequence=['#64748b']
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template='plotly_white', font_color='#1f2d26', plot_bgcolor='white', paper_bgcolor='white', yaxis_title='Registros')
    return fig


def fig_importancias(importancias):
    df = importancias.sort_values('Importancia', ascending=True)
    fig = px.bar(
        df,
        x='Importancia',
        y='Variable',
        orientation='h',
        title='Variables que más influyen en la predicción',
        color_discrete_sequence=['#2f7d57']
    )
    fig.update_layout(template='plotly_white', font_color='#1f2d26', plot_bgcolor='white', paper_bgcolor='white', yaxis_title='')
    return fig


def fig_confusion(y_real, y_pred, clases):
    matriz = confusion_matrix(y_real, y_pred, labels=clases)
    fig = px.imshow(
        matriz,
        x=[etiqueta_clase(c) for c in clases],
        y=[etiqueta_clase(c) for c in clases],
        text_auto=True,
        color_continuous_scale='Greens',
        labels={'x': 'Predicción', 'y': 'Valor real', 'color': 'Casos'},
        title='Matriz de confusión del modelo'
    )
    fig.update_layout(template='plotly_white', font_color='#1f2d26', paper_bgcolor='white')
    return fig


def fig_correlacion(df):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdYlGn',
        title='Relación entre las variables del suelo'
    )
    fig.update_layout(template='plotly_white', font_color='#1f2d26', height=650, paper_bgcolor='white')
    return fig


def fig_probabilidades(clases, probabilidades):
    temp = pd.DataFrame({'Clase': [etiqueta_clase(c) for c in clases], 'Probabilidad': probabilidades})
    fig = px.bar(
        temp,
        x='Clase',
        y='Probabilidad',
        text='Probabilidad',
        title='Probabilidades de fertilidad',
        color='Clase',
        color_discrete_map={etiqueta_clase(c): CLASES.get(str(int(c)), {}).get('color', '#64748b') for c in clases}
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(template='plotly_white', font_color='#1f2d26', yaxis_tickformat='.0%', yaxis_range=[0, 1.10], plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
    return fig


def fig_comparacion_muestra(muestra_df, X):
    posiciones = [100 * (X[c] <= muestra_df.iloc[0][c]).mean() for c in X.columns]
    fig = go.Figure(go.Bar(
        x=posiciones, y=[COMPONENTES[c] for c in X.columns], orientation='h',
        marker_color='#24543a', text=[f'{p:.1f}%' for p in posiciones],
        hovertemplate='%{y}: %{x:.1f}% de registros iguales o menores<extra></extra>',
    ))
    fig.update_layout(title='Posición de cada componente dentro del dataset',
                      xaxis_title='Registros con un valor igual o menor (%)',
                      xaxis_range=[0, 100], height=540, template='plotly_white')
    return fig


def fig_componente(muestra, X, variable):
    fig = go.Figure(go.Box(x=X[variable], name='Dataset', boxpoints=False,
                           marker_color='#556329'))
    fig.add_trace(go.Scatter(x=[muestra.iloc[0][variable]], y=['Dataset'],
                            mode='markers', name='Tu muestra',
                            marker=dict(color='#24543a', size=16, symbol='diamond')))
    fig.update_layout(title=COMPONENTES[variable], template='plotly_white',
                      xaxis_title='Valor en la escala original del CSV', height=300)
    return fig


def mostrar_inicio(model):
    st.markdown(
        '<div class="hero"><h1>🌱 FertiIA · Diagnóstico inteligente del suelo</h1>'
        '<p>Aplicación de apoyo para estimar la <b>clase de fertilidad del suelo</b> a partir de nutrientes y propiedades químicas, usando Machine Learning con Random Forest.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-tag">RESUMEN DEL SISTEMA</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Registros analizados', f"{len(model['df']):,}")
    m2.metric('Variables de entrada', len(model['cols']))
    m3.metric('Exactitud', f"{model['acc']:.1%}")
    m4.metric('F1 ponderado', f"{model['f1']:.1%}")

    st.markdown('<div class="section-tag">¿CÓMO FUNCIONA?</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="soft-card"><h4>1. Seleccionas las cantidades</h4><p>Eliges un nivel bajo, medio o alto para cada componente del suelo.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-card"><h4>2. El modelo analiza</h4><p>FertiIA compara tu muestra con el patrón aprendido del dataset y estima la clase de fertilidad.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="soft-card"><h4>3. Obtienes la explicación</h4><p>La app muestra la clase predicha, las probabilidades y gráficas para interpretar el resultado.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="tip-box"><b>Importante:</b> esta herramienta interpreta mediciones que ya fueron tomadas del suelo. No reemplaza el laboratorio ni la asesoría agronómica; es un apoyo para el análisis académico.</div>', unsafe_allow_html=True)


def tab_prediccion(model):
    st.subheader('Analiza una muestra de suelo')
    st.write('Selecciona una cantidad para cada componente. Entre paréntesis aparece el valor que analizará el modelo.')
    st.caption('Baja, Media y Alta son los percentiles 25, 50 y 75 del CSV actual: categorías relativas a estos datos, no límites agronómicos validados. Un pH alto o bajo no significa automáticamente un suelo bueno o malo. Las opciones son aproximaciones, no mediciones de tu terreno.')

    izquierda, derecha = st.columns([1.55, 1], gap='large')
    with izquierda:
        with st.container(border=True):
            cab1, cab2 = st.columns(2)
            cab1.markdown('**Componente del suelo**')
            cab2.markdown('**Cantidad**')
            valores = {}
            for variable in model['cols']:
                serie = model['X'][variable]
                niveles = serie.quantile([.25, .5, .75]).tolist()
                nombre, control = st.columns(2)
                nombre.markdown(f'**{COMPONENTES[variable]}**')
                with control:
                    indice = st.selectbox(
                        f'Cantidad de {COMPONENTES[variable]}', [0, 1, 2], index=1,
                        format_func=lambda i, n=niveles: f"{['Baja', 'Media', 'Alta'][i]} ({numero_visible(n[i])})",
                        key=f"nivel_{st.session_state['dataset_id']}_{variable}",
                        on_change=limpiar_prediccion,
                        label_visibility='collapsed',
                        help=f'Rango observado en el CSV: {serie.min():g} a {serie.max():g}'
                    )
                    valores[variable] = niveles[indice]
            analizar = st.button('🌱 Analizar fertilidad', type='primary', use_container_width=True)

    with derecha:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('#### ¿Qué significan estas variables?')
        st.markdown(
            '- **N, P, K:** nutrientes principales del suelo.\n'
            '- **pH:** nivel de acidez o alcalinidad.\n'
            '- **EC:** conductividad eléctrica.\n'
            '- **OC:** carbono orgánico.\n'
            '- **S, Zn, Fe, Cu, Mn, B:** nutrientes secundarios y micronutrientes.'
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('#### Nota para interpretar la salida')
        st.markdown(
            'La clase corresponde a los códigos originales del dataset (**0, 1 o 2**). '
            'Los nombres y colores se habilitan únicamente para el CSV con documentación vinculada por huella digital. '
            'Las gráficas describen los datos y el modelo; no demuestran causalidad.'
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if analizar:
        try:
            muestra = pd.DataFrame([valores], columns=model['cols']).astype(model['X'].dtypes.to_dict())
            muestra_scaled = model['scaler'].transform(muestra)
            pred = model['clf'].predict(muestra_scaled)[0]
            proba = model['clf'].predict_proba(muestra_scaled)[0]
        except Exception as exc:
            limpiar_prediccion()
            st.error(f'No se pudo analizar la muestra. Revisa el dataset o vuelve a entrenar: {exc}')
            return

        st.session_state['ultima_muestra'] = muestra
        st.session_state['ultima_pred'] = pred
        st.session_state['ultima_proba'] = proba

    if 'ultima_muestra' in st.session_state:
        muestra = st.session_state['ultima_muestra']
        pred = st.session_state['ultima_pred']
        proba = st.session_state['ultima_proba']

        st.markdown('### Resultado del análisis')
        info = CLASES.get(str(int(pred)))
        probabilidad = proba[list(model['clf'].classes_).index(pred)]
        if info:
            st.markdown(
                f'<div role="status" style="background:#101713;color:#f5f8f5;border:1px solid #3a4a40;'
                f'border-radius:22px;padding:1.7rem 1.6rem;text-align:center">'
                f'<div style="font-weight:650;text-align:left;font-size:1.2rem">🌿 FertiIA · Resultado del análisis</div>'
                f'<div style="font-size:2.4rem;margin-top:.65rem">{info["emoji"]}</div>'
                f'<div style="font-size:clamp(1.5rem,3vw,2rem);font-weight:850;'
                f'color:{info["color"]};margin:.4rem 0">{info["nombre"]}</div>'
                f'<p style="color:#e0e8e1">{info["descripcion"]}</p>'
                f'<p style="font-weight:650;color:#e0e8e1">Probabilidad estimada para esta clase: {probabilidad:.1%}</p>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown('#### Probabilidades del modelo')
            for clase, p in zip(model['clf'].classes_, proba):
                detalle = CLASES.get(str(int(clase)), {})
                nombre_clase = detalle.get('nombre', f'CLASE {int(clase)}')
                color = detalle.get('color', '#64748b')
                pct = max(0.0, min(100.0, 100.0 * float(p)))
                st.markdown(
                    f'<div style="margin:.6rem 0 1.0rem">'
                    f'<div style="display:flex;justify-content:space-between;gap:1rem;'
                    f'font-weight:600;margin-bottom:.35rem"><span>{nombre_clase}</span>'
                    f'<span>{pct:.1f}%</span></div>'
                    f'<div role="img" aria-label="{nombre_clase}: {pct:.1f}%" '
                    f'style="height:14px;background:#dfe5df;border-radius:12px;overflow:hidden">'
                    f'<div style="height:100%;width:{pct:.2f}%;background:{color};border-radius:12px"></div>'
                    f'</div></div>', unsafe_allow_html=True
                )
            st.caption('Estimaciones de Random Forest, no certezas ni mediciones directas de fertilidad. El amarillo distingue la categoría intermedia, no una advertencia.')
            st.caption('Etiquetas respaldadas por publicaciones del dataset Kaggle y habilitadas solo para la versión documentada del CSV; no equivalen a un diagnóstico agronómico certificado.')
        else:
            st.markdown(f'## {etiqueta_clase(pred)} · Nivel sin correspondencia documentada')
            st.info('Este dataset no coincide con el archivo cuyas clases fueron documentadas. Se conservan las clases numéricas y no se asignan colores ni interpretaciones supuestas.')
        r1, r2, r3 = st.columns([.9, .9, 1.2])
        r1.metric('Clase predicha', str(pred))
        r2.metric('Probabilidad de la clase predicha', f'{probabilidad:.1%}')
        r3.caption('La probabilidad estimada puede ser poco confiable, especialmente para la clase 2, que tiene pocos registros.')

        g1, g2 = st.columns(2, gap='large')
        with g1:
            st.plotly_chart(fig_probabilidades(model['clf'].classes_, proba), use_container_width=True)
        with g2:
            st.plotly_chart(fig_comparacion_muestra(muestra, model['X']), use_container_width=True)
        st.caption('La comparación usa porcentajes para no mezclar escalas. Por ejemplo, 75% significa que tres de cada cuatro registros tienen un valor igual o menor; no significa 75% de fertilidad. Los valores repetidos pueden cambiar esta posición.')
        st.markdown('#### Componentes seleccionados y distribución real')
        variable = st.selectbox('Componente que deseas explorar', model['cols'], format_func=lambda c: COMPONENTES[c])
        st.plotly_chart(fig_componente(muestra, model['X'], variable), use_container_width=True)
        st.caption('El rombo es tu muestra; la caja contiene la mitad central del dataset y la línea interior marca la mediana. Se muestra una variable por vez porque las escalas y unidades son distintas; el CSV no documenta sus unidades.')
        st.dataframe(pd.DataFrame({'Componente': [COMPONENTES[c] for c in model['cols']], 'Valor seleccionado': muestra.iloc[0].to_numpy()}), hide_index=True, use_container_width=True)

        export_df = muestra.copy()
        export_df['Output_predicho'] = pred
        export_df['Fecha'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.download_button(
            '⬇️ Descargar resultado en CSV',
            export_df.to_csv(index=False).encode('utf-8'),
            file_name='resultado_prediccion_fertilia.csv',
            mime='text/csv'
        )


def tab_analisis(model):
    st.subheader('Gráficas para comprender el funcionamiento del sistema')
    st.write('Estas visualizaciones ayudan a entender con qué datos se entrenó el modelo y qué variables influyen más en la predicción.')

    r1, r2, r3, r4 = st.columns(4)
    r1.metric('Precisión', f"{model['precision']:.1%}")
    r2.metric('Recall', f"{model['recall']:.1%}")
    r3.metric('F1-Score', f"{model['f1']:.1%}")
    r4.metric('Clases detectadas', model['y'].nunique())

    st.markdown('#### Evaluación por clase · datos de prueba')
    filas = []
    for clase in model['clf'].classes_:
        metricas = model['reporte'][str(int(clase))]
        filas.append({'Clase': etiqueta_clase(clase), 'Precisión': metricas['precision'],
                      'Recall': metricas['recall'], 'F1': metricas['f1-score'],
                      'Registros de prueba': int(metricas['support']),
                      'Registros totales': int((model['y'] == clase).sum())})
    st.dataframe(pd.DataFrame(filas).style.format({'Precisión': '{:.1%}', 'Recall': '{:.1%}', 'F1': '{:.1%}'}),
                 hide_index=True, use_container_width=True)
    st.caption('Precisión: aciertos entre las predicciones de esa clase. Recall: proporción de casos reales recuperados. F1: balance entre ambos. La clase 2 tiene solo 39 registros en el CSV original; pocos casos de prueba hacen que sus métricas sean inestables. SMOTE no añade casos reales a la evaluación. Estas métricas y las gráficas no demuestran causalidad.')

    g1, g2 = st.columns(2, gap='large')
    with g1:
        st.plotly_chart(fig_distribucion_clases(model['df']), use_container_width=True)
    with g2:
        st.plotly_chart(fig_importancias(model['importancias']), use_container_width=True)

    g3, g4 = st.columns(2, gap='large')
    with g3:
        st.plotly_chart(fig_confusion(model['y_test'], model['preds'], list(model['clf'].classes_)), use_container_width=True)
    with g4:
        st.plotly_chart(fig_correlacion(model['df']), use_container_width=True)

    with st.expander('Ver una muestra del dataset usado por la aplicación'):
        st.dataframe(model['df'].head(100), use_container_width=True)


def tab_sobre(model):
    st.subheader('Apartado explicativo para el usuario')
    st.markdown(
        '<div class="soft-card">'
        '<h4>¿Qué hace esta página?</h4>'
        '<p>FertiIA ayuda a <b>estimar la fertilidad del suelo</b> usando un modelo de Machine Learning entrenado con datos históricos. '
        'El usuario ingresa las mediciones del suelo y la aplicación devuelve la clase de fertilidad más probable.</p>'
        '<h4>¿Cómo toma la decisión?</h4>'
        '<p>El sistema usa un <b>Random Forest</b>, que combina varios árboles de decisión. Antes de entrenar, los datos se dividen en entrenamiento y prueba, '
        'se balancean con <b>SMOTE</b> y luego se escalan con <b>StandardScaler</b>.</p>'
        '<h4>¿Qué muestran las gráficas?</h4>'
        '<ul>'
        '<li><b>Distribución de clases:</b> cuántos ejemplos hay de cada clase.</li>'
        '<li><b>Importancia de variables:</b> qué nutrientes influyen más en la predicción.</li>'
        '<li><b>Matriz de confusión:</b> en qué tanto acierta o se equivoca el modelo.</li>'
        '<li><b>Correlación:</b> cómo se relacionan las variables del suelo entre sí.</li>'
        '</ul>'
        '<p><b>Advertencia:</b> esta aplicación tiene fines académicos y de apoyo. No sustituye un diagnóstico agronómico profesional.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('### Estructura del modelo')
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown('<div class="soft-card"><h4>Paso 1</h4><p>Se cargan y validan los datos del suelo.</p></div>', unsafe_allow_html=True)
    p2.markdown('<div class="soft-card"><h4>Paso 2</h4><p>Se balancean las clases con SMOTE.</p></div>', unsafe_allow_html=True)
    p3.markdown('<div class="soft-card"><h4>Paso 3</h4><p>Se escalan las variables con StandardScaler.</p></div>', unsafe_allow_html=True)
    p4.markdown('<div class="soft-card"><h4>Paso 4</h4><p>Se entrena Random Forest y se generan predicciones.</p></div>', unsafe_allow_html=True)


datos = cargar_datos_bytes()
dataset_id = hashlib.sha256(datos).hexdigest() if datos is not None else None
CLASES = cargar_clases(dataset_id)
if st.session_state.get('dataset_id') != dataset_id:
    limpiar_prediccion()
    for clave in list(st.session_state):
        if clave.startswith('nivel_'):
            del st.session_state[clave]
st.session_state['dataset_id'] = dataset_id
if datos is None:
    limpiar_prediccion()
    st.sidebar.error('Dataset no disponible')
    st.markdown(
        '<div class="hero"><h1>🌱 FertiIA · Diagnóstico inteligente del suelo</h1>'
        '<p>Sube tu archivo <b>dataset1.csv</b> o coloca el dataset en la misma carpeta de la aplicación para comenzar.</p></div>',
        unsafe_allow_html=True,
    )
    st.error('No se encontró un dataset válido. Sube un CSV con la estructura del proyecto para habilitar el análisis.')
    st.stop()

try:
    model = entrenar(datos)
except Exception as e:
    limpiar_prediccion()
    st.sidebar.error('Dataset inválido o entrenamiento no disponible')
    st.error(f'No fue posible preparar el modelo: {e}')
    st.stop()

st.sidebar.success('Dataset cargado y modelo listo')
st.sidebar.metric('Registros', len(model['df']))
st.sidebar.metric('Variables de entrada', len(model['cols']))
st.sidebar.caption('12 componentes y la columna objetivo Output. Al retirar el archivo subido se restaura el CSV del proyecto.')

mostrar_inicio(model)

t1, t2, t3 = st.tabs(['🔬 Predicción', '📊 Gráficas y análisis', '📖 Ayuda al usuario'])
with t1:
    tab_prediccion(model)
with t2:
    tab_analisis(model)
with t3:
    tab_sobre(model)

st.markdown(
    '<div class="footer-note">Desarrollado como interfaz académica para un proyecto de Sistemas Inteligentes orientado a la fertilidad del suelo.</div>',
    unsafe_allow_html=True
)
