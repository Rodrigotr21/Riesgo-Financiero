# dashboard_riesgo_financiero_peru.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
from deep_translator import GoogleTranslator

# ============================================================================
# 1. CONFIGURACIÓN INICIAL (DEBE SER LA PRIMERA INSTRUCCIÓN DE STREAMLIT)
# ============================================================================
st.set_page_config(
    page_title="Dashboard de Riesgo Financiero - ROSTADINA EIRL",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# 2. OPTIMIZACIÓN Y CACHÉ DE MODELOS DE IA (MOTOR DE RIESGO SEM)
# ============================================================================
@st.cache_resource
def cargar_modelos_ia():
    """Carga los modelos pesados de IA en memoria RAM una sola vez."""
    analista = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    traductor = GoogleTranslator(source='auto', target='en')
    return analista, traductor

def procesar_riesgo_ia(noticias_fuente):
    """Procesa los titulares, traduce, calcula sentimiento y extrae el SEM."""
    analista_ia, traductor = cargar_modelos_ia()
    scores_por_capa = {"macro": [], "social": [], "micro": []}
    detalles_noticias = []
    
    for item in noticias_fuente:
        noticia = item["texto"]
        capa = item["capa"]
        
        texto_analizar = noticia if noticia.isascii() else traductor.translate(noticia)
        
        res = analista_ia(texto_analizar)[0]
        label, score = res['label'], res['score']
        
        w = -1 if label == 'negative' else (1 if label == 'positive' else 0)
        impacto_ponderado = w * score
        scores_por_capa[capa].append(impacto_ponderado)
        
        detalles_noticias.append({
            "noticia": noticia,
            "capa": capa.upper(),
            "sentimiento": label.upper(),
            "confianza": score
        })
        
    I_macro = np.mean(scores_por_capa["macro"]) if scores_por_capa["macro"] else 0.0
    I_social = np.mean(scores_por_capa["social"]) if scores_por_capa["social"] else 0.0
    I_micro = np.mean(scores_por_capa["micro"]) if scores_por_capa["micro"] else 0.0
    
    S_raw = (0.40 * I_macro) + (0.35 * I_social) + (0.25 * I_micro)
    SEM = 50 * (1 - S_raw)
    
    return SEM, I_macro, I_social, I_micro, detalles_noticias

# ============================================================================
# 3. INICIALIZAR SESSION_STATE - CRÍTICO PARA STREAMLIT
# ============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'close_prices' not in st.session_state:
    st.session_state.close_prices = pd.DataFrame()
if 'returns' not in st.session_state:
    st.session_state.returns = pd.DataFrame()
if 'sp500_returns' not in st.session_state:
    st.session_state.sp500_returns = pd.Series()
if 'portfolio_configured' not in st.session_state:
    st.session_state.portfolio_configured = False
if 'portfolio_dist' not in st.session_state:
    st.session_state.portfolio_dist = {}
if 'capital_total' not in st.session_state:
    st.session_state.capital_total = 1000000
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ["BAP", "SCCO", "BVN"]
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = "Todos"

# ============================================================================
# 4. ESTILOS CSS PERSONALIZADOS
# ============================================================================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #3B82F6; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 10px; }
    .rostadina-banner { background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
    .alert-box { background-color: #2c3e50; border-left: 5px solid #e74c3c; padding: 15px; border-radius: 5px; margin-bottom: 10px; color: white !important; }
    .alert-box strong { color: #ecf0f1 !important; }
    .alert-box em { color: #bdc3c7 !important; }
    .success-box { background-color: #27ae60; border-left: 5px solid #2ecc71; padding: 15px; border-radius: 5px; margin-bottom: 10px; color: white !important; }
    .success-box strong { color: white !important; }
    .portfolio-box { background-color: #1E3A8A; border: 2px solid #3B82F6; border-radius: 10px; padding: 20px; margin-bottom: 20px; color: white !important; }
    .portfolio-box h3 { color: white !important; }
    .portfolio-box p { color: #ecf0f1 !important; }
    .warning-box { background-color: #f39c12; border-left: 5px solid #e67e22; padding: 15px; border-radius: 5px; margin-bottom: 10px; color: #2c3e50 !important; }
    .warning-box strong { color: #2c3e50 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 5. BARRA LATERAL (SIDEBAR) - CONTROL DE ENTRADAS Y TICKERS DEL PERÚ
# ============================================================================
st.sidebar.title("Configuración de Mercado")

# Sector Económico de la BVL para filtrar visualizaciones
sector_seleccionado = st.sidebar.selectbox(
    "Sector Económico Local:",
    ["Todos", "Financiero", "Minero", "Consumo/Industrial"]
)
st.session_state.selected_sector = sector_seleccionado

# Tickers por defecto purgados de activos extranjeros - 100% Bandera Peruana
# BAP (Credicorp), SCCO (Southern Copper), BVN (Buenaventura), IFS (Intercorp Financial), ALICORC1.LM (Alicorp), CREDIC1.LM (Banco de Crédito)
tickers_por_defecto = ["BAP", "SCCO", "BVN", "IFS", "ALICORC1.LM"]

tickers_input = st.sidebar.text_input(
    "Tickers de la BVL / ADRs Peruanos (Separados por comas):",
    value=", ".join(tickers_por_defecto)
)

# Procesar y limpiar la lista de tickers ingresados
lista_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
st.session_state.selected_tickers = lista_tickers

# Configuración de horizontes temporales
dias_historicos = st.sidebar.slider("Historial de Análisis (Días):", 30, 365 * 5, 365)
end_date = datetime.now()
start_date = end_date - timedelta(days=dias_historicos)

# ============================================================================
# 6. MOTOR DE CONSULTA DE APIS (YAHOO FINANCE) Y PROCESAMIENTO MATEMÁTICO
# ============================================================================
if st.sidebar.button("🔄 Ejecutar Análisis de Riesgo", key="btn_cargar_datos"):
    if not lista_tickers:
        st.sidebar.error("❌ Por favor, introduce al menos un ticker válido.")
    else:
        with st.spinner("Conectando con Yahoo Finance y descargando activos peruanos..."):
            try:
                # Descarga masiva de los precios de cierre de los activos locales
                raw_data = yf.download(lista_tickers, start=start_date, end=end_date)
                
                if raw_data.empty:
                    st.sidebar.error("❌ No se encontraron datos para los tickers ingresados.")
                else:
                    # Estructurar DataFrame en base a cierres ajustados o normales
                    if 'Adj Close' in raw_data and not raw_data['Adj Close'].empty:
                        df_close = raw_data['Adj Close']
                    else:
                        df_close = raw_data['Close']
                    
                    # Manejar caso de que se consulte un solo ticker (Yahoo Finance cambia la forma del DataFrame)
                    if isinstance(df_close, pd.Series):
                        df_close = df_close.to_frame(name=lista_tickers[0])
                        
                    # Limpieza de nulos por desalineación de días festivos en la BVL
                    df_close = df_close.ffill().bfill()
                    
                    # Calcular retornos porcentuales diarios
                    df_returns = df_close.pct_change().dropna()
                    
                    # Guardar en el Session State global para persistencia entre pestañas
                    st.session_state.close_prices = df_close
                    st.session_state.returns = df_returns
                    st.session_state.data_loaded = True
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error al procesar datos financieros: {str(e)}")
        
        # Descarga del Benchmark Neto Peruano: Índice General BVL (^SPBLPGPT)
        with st.spinner("Descargando datos del Índice General BVL..."):
            try:
                sp500_data = yf.download("^SPBLPGPT", start=start_date, end=end_date)
                if not sp500_data.empty:
                    if 'Adj Close' in sp500_data:
                        bench_series = sp500_data['Adj Close']
                    else:
                        bench_series = sp500_data['Close']
                    st.session_state.sp500_returns = bench_series.pct_change().dropna()
                else:
                    # En caso de caída de Yahoo o falta de datos del índice local, se usa el promedio del mercado
                    if not st.session_state.returns.empty:
                        st.session_state.sp500_returns = st.session_state.returns.mean(axis=1)
            except Exception:
                # Fallback de seguridad ante desconexiones de la API
                if not st.session_state.returns.empty:
                    st.session_state.sp500_returns = st.session_state.returns.mean(axis=1)

# Mensaje de estado en el Sidebar
if st.session_state.data_loaded:
    st.sidebar.success(f"📊 Base de datos activa: {len(st.session_state.selected_tickers)} activos.")
else:
    st.sidebar.info("💡 Haz clic en 'Ejecutar Análisis de Riesgo' para calcular las métricas.")

# ============================================================================
# 7. ASIGNACIÓN DINÁMICA DE SECTORES (Mapeo Forense para los Activos del Perú)
# ============================================================================
def obtener_sector_peru(ticker):
    ticker_upper = ticker.upper()
    if "BAP" in ticker_upper or "IFS" in ticker_upper or "CREDIC1" in ticker_upper or "BBVAC1" in ticker_upper:
        return "Financiero"
    elif "SCCO" in ticker_upper or "BVN" in ticker_upper or "VOLCABC1" in ticker_upper or "SPCC" in ticker_upper:
        return "Minero"
    else:
        return "Consumo/Industrial"

# ============================================================================
# 8. RENDERIZADO DE PESTAÑAS Y DASHBOARD PRINCIPAL
# ============================================================================
if st.session_state.data_loaded:
    # Creación de las 8 pestañas (incluyendo la nueva de IA)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Resumen", 
        "📊 Precios", 
        "⚠️ Riesgo",
        "📉 Drawdown", 
        "🔄 Correlaciones", 
        "🎯 Recomendaciones", 
        "🏢 Portfolio ROSTADINA",
        "🤖 Auditoría de Riesgo IA (SEM)"
    ])

    df_close = st.session_state.close_prices
    df_returns = st.session_state.returns
    bench_returns = st.session_state.sp500_returns

    # --- TAB 1: RESUMEN ---
    with tab1:
        st.header("📈 Resumen Ejecutivo del Mercado (Perú)")
        st.markdown("Métricas clave de los activos seleccionados.")
        
        # Calcular métricas básicas
        retorno_total = (df_close.iloc[-1] / df_close.iloc[0] - 1) * 100
        volatilidad_anual = df_returns.std() * np.sqrt(252) * 100
        
        cols = st.columns(min(len(df_close.columns), 4))
        for i, col in enumerate(df_close.columns):
            if i < 4:
                with cols[i]:
                    st.metric(
                        label=f"{col} - Retorno", 
                        value=f"{df_close[col].iloc[-1]:.2f}", 
                        delta=f"{retorno_total[col]:.2f}%"
                    )
                    st.caption(f"Volatilidad: {volatilidad_anual[col]:.2f}%")

    # --- TAB 2: PRECIOS ---
    with tab2:
        st.header("📊 Evolución Histórica de Precios")
        fig_precios = px.line(df_close, title="Precios de Cierre (Base 100 recomendada para comparación)")
        fig_precios.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_precios, use_container_width=True)

    # --- TAB 3: RIESGO ---
    with tab3:
        st.header("⚠️ Análisis de Volatilidad y Riesgo (VaR)")
        # VaR Histórico al 95%
        var_95 = df_returns.quantile(0.05) * 100
        st.write("Value at Risk (VaR) al 95% de confianza (Pérdida máxima diaria esperada):")
        st.bar_chart(var_95 * -1) # Mostrar en positivo para el gráfico

    # --- TAB 4: DRAWDOWN ---
    with tab4:
        st.header("📉 Análisis de Drawdown (Caídas Máximas)")
        roll_max = df_close.cummax()
        drawdown = (df_close / roll_max) - 1
        fig_dd = px.line(drawdown * 100, title="Drawdown Porcentual (%)")
        fig_dd.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_dd, use_container_width=True)

    # --- TAB 5: CORRELACIONES ---
    with tab5:
        st.header("🔄 Matriz de Correlaciones")
        st.markdown("Evalúa cómo se mueven los activos peruanos entre sí.")
        corr_matrix = df_returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 6: RECOMENDACIONES ---
    with tab6:
        st.header("🎯 Sugerencias de Posicionamiento")
        st.markdown("Basado puramente en el desempeño histórico reciente (Sharpe Ratio).")
        # Sharpe Ratio simplificado (Risk free = 0 para simulación)
        sharpe = (df_returns.mean() / df_returns.std()) * np.sqrt(252)
        st.dataframe(sharpe.rename("Sharpe Ratio").sort_values(ascending=False))

    # --- TAB 7: PORTFOLIO ROSTADINA ---
    with tab7:
        st.header("🏢 Configuración de Portfolio ROSTADINA")
        st.markdown("Estructura tu propia cartera de inversión local.")
        # Simulación de pesos equitativos
        pesos = np.ones(len(df_close.columns)) / len(df_close.columns)
        retorno_port = (df_returns * pesos).sum(axis=1)
        retorno_acum_port = (1 + retorno_port).cumprod()
        
        fig_port = px.line(retorno_acum_port, title="Crecimiento de S/ 1.00 Invertido (Equiponderado)")
        fig_port.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_port, use_container_width=True)

    # ============================================================================
    # --- TAB 8: MOTOR DE AUDITORÍA DE RIESGO CON INTELIGENCIA ARTIFICIAL (SEM) ---
    # ============================================================================
    with tab8:
        st.header("🤖 ROSTADINA AI — Especificación Multi-Factor de Riesgo")
        st.markdown("Evaluación algorítmica tridimensional que procesa simultáneamente variables macroeconómicas globales, microeconómicas corporativas y la coyuntura político-social de Perú.")
        
        # 1. Base de Ingesta del Modelo
        st.subheader("📰 Ingesta Actual de Hechos de Importancia y Noticias (Perú)")
        
        noticias_actuales = [
            {"texto": "Global copper prices hit record high amid supply chain disruptions.", "capa": "macro"},
            {"texto": "Federal Reserve signals interest rate cuts for the upcoming quarter.", "capa": "macro"},
            {"texto": "Nuevos conflictos sociales paralizan por completo el corredor minero en el sur peruano.", "capa": "social"},
            {"texto": "Incertidumbre política genera ruido institucional en el Congreso de la República.", "capa": "social"},
            {"texto": "Credicorp reports record high profits and revenue growth for the quarter.", "capa": "micro"}
        ]
        
        with st.expander("📋 Ver fuentes documentales precargadas en el Pipeline"):
            df_noticias = pd.DataFrame(noticias_actuales)
            df_noticias.columns = ["Titular / Hecho de Importancia", "Capa Analítica"]
            st.table(df_noticias)
            
        # 2. Ejecución del Core de IA
        if st.button("🔄 Lanzar Auditoría de Sentimiento & Estrés Colectivo", key="btn_ia_sem"):
            with st.spinner("Iniciando Transformers... Traduciendo al inglés técnico e infiriendo mediante redes neuronales FinBERT..."):
                
                # Llamada limpia a la función que definimos en la PARTE 1
                SEM, I_macro, I_social, I_micro, lista_noticias = procesar_riesgo_ia(noticias_actuales)
                
                st.markdown("### 📊 Indicadores Sectoriales Ponderados")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sub-Índice Macro (Global - Peso 40%)", f"{I_macro:+.4f}")
                col2.metric("Sub-Índice Social (Perú - Peso 35%)", f"{I_social:+.4f}")
                col3.metric("Sub-Índice Micro (BVL - Peso 25%)", f"{I_micro:+.4f}")
                
                st.markdown("---")
                
                col_grafico, col_consejo = st.columns([2, 3])
                
                with col_grafico:
                    st.markdown("<h4 style='text-align: center;'>Ubicación del SEM en la Curva de Estrés</h4>", unsafe_allow_html=True)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=SEM,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
                            'bar': {'color': "#2c3e50"},
                            'bgcolor': "white",
                            'borderwidth': 1,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 25], 'color': '#2ecc71'},   
                                {'range': [25, 50], 'color': '#f1c40f'},  
                                {'range': [50, 75], 'color': '#e67e22'},  
                                {'range': [75, 100], 'color': '#e74c3c'}  
                            ],
                        }
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=10, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with col_consejo:
                    st.markdown("<h4>Métrica de Intervalo y Guía de Decisiones Directivas</h4>", unsafe_allow_html=True)
                    if 0 <= SEM <= 25:
                        st.success("### 🟢 RIESGO COMPRENSIVO BAJO")
                        consejo = "**Consejo Corporativo:** Fase de expansión de portafolios. Maximizar la exposición en activos cíclicos y renta variable de la BVL."
                    elif 25 < SEM <= 50:
                        st.warning("### 🟡 RIESGO MODERADO RESILIENTE")
                        consejo = "**Consejo Corporativo:** Mantener posiciones. Se sugiere estructurar coberturas cambiarias parciales (Forward PEN/USD) frente a volatilidad interna."
                    elif 50 < SEM <= 75:
                        st.error("### 🟠 RIESGO COMPORTAMENTAL ALTO")
                        consejo = "**Consejo Corporativo:** Reducir duración de portafolios de renta fija. Direccionar flujos excedentes hacia activos defensivos de alta liquidez."
                    else:
                        st.error("### 🔴 ALERTA SISTÉMICA / ESTRÉS CRÍTICO")
                        consejo = "**Consejo Corporativo:** Preservación absoluta del capital corporativo. Liquidar posiciones de Beta elevado. Refugio en Dólares."
                    
                    st.write(consejo)
                    
                st.markdown("---")
                
                with st.expander("🔍 Ver Auditoría de Inferencia y Confianza por Titular"):
                    for n in lista_noticias:
                        badge = "🟢" if n['sentimiento'] == 'POSITIVE' else ("🔴" if n['sentimiento'] == 'NEGATIVE' else "⚪")
                        st.markdown(f"**{badge} [{n['capa']}]** | {n['noticia']}")
                        st.caption(f"Inferencia FinBERT: *{n['sentimiento']}* | Confianza del Modelo: **{n['confianza']:.2%}**")
                        
                with st.expander("🧮 Ver Especificación de Ecuaciones y Ponderaciones"):
                    st.latex(r"S_{raw} = 0.40 \cdot I_{macro} + 0.35 \cdot I_{social} + 0.25 \cdot I_{micro}")
                    st.latex(r"SEM = 50 \times (1 - S_{raw})")
                    
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📥 Exportar Informe (PDF)", key="btn_pdf_premium"):
                    st.info("⭐ Función Premium. Conecta con Stripe para habilitar descargas ejecutivas automatizadas para Directorios.")

else:
    # Mensaje principal cuando aún no se han cargado datos
    st.info("👋 Bienvenido al Monitor Forense. Configura los activos en la barra lateral izquierda y haz clic en 'Ejecutar Análisis de Riesgo' para comenzar.")

# ============================================================================
# PIE DE PÁGINA Y DISCLAIMER LEGAL OBLIGATORIO
# ============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    """
    <div style="background-color: #f8f9fa; padding: 15px; border-left: 5px solid #ffc107; border-radius: 4px;">
        <p style="margin: 0; font-size: 0.85rem; color: #6c757d; line-height: 1.4;">
            <strong>⚠️ AVISO DE EXENCIÓN DE RESPONSABILIDAD (DISCLAIMER LEGAL):</strong> 
            Los reportes, análisis econométricos y scores generados por <strong>ROSTADINA AI</strong> 
            son herramientas puramente algorítmicas con fines académicos y de simulación. 
            No constituyen asesoría financiera vinculante ni recomendaciones de compra/venta en la BVL.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="text-align: center; color: #a0a0a0; font-size: 0.8rem; margin-top: 20px;">
        <strong>ROSTADINA EIRL - Inteligencia Financiera Corporativa</strong> | 
        Última actualización: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>
    """, 
    unsafe_allow_html=True
)
