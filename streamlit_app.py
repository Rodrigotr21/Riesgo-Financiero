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

# ============================================================================
# CONFIGURACIÓN INICIAL - DEBE SER LO PRIMERO
# ============================================================================
st.set_page_config(
    page_title="Dashboard de Riesgo Financiero - ROSTADINA EIRL",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# INICIALIZAR SESSION_STATE - CRÍTICO PARA STREAMLIT
# ============================================================================
# Estado básico de datos
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'close_prices' not in st.session_state:
    st.session_state.close_prices = pd.DataFrame()
if 'returns' not in st.session_state:
    st.session_state.returns = pd.DataFrame()
if 'sp500_returns' not in st.session_state:
    st.session_state.sp500_returns = pd.Series()

# Estado para portfolio (opcional)
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
# ESTILOS CSS PERSONALIZADOS PARA ROSTADINA - CORREGIDOS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }
    .rostadina-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* ALERTAS CORREGIDAS - NEGRO CON LETRA BLANCA */
    .alert-box {
        background-color: #2c3e50;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: white !important;
    }
    .alert-box strong {
        color: #ecf0f1 !important;
    }
    .alert-box em {
        color: #bdc3c7 !important;
    }
    .success-box {
        background-color: #27ae60;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: white !important;
    }
    .success-box strong {
        color: white !important;
    }
    /* PORTFOLIO BOX CORREGIDO - AZUL OSCURO CON LETRA BLANCA */
    .portfolio-box {
        background-color: #1E3A8A;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        color: white !important;
    }
    .portfolio-box h3 {
        color: white !important;
    }
    .portfolio-box p {
        color: #ecf0f1 !important;
    }
    .warning-box {
        background-color: #f39c12;
        border-left: 5px solid #e67e22;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: #2c3e50 !important;
    }
    .warning-box strong {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
def descargar_datos_simples(tickers):
    """Descarga datos de forma simple"""
    try:
        # Filtrar tickers válidos
        tickers_validos = []
        tickers_no_disponibles = []
        
        for ticker in tickers:
            try:
                # Verificar si el ticker existe
                data_test = yf.download(ticker, period="5d", progress=False)
                if not data_test.empty and 'Close' in data_test.columns:
                    tickers_validos.append(ticker)
                else:
                    tickers_no_disponibles.append(ticker)
            except Exception as e:
                tickers_no_disponibles.append(ticker)
        
        if tickers_no_disponibles:
            st.sidebar.warning(f"Tickers no disponibles: {', '.join(tickers_no_disponibles)}")
        
        if not tickers_validos:
            st.error("Ningún ticker está disponible. Por favor selecciona otros.")
            return pd.DataFrame()
        
        data = yf.download(
            tickers_validos,
            start=st.session_state.fecha_inicio,
            end=st.session_state.fecha_fin,
            progress=False,
            group_by='ticker'
        )
        
        if data.empty:
            return pd.DataFrame()
        
        # Extraer precios de cierre
        if data.columns.nlevels > 1:
            close_prices = data.xs('Close', axis=1, level=1)
        else:
            close_prices = data[['Close']] if 'Close' in data.columns else data
        
        return close_prices.dropna(how='all')
    
    except Exception as e:
        st.error(f"Error descargando datos: {str(e)[:100]}")
        return pd.DataFrame()

def calcular_drawdown(precios):
    """Calcula drawdown"""
    try:
        ret_acum = (1 + precios.pct_change().fillna(0)).cumprod()
        max_rodante = ret_acum.expanding().max()
        drawdown = (ret_acum - max_rodante) / max_rodante
        return drawdown
    except:
        return pd.Series()

def calcular_metricas_portfolio():
    """Calcula métricas del portfolio ROSTADINA (si está configurado)"""
    if not st.session_state.data_loaded or not st.session_state.portfolio_configured:
        return None
    
    try:
        returns = st.session_state.returns
        portfolio_dist = st.session_state.portfolio_dist
        capital_total = st.session_state.capital_total
        
        # Verificar distribución válida
        if not portfolio_dist or len(portfolio_dist) != len(returns.columns):
            return None
        
        # Calcular retorno ponderado del portfolio
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        
        for empresa, peso in portfolio_dist.items():
            if empresa in returns.columns:
                peso_decimal = peso / 100
                portfolio_returns = portfolio_returns.add(returns[empresa] * peso_decimal, fill_value=0)
        
        # Métricas del portfolio
        vol_anual = portfolio_returns.std() * np.sqrt(252) * 100
        ret_anual = ((1 + portfolio_returns.mean())**252 - 1) * 100
        
        # VaR del portfolio
        var_95 = portfolio_returns.quantile(0.05) * 100
        
        # Sharpe Ratio
        sharpe = np.nan
        if vol_anual > 0:
            sharpe = (ret_anual/100 - st.session_state.tasa_riesgo) / (vol_anual/100)
        
        # Pérdida máxima diaria
        max_perdida = portfolio_returns.min() * 100
        
        # Valor en riesgo absoluto
        var_absoluto = (capital_total * abs(var_95)) / 100
        
        return {
            "vol_anual": vol_anual,
            "ret_anual": ret_anual,
            "var_95": var_95,
            "sharpe": sharpe,
            "max_perdida": max_perdida,
            "var_absoluto": var_absoluto,
            "portfolio_returns": portfolio_returns,
            "capital_total": capital_total
        }
    
    except Exception as e:
        st.error(f"Error calculando métricas del portfolio: {e}")
        return None

def sistema_alertas_generales():
    """Sistema de alertas automáticas generales (no dependen de portfolio)"""
    if not st.session_state.data_loaded:
        return []
    
    alertas = []
    returns = st.session_state.returns
    close_prices = st.session_state.close_prices
    
    # Umbrales de alerta
    UMBRAL_VOL_ALTA = 35
    UMBRAL_VOL_MUY_ALTA = 45
    UMBRAL_DRAWDOWN = -0.20
    UMBRAL_CORRELACION = 0.85
    
    # 1. Alertas por volatilidad
    for empresa in returns.columns:
        vol_actual = returns[empresa].std() * np.sqrt(252) * 100
        
        if vol_actual > UMBRAL_VOL_MUY_ALTA:
            alertas.append({
                "nivel": "🔴 CRÍTICO",
                "mensaje": f"<strong>{empresa}</strong>: Volatilidad extremadamente alta ({vol_actual:.1f}%)",
                "recomendacion": "Considerar reducir posición o implementar cobertura"
            })
        elif vol_actual > UMBRAL_VOL_ALTA:
            alertas.append({
                "nivel": "🟡 ADVERTENCIA",
                "mensaje": f"<strong>{empresa}</strong>: Volatilidad elevada ({vol_actual:.1f}%)",
                "recomendacion": "Monitorear de cerca"
            })
    
    # 2. Alertas por drawdown
    for empresa in close_prices.columns:
        dd = calcular_drawdown(close_prices[empresa])
        if not dd.empty and dd.min() < UMBRAL_DRAWDOWN:
            alertas.append({
                "nivel": "🔴 CRÍTICO",
                "mensaje": f"<strong>{empresa}</strong>: Drawdown histórico > 20% ({dd.min()*100:.1f}%)",
                "recomendacion": "Revisar estrategia de salida"
            })
    
    # 3. Alertas por correlación excesiva
    if len(returns.columns) > 1:
        corr_matrix = returns.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > UMBRAL_CORRELACION:
                    alertas.append({
                        "nivel": "🟡 ADVERTENCIA",
                        "mensaje": f"Alta correlación entre <strong>{corr_matrix.columns[i]}</strong> y <strong>{corr_matrix.columns[j]}</strong> ({corr_matrix.iloc[i, j]:.2f})",
                        "recomendacion": "Considerar diversificar con activos no correlacionados"
                    })
    
    return alertas

# ============================================================================
# BANNER SUPERIOR CON LOGO ROSTADINA
# ============================================================================
col_logo, col_titulo = st.columns([1, 3])

with col_logo:
    # RUTA ABSOLUTA PARA EVITAR PROBLEMAS
    ruta_logo = "/Users/rodrigoluisyauliblas/Python/ROSTADINA/Logo_ROSTADINA.jpeg"
    
    # Intentar cargar el logo, si falla mostrar texto alternativo
    try:
        st.image(ruta_logo, width=150)
    except Exception as e:
        # Mostrar placeholder si no hay logo
        st.markdown('<div style="text-align: center; padding: 40px; background: #1E3A8A; color: white; border-radius: 10px;">', unsafe_allow_html=True)
        st.markdown('<h2>🏢</h2>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.5rem; font-weight: bold;">ROSTADINA</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1rem;">EIRL</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.error(f"Logo no encontrado: {ruta_logo}")
    
    st.markdown('<div style="font-size: 1.2rem; color: #1E3A8A; font-weight: bold; text-align: center;">ROSTADINA EIRL</div>', unsafe_allow_html=True)

with col_titulo:
    st.markdown('<h1 class="main-header">📊 Dashboard de Análisis Financiero</h1>', unsafe_allow_html=True)
    st.markdown('<div class="rostadina-banner">Análisis de Mercado Peruano - Datos en Tiempo Real</div>', unsafe_allow_html=True)

# ============================================================================
# BARRA LATERAL SIMPLIFICADA (NO REQUIERE PORTFOLIO)
# ============================================================================

# En el sidebar
with st.sidebar:
    st.markdown("### 🏢 **ROSTADINA EIRL**")
    st.markdown("---")
    
    # Contenedor para logo + texto en sidebar
    st.markdown('<div style="text-align: center; margin-bottom: 20px;">', unsafe_allow_html=True)
    
    # Logo pequeño arriba
    try:
        st.image(ruta_logo, width=70)
    except:
        st.markdown('<div style="font-size: 2rem; margin-bottom: 5px;">🏢</div>', unsafe_allow_html=True)
    
    # Texto pequeño debajo
    st.markdown('<div style="font-size: 0.9rem; color: #1E3A8A; font-weight: bold;">ROSTADINA EIRL</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Configuración General")
    
    # Período
    fecha_inicio = st.date_input(
        "Fecha de inicio",
        value=datetime(2020, 1, 1)
    )
    
    fecha_fin = st.date_input(
        "Fecha de fin",
        value=datetime.today()
    )
    
    # LISTA COMPLETA DE EMPRESAS PERUANAS
    st.markdown("### 📊 Empresas Peruanas")
    
    empresas_peruanas = {
        "Todos": ["BAP", "SCCO", "BVN", "IFS", "CREDITC1", "INTERBC1", 
                 "NEXA", "SID", "RIMC", "CORAREC1", "ALICORC1", 
                 "ENGIEC1", "LUSURC1", "CVERDEC1", "NESTLEP", 
                 "VOLCABC1", "CASAGRC1"],
        "Financieras": ["BAP", "IFS", "CREDITC1", "INTERBC1"],
        "Minera": ["SCCO", "BVN", "NEXA", "SID", "RIMC"],
        "Industrial": ["CORAREC1", "ALICORC1"],
        "Energía": ["ENGIEC1", "LUSURC1"],
        "Consumo": ["CVERDEC1", "NESTLEP"],
        "Diversas": ["VOLCABC1", "CASAGRC1"]
    }
    
    # Selección por sector
    sector_seleccionado = st.selectbox(
        "Filtrar por sector:",
        list(empresas_peruanas.keys()),
        index=0  # "Todos" por defecto
    )
    
    # Actualizar las opciones según el sector seleccionado
    opciones_tickers = empresas_peruanas[sector_seleccionado]
    
    # Determinar valores por defecto que estén en las opciones actuales
    default_tickers = []
    for ticker in st.session_state.selected_tickers:
        if ticker in opciones_tickers:
            default_tickers.append(ticker)
    
    # Si no hay defaults válidos, usar los primeros disponibles
    if not default_tickers and opciones_tickers:
        default_tickers = opciones_tickers[:min(3, len(opciones_tickers))]
    
    tickers_seleccionados = st.multiselect(
        "Selecciona empresas para análisis:",
        opciones_tickers,
        default=default_tickers
    )
    
    # Guardar selección actual
    st.session_state.selected_tickers = tickers_seleccionados
    st.session_state.selected_sector = sector_seleccionado
    
    if not tickers_seleccionados:
        st.warning("Selecciona al menos una empresa")
        st.stop()
    
    # Parámetros generales
    st.markdown("### 🎯 Parámetros de Riesgo")
    nivel_confianza = st.slider("Confianza VaR (%)", 90, 99, 95)
    tasa_riesgo = st.slider("Tasa libre riesgo (%)", 1.0, 10.0, 4.0) / 100
    
    # Botón para cargar datos GENERALES
    if st.button("📥 Cargar Datos de Mercado", type="primary", use_container_width=True):
        st.session_state.tickers = tickers_seleccionados
        st.session_state.fecha_inicio = fecha_inicio
        st.session_state.fecha_fin = fecha_fin
        st.session_state.nivel_confianza = nivel_confianza
        st.session_state.tasa_riesgo = tasa_riesgo
        
        # Resetear datos cargados para forzar recarga
        st.session_state.data_loaded = False
        st.session_state.portfolio_configured = False  # Portfolio NO requerido
        st.rerun()

# ============================================================================
# PROCESAMIENTO DE DATOS (INDEPENDIENTE DE PORTFOLIO)
# ============================================================================
if 'tickers' in st.session_state and st.session_state.tickers:
    if not st.session_state.data_loaded:
        with st.spinner("Descargando datos de mercado..."):
            # Descargar datos
            close_prices = descargar_datos_simples(st.session_state.tickers)
            
            if close_prices.empty:
                st.error("No se pudieron descargar datos. Verifica los tickers y tu conexión a internet.")
                st.stop()
            
            # Calcular retornos
            returns = close_prices.pct_change().dropna()
            
            if returns.empty:
                st.error("No hay suficientes datos para análisis.")
                st.stop()
            
            # Descargar S&P 500 (opcional)
            try:
                sp500 = yf.download('^GSPC', 
                                   start=st.session_state.fecha_inicio,
                                   end=st.session_state.fecha_fin,
                                   progress=False)
                
                if not sp500.empty:
                    if 'Close' in sp500.columns:
                        sp500_close = sp500['Close']
                    else:
                        sp500_close = sp500.iloc[:, 0]
                    
                    sp500_returns = sp500_close.pct_change().dropna()
                else:
                    sp500_returns = pd.Series()
            except:
                sp500_returns = pd.Series()
            
            # Guardar en session_state
            st.session_state.close_prices = close_prices
            st.session_state.returns = returns
            st.session_state.sp500_returns = sp500_returns
            st.session_state.data_loaded = True
            
            st.success(f"✅ Datos de mercado cargados: {len(close_prices)} días, {len(close_prices.columns)} empresas")

# ============================================================================
# VERIFICAR SI HAY DATOS DE MERCADO
# ============================================================================
if not st.session_state.data_loaded:
    st.info("👈 **Selecciona empresas y haz clic en 'Cargar Datos de Mercado' para comenzar**")
    st.stop()

# Obtener datos (siempre disponibles)
close_prices = st.session_state.close_prices
returns = st.session_state.returns

# ============================================================================
# NOTAS E INSTRUCCIONES PARA EL USUARIO
# ============================================================================
with st.expander("📚 **GUÍA RÁPIDA DE USO - ROSTADINA EIRL**", expanded=True):
    st.markdown("""
    ### 🎯 **OBJETIVO DEL DASHBOARD**
    Esta herramienta permite analizar el riesgo financiero de empresas peruanas listadas en bolsa 
    y configurar portfolios de inversión personalizados para **ROSTADINA EIRL**.
    
    **Nota**: Estos datos son de actualización diaria, es decir, los precios a ver no son de la última hora, sino de la hora
    de cierre del día anterior.
    
    ### 📋 **PASOS RECOMENDADOS:**
    
    1. **📊 SELECCIÓN DE EMPRESAS** (Barra lateral)
       - Selecciona el sector y las empresas a analizar
       - Haz clic en **"Cargar Datos de Mercado"**
    
    2. **📈 ANÁLISIS GENERAL** (Pestañas 1-6)
       - **Resumen**: Métricas clave del mercado
       - **Precios**: Evolución histórica por empresa
       - **Riesgo**: Comparativa detallada de métricas
       - **Drawdown**: Máximas caídas históricas
       - **Correlaciones**: Diversificación del portfolio
       - **Recomendaciones**: Estrategias por perfil
    
    3. **🏢 CONFIGURACIÓN DE PORTFOLIO** (Pestaña 7 - Opcional)
       - Define capital total a invertir
       - Configura distribución porcentual
       - Activa análisis específico de ROSTADINA
    
    ### ⚠️ **CONSIDERACIONES IMPORTANTES:**
    - Los datos provienen de **Yahoo Finance** en tiempo real
    - Las métricas de riesgo son **estimaciones históricas**
    - **No constituyen recomendación de inversión**
    - Consulta con tu asesor financiero antes de tomar decisiones
    """)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        ### 📞 **CONTACTO Y SOPORTE**
        **ROSTADINA EIRL**  
        📧 contacto@rostadina.com  
        📱 +51 999 888 777  
        🏢 Av. Principal 123, Lima
        """)
    
    with col_info2:
        st.markdown("""
        ### 🕒 **ACTUALIZACIÓN DE DATOS**
        - Datos actualizados diariamente
        - Histórico desde 2020
        - Horario de actualización: 9:00 AM (PET)
        - Última actualización: {}
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M")))

# ============================================================================
# SISTEMA DE ALERTAS GENERALES (SIEMPRE VISIBLE) - CORREGIDO
# ============================================================================
alertas = sistema_alertas_generales()

if alertas:
    st.markdown("### ⚠️ **Alertas del Mercado**")
    
    # Separar alertas por nivel
    alertas_criticas = [a for a in alertas if a["nivel"] == "🔴 CRÍTICO"]
    alertas_advertencia = [a for a in alertas if a["nivel"] == "🟡 ADVERTENCIA"]
    
    if alertas_criticas:
        for alerta in alertas_criticas:
            st.markdown(f"""
            <div class="alert-box">
                <strong>{alerta["nivel"]}</strong><br>
                {alerta["mensaje"]}<br>
                <em>Recomendación: {alerta["recomendacion"]}</em>
            </div>
            """, unsafe_allow_html=True)
    
    if alertas_advertencia:
        for alerta in alertas_advertencia:
            st.markdown(f"""
            <div class="alert-box">
                <strong>{alerta["nivel"]}</strong><br>
                {alerta["mensaje"]}<br>
                <em>Recomendación: {alerta["recomendacion"]}</em>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ ESTADO DEL MERCADO ESTABLE</strong><br>
        No se detectaron alertas críticas en las empresas seleccionadas
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TABS PRINCIPALES (INDEPENDIENTES DE PORTFOLIO)
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Resumen", 
    "📊 Precios", 
    "⚠️ Riesgo",
    "📉 Drawdown", 
    "🔄 Correlaciones", 
    "🎯 Recomendaciones",
    "🏢 Portfolio ROSTADINA"  # PORTFOLIO OPCIONAL EN PESTAÑA SEPARADA
])

# ============================================================================
# TAB 1: RESUMEN GENERAL (SIN PORTFOLIO)
# ============================================================================
with tab1:
    st.header("📊 Resumen Ejecutivo del Mercado")
    
    # Métricas generales del mercado
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        vol = returns.std().mean() * np.sqrt(252) * 100
        st.metric("Volatilidad Promedio", f"{vol:.1f}%", 
                 help="Volatilidad anualizada promedio de las empresas seleccionadas")
    
    with col2:
        ret = ((1 + returns.mean())**252 - 1).mean() * 100
        st.metric("Retorno Promedio Anual", f"{ret:.1f}%",
                 help="Retorno anualizado promedio esperado")
    
    with col3:
        var = returns.quantile(0.05).mean() * 100
        st.metric("VaR 95% Diario", f"{abs(var):.1f}%",
                 help="Pérdida máxima diaria con 95% confianza (promedio)")
    
    with col4:
        sharpe = ((returns.mean() * 252 - st.session_state.tasa_riesgo) / 
                 (returns.std() * np.sqrt(252))).mean()
        st.metric("Sharpe Ratio Promedio", f"{sharpe:.2f}",
                 help="Retorno ajustado por riesgo (mayor es mejor)")
    
    st.markdown("---")
    
    # Top empresas por diferentes métricas
    st.subheader("🏆 Ranking de Empresas")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.markdown("**📈 Mejor Retorno Anual**")
        retornos_anuales = ((1 + returns.mean())**252 - 1) * 100
        top_retorno = retornos_anuales.sort_values(ascending=False).head(3)
        for i, (empresa, valor) in enumerate(top_retorno.items(), 1):
            st.markdown(f"{i}. **{empresa}**: {valor:.1f}%")
    
    with col_r2:
        st.markdown("**📉 Menor Volatilidad**")
        volatilidades = returns.std() * np.sqrt(252) * 100
        top_vol = volatilidades.sort_values().head(3)
        for i, (empresa, valor) in enumerate(top_vol.items(), 1):
            st.markdown(f"{i}. **{empresa}**: {valor:.1f}%")
    
    with col_r3:
        st.markdown("**🎯 Mejor Sharpe Ratio**")
        sharpe_ratios = (returns.mean() * 252 - st.session_state.tasa_riesgo) / (returns.std() * np.sqrt(252))
        top_sharpe = sharpe_ratios.sort_values(ascending=False).head(3)
        for i, (empresa, valor) in enumerate(top_sharpe.items(), 1):
            st.markdown(f"{i}. **{empresa}**: {valor:.2f}")
    
    st.markdown("---")
    
    # Gráfico de precios normalizados
    st.subheader("📈 Evolución de Precios Normalizados (Base 100)")
    
    if not close_prices.empty:
        precios_norm = close_prices / close_prices.iloc[0] * 100
        
        fig_precios = go.Figure()
        for col in precios_norm.columns:
            fig_precios.add_trace(go.Scatter(
                x=precios_norm.index,
                y=precios_norm[col],
                name=col,
                mode='lines',
                hovertemplate='<b>%{x}</b><br>%{y:.1f}'
            ))
        
        fig_precios.update_layout(
            height=400,
            title="Comparación de Performance (Base 100)",
            xaxis_title="Fecha",
            yaxis_title="Precio Normalizado",
            hovermode='x unified'
        )
        st.plotly_chart(fig_precios, use_container_width=True)

# ============================================================================
# TAB 2: PRECIOS (SIN CAMBIOS)
# ============================================================================
with tab2:
    st.markdown('<h2 class="sub-header">📊 Análisis de Precios y Retornos</h2>', unsafe_allow_html=True)
    
    if not close_prices.empty:
        # Selector de empresa
        empresa_seleccionada = st.selectbox(
            "Selecciona una empresa para análisis detallado:",
            options=close_prices.columns.tolist(),
            key="empresa_precios"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Precios de {empresa_seleccionada}")
            
            # Gráfico de precios históricos
            fig_precio = go.Figure()
            fig_precio.add_trace(go.Scatter(
                x=close_prices.index,
                y=close_prices[empresa_seleccionada],
                mode='lines',
                name='Precio',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ))
            
            # Media móvil de 30 días
            if len(close_prices) > 30:
                media_movil = close_prices[empresa_seleccionada].rolling(window=30).mean()
                fig_precio.add_trace(go.Scatter(
                    x=media_movil.index,
                    y=media_movil,
                    mode='lines',
                    name='Media Móvil 30 días',
                    line=dict(color='red', width=1, dash='dash')
                ))
            
            fig_precio.update_layout(
                height=400,
                title=f"Precio Histórico - {empresa_seleccionada}",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_precio, use_container_width=True)
        
        with col2:
            st.subheader("📊 Estadísticas")
            
            # Métricas básicas
            precio_actual = close_prices[empresa_seleccionada].iloc[-1]
            precio_max = close_prices[empresa_seleccionada].max()
            precio_min = close_prices[empresa_seleccionada].min()
            precio_prom = close_prices[empresa_seleccionada].mean()
            
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Precio Actual", f"${precio_actual:.2f}")
            st.metric("Máximo Histórico", f"${precio_max:.2f}")
            st.metric("Mínimo Histórico", f"${precio_min:.2f}")
            st.metric("Precio Promedio", f"${precio_prom:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Volatilidad diaria
            if empresa_seleccionada in returns.columns:
                vol_diaria = returns[empresa_seleccionada].std() * 100
                st.metric("Volatilidad Diaria", f"{vol_diaria:.2f}%")
        
        # Retornos diarios
        st.subheader("📈 Retornos Diarios")
        
        if empresa_seleccionada in returns.columns:
            fig_retornos = go.Figure()
            fig_retornos.add_trace(go.Scatter(
                x=returns.index,
                y=returns[empresa_seleccionada] * 100,
                mode='lines',
                name='Retornos',
                line=dict(color='green', width=1),
                hovertemplate='<b>%{x}</b><br>Retorno: %{y:.2f}%'
            ))
            
            fig_retornos.update_layout(
                height=300,
                title=f"Retornos Diarios - {empresa_seleccionada}",
                xaxis_title="Fecha",
                yaxis_title="Retorno (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_retornos, use_container_width=True)
            
            # Histograma de retornos
            st.subheader("📊 Distribución de Retornos")
            
            fig_hist = px.histogram(
                x=returns[empresa_seleccionada] * 100,
                nbins=50,
                title=f"Distribución de Retornos Diarios - {empresa_seleccionada}",
                labels={'x': 'Retorno Diario (%)', 'y': 'Frecuencia'}
            )
            
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# TAB 3: RIESGO (SIN CAMBIOS)
# ============================================================================
with tab3:
    st.markdown('<h2 class="sub-header">⚠️ Análisis Detallado de Riesgo</h2>', unsafe_allow_html=True)
    
    if not returns.empty:
        # Seleccionar métrica para comparar
        col1, col2 = st.columns(2)
        
        with col1:
            metrica_comparar = st.selectbox(
                "Comparar por:",
                ["Volatilidad Anual", "VaR 95%", "Retorno Anual", "Sharpe Ratio"],
                key="metrica_riesgo"
            )
        
        with col2:
            orden_comparacion = st.radio(
                "Orden:",
                ["Mayor a menor", "Menor a mayor"],
                horizontal=True,
                key="orden_riesgo"
            )
        
        # Calcular métricas para cada empresa
        datos_comparacion = []
        
        for empresa in returns.columns:
            # Volatilidad anual
            vol_anual = returns[empresa].std() * np.sqrt(252) * 100
            
            # Retorno anual
            ret_anual = ((1 + returns[empresa].mean())**252 - 1) * 100
            
            # VaR 95%
            var_95 = returns[empresa].quantile(0.05) * 100
            
            # Sharpe Ratio
            sharpe = np.nan
            if vol_anual > 0:
                sharpe = (ret_anual/100 - st.session_state.tasa_riesgo) / (vol_anual/100)
            
            # Seleccionar valor para comparación
            if metrica_comparar == "Volatilidad Anual":
                valor = vol_anual if not pd.isna(vol_anual) else 0
            elif metrica_comparar == "VaR 95%":
                valor = abs(var_95) if not pd.isna(var_95) else 0
            elif metrica_comparar == "Retorno Anual":
                valor = ret_anual if not pd.isna(ret_anual) else 0
            elif metrica_comparar == "Sharpe Ratio":
                valor = sharpe if not pd.isna(sharpe) else 0
            
            datos_comparacion.append({
                "Empresa": empresa,
                "Valor": valor,
                "Volatilidad Anual": vol_anual,
                "Retorno Anual": ret_anual,
                "VaR 95%": abs(var_95) if not pd.isna(var_95) else np.nan,
                "Sharpe Ratio": sharpe
            })
        
        # Convertir a DataFrame y ordenar
        df_comparacion = pd.DataFrame(datos_comparacion)
        ascendente = (orden_comparacion == "Menor a mayor")
        df_comparacion = df_comparacion.sort_values("Valor", ascending=ascendente)
        
        # Gráfico de barras comparativo
        fig_comparacion = go.Figure()
        
        # Definir colores basados en el valor
        colores = []
        for valor in df_comparacion["Valor"]:
            if pd.isna(valor):
                colores.append('gray')
            elif metrica_comparar in ["Volatilidad Anual", "VaR 95%"]:
                # Rojo para valores altos de riesgo
                if valor > df_comparacion["Valor"].quantile(0.75):
                    colores.append('red')
                elif valor > df_comparacion["Valor"].quantile(0.25):
                    colores.append('orange')
                else:
                    colores.append('green')
            else:
                # Verde para valores altos de retorno/Sharpe
                if valor > df_comparacion["Valor"].quantile(0.75):
                    colores.append('green')
                elif valor > df_comparacion["Valor"].quantile(0.25):
                    colores.append('orange')
                else:
                    colores.append('red')
        
        fig_comparacion.add_trace(go.Bar(
            x=df_comparacion["Empresa"],
            y=df_comparacion["Valor"],
            marker_color=colores,
            text=df_comparacion["Valor"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"),
            textposition='auto'
        ))
        
        fig_comparacion.update_layout(
            height=400,
            title=f"Comparación de {metrica_comparar}",
            xaxis_title="Empresa",
            yaxis_title=metrica_comparar
        )
        st.plotly_chart(fig_comparacion, use_container_width=True)
        
        # Tabla completa de métricas de riesgo
        st.subheader("📋 Tabla Completa de Métricas de Riesgo")
        
        tabla_riesgo = []
        for empresa in returns.columns:
            # Calcular todas las métricas
            vol_anual = returns[empresa].std() * np.sqrt(252) * 100
            ret_anual = ((1 + returns[empresa].mean())**252 - 1) * 100
            var_95 = returns[empresa].quantile(0.05) * 100
            
            # Sharpe Ratio
            sharpe = np.nan
            if vol_anual > 0:
                sharpe = (ret_anual/100 - st.session_state.tasa_riesgo) / (vol_anual/100)
            
            # Máximo y mínimo retorno diario
            max_ret = returns[empresa].max() * 100
            min_ret = returns[empresa].min() * 100
            
            # Clasificación de riesgo
            if pd.isna(vol_anual):
                riesgo = "N/A"
            elif vol_anual < 15:
                riesgo = "🟢 BAJO"
            elif vol_anual < 25:
                riesgo = "🟡 MODERADO"
            elif vol_anual < 35:
                riesgo = "🟠 ALTO"
            else:
                riesgo = "🔴 MUY ALTO"
            
            tabla_riesgo.append({
                "Empresa": empresa,
                "Volatilidad Anual": f"{vol_anual:.1f}%" if not pd.isna(vol_anual) else "N/A",
                "Retorno Anual": f"{ret_anual:.1f}%" if not pd.isna(ret_anual) else "N/A",
                "VaR 95% Diario": f"{abs(var_95):.1f}%" if not pd.isna(var_95) else "N/A",
                "Sharpe Ratio": f"{sharpe:.2f}" if not pd.isna(sharpe) else "N/A",
                "Máx Retorno Diario": f"{max_ret:.1f}%",
                "Mín Retorno Diario": f"{min_ret:.1f}%",
                "Nivel Riesgo": riesgo
            })
        
        df_tabla_riesgo = pd.DataFrame(tabla_riesgo)
        st.dataframe(df_tabla_riesgo, use_container_width=True, hide_index=True)
        
        # Volatilidad rolling
        st.subheader("📈 Volatilidad Rolling (60 días)")
        
        fig_vol_rolling = go.Figure()
        ventana = 60
        
        for empresa in returns.columns:
            if len(returns[empresa]) > ventana:
                vol_rolling = returns[empresa].rolling(window=ventana).std() * np.sqrt(252) * 100
                fig_vol_rolling.add_trace(go.Scatter(
                    x=vol_rolling.index,
                    y=vol_rolling,
                    mode='lines',
                    name=empresa,
                    hovertemplate='<b>%{x}</b><br>Volatilidad: %{y:.1f}%'
                ))
        
        if len(fig_vol_rolling.data) > 0:
            fig_vol_rolling.update_layout(
                height=400,
                title=f"Volatilidad Anualizada Rolling ({ventana} días)",
                xaxis_title="Fecha",
                yaxis_title="Volatilidad Anualizada (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_vol_rolling, use_container_width=True)
        else:
            st.info(f"Se necesitan más de {ventana} días de datos para calcular volatilidad rolling")

# ============================================================================
# TAB 4: DRAWDOWN (SIN CAMBIOS)
# ============================================================================
with tab4:
    st.header("📉 Análisis de Drawdown")
    
    # Gráfico de drawdown
    fig_dd = go.Figure()
    
    for empresa in close_prices.columns:
        dd_serie = calcular_drawdown(close_prices[empresa])
        if not dd_serie.empty:
            fig_dd.add_trace(go.Scatter(
                x=dd_serie.index,
                y=dd_serie * 100,
                name=empresa,
                mode='lines'
            ))
    
    fig_dd.update_layout(
        height=400,
        title="Drawdown Histórico",
        xaxis_title="Fecha",
        yaxis_title="Drawdown (%)"
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Máximos drawdowns
    st.subheader("Máximos Drawdowns")
    
    dd_data = []
    for empresa in close_prices.columns:
        dd_serie = calcular_drawdown(close_prices[empresa])
        if not dd_serie.empty:
            max_dd = dd_serie.min() * 100
            fecha_dd = dd_serie.idxmin()
            dd_data.append({
                "Empresa": empresa,
                "Drawdown Máx": f"{abs(max_dd):.1f}%",
                "Fecha Máx Drawdown": fecha_dd.strftime('%Y-%m-%d')
            })
    
    if dd_data:
        st.table(pd.DataFrame(dd_data))

# ============================================================================
# TAB 5: CORRELACIONES (SIN CAMBIOS)
# ============================================================================
with tab5:
    st.header("🔄 Análisis de Correlaciones")
    
    if len(returns.columns) > 1:
        # Matriz de correlación
        corr_matrix = returns.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            height=500,
            title="Matriz de Correlación",
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Análisis de diversificación
        st.subheader("📊 Análisis de Diversificación")
        
        if len(corr_matrix) > 1:
            # Calcular correlación promedio
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            corr_vals = corr_matrix.where(mask).stack()
            
            if len(corr_vals) > 0:
                avg_corr = corr_vals.mean()
                st.metric("Correlación Promedio", f"{avg_corr:.3f}")
                
                if avg_corr > 0.7:
                    st.error("⚠️ **Alta correlación** - Poca diversificación entre activos")
                    st.info("Recomendación: Considerar agregar activos de diferentes sectores")
                elif avg_corr > 0.4:
                    st.warning("⚠️ **Correlación moderada** - Diversificación aceptable")
                else:
                    st.success("✅ **Baja correlación** - Buena diversificación del portfolio")
    else:
        st.info("ℹ️ Se necesitan al menos 2 empresas para analizar correlaciones")

# ============================================================================
# TAB 6: RECOMENDACIONES (SIN CAMBIOS)
# ============================================================================
with tab6:
    st.header("🎯 Recomendaciones de Inversión")
    
    # Clasificación por riesgo
    st.subheader("📊 Clasificación por Nivel de Riesgo")
    
    riesgo_data = []
    for empresa in returns.columns:
        vol = returns[empresa].std() * np.sqrt(252) * 100
        ret_anual = ((1 + returns[empresa].mean())**252 - 1) * 100
        
        if pd.isna(vol) or vol < 15:
            riesgo = "BAJO"
            recomendacion = "Adecuada para perfiles conservadores"
        elif vol < 25:
            riesgo = "MODERADO"
            recomendacion = "Balance riesgo/retorno para perfiles moderados"
        elif vol < 35:
            riesgo = "ALTO"
            recomendacion = "Solo para perfiles agresivos con tolerancia al riesgo"
        else:
            riesgo = "MUY ALTO"
            recomendacion = "Alto riesgo - Considerar posiciones pequeñas"
        
        riesgo_data.append({
            "Empresa": empresa,
            "Volatilidad": f"{vol:.1f}%",
            "Retorno Anual": f"{ret_anual:.1f}%" if not pd.isna(ret_anual) else "N/A",
            "Nivel Riesgo": riesgo,
            "Recomendación": recomendacion
        })
    
    st.table(pd.DataFrame(riesgo_data))
    
    # Recomendaciones generales por perfil
    st.subheader("👤 Estrategias por Perfil de Inversor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("**🏦 Perfil Conservador**")
        st.markdown("- **Objetivo**: Preservar capital")
        st.markdown("- **Horizonte**: Largo plazo (5+ años)")
        st.markdown("- **Estrategia**: Buy & Hold")
        st.markdown("- **Empresas recomendadas**: Baja volatilidad, dividendos estables")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("**⚖️ Perfil Moderado**")
        st.markdown("- **Objetivo**: Crecimiento con control de riesgo")
        st.markdown("- **Horizonte**: Medio plazo (2-5 años)")
        st.markdown("- **Estrategia**: Diversificación sectorial")
        st.markdown("- **Empresas recomendadas**: Mix de crecimiento y valor")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("**🚀 Perfil Agresivo**")
        st.markdown("- **Objetivo**: Maximizar retornos")
        st.markdown("- **Horizonte**: Corto/medio plazo (1-3 años)")
        st.markdown("- **Estrategia**: Trading activo, momentum")
        st.markdown("- **Empresas recomendadas**: Alta volatilidad, alto crecimiento")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# TAB 7: PORTFOLIO ROSTADINA (OPCIONAL - CONFIGURACIÓN ESPECÍFICA)
# ============================================================================
with tab7:
    st.markdown('<h2 class="main-header">🏢 Portfolio de Inversión - ROSTADINA EIRL</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="portfolio-box">
    <h3>🎯 Configuración de Portfolio Específico</h3>
    <p>En esta sección puedes configurar un portfolio de inversión personalizado para ROSTADINA EIRL.
    Las demás pestañas funcionan independientemente de esta configuración.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar que hay empresas cargadas
    if not st.session_state.data_loaded:
        st.warning("Primero carga datos de mercado en la barra lateral")
        st.stop()
    
    # Configuración del portfolio (solo en esta pestaña)
    with st.expander("⚙️ **Configurar Portfolio ROSTADINA**", expanded=True):
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            capital_total = st.number_input(
                "💰 Capital Total a Invertir (USD)", 
                min_value=10000, 
                max_value=10000000, 
                value=1000000,
                step=50000,
                key="portfolio_capital"
            )
            st.session_state.capital_total = capital_total
        
        with col_c2:
            st.markdown("**📊 Empresas Disponibles**")
            empresas_disponibles = list(returns.columns)
            st.info(f"{len(empresas_disponibles)} empresas disponibles del análisis general")
        
        st.markdown("---")
        st.markdown("#### 🎯 Distribución del Portfolio")
        
        # Inicializar distribución si no existe
        if not st.session_state.portfolio_dist:
            empresas_disponibles = list(returns.columns)
            if empresas_disponibles:
                n_empresas = len(empresas_disponibles)
                for ticker in empresas_disponibles:
                    st.session_state.portfolio_dist[ticker] = round(100/n_empresas, 1)
        
        # Sliders para distribución
        empresas_disponibles = list(returns.columns)
        
        if not empresas_disponibles:
            st.error("No hay empresas disponibles para configurar el portfolio")
            st.stop()
        
        # REVISAR Y AJUSTAR LA SUMA INICIAL
        suma_actual = sum(st.session_state.portfolio_dist.get(ticker, 0) 
                         for ticker in empresas_disponibles)
        
        if abs(suma_actual - 100) > 0.1:
            # Normalizar a 100%
            for ticker in empresas_disponibles:
                if suma_actual > 0:
                    st.session_state.portfolio_dist[ticker] = (
                        st.session_state.portfolio_dist.get(ticker, 0) * 100 / suma_actual
                    )
                else:
                    st.session_state.portfolio_dist[ticker] = 100 / len(empresas_disponibles)
        
        # Mostrar sliders con validación
        suma_acumulada = 0
        
        for i, ticker in enumerate(empresas_disponibles):
            valor_actual = st.session_state.portfolio_dist.get(ticker, 0)
            
            # Calcular máximo permitido
            suma_restante = 100 - suma_acumulada
            
            # Asegurar que max_valor sea al menos 1 más que min_value
            max_valor = min(100, int(suma_restante + valor_actual))
            
            # CORRECCIÓN DEL ERROR: Asegurar que max_valor > 0
            if max_valor <= 0:
                max_valor = 1  # Valor mínimo válido
            
            nuevo_valor = st.slider(
                f"{ticker} (%)",
                0, 
                max_valor, 
                int(valor_actual),
                key=f"portfolio_slider_{ticker}"
            )
            
            st.session_state.portfolio_dist[ticker] = nuevo_valor
            suma_acumulada += nuevo_valor
        
        # Para la última empresa, ajustar automáticamente para sumar 100%
        if len(empresas_disponibles) > 0:
            ultima_empresa = empresas_disponibles[-1]
            suma_sin_ultima = sum(
                st.session_state.portfolio_dist.get(t, 0) 
                for t in empresas_disponibles[:-1]
            )
            
            if suma_sin_ultima < 100:
                st.session_state.portfolio_dist[ultima_empresa] = 100 - suma_sin_ultima
                st.info(f"{ultima_empresa} ajustado a: {100 - suma_sin_ultima:.1f}%")
        
        # Verificar y mostrar suma
        suma_final = sum(st.session_state.portfolio_dist.values())
        st.markdown(f"**Suma total:** {suma_final:.1f}%")
        
        if abs(suma_final - 100) > 0.1:
            st.error(f"⚠️ La distribución debe sumar exactamente 100% (actual: {suma_final:.1f}%)")
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**Sugerencia:** Ajusta los valores manualmente o usa el botón de distribución equitativa")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Botón para distribución equitativa
        if st.button("⚖️ Distribución Equitativa", use_container_width=True):
            n_empresas = len(empresas_disponibles)
            for ticker in empresas_disponibles:
                st.session_state.portfolio_dist[ticker] = round(100/n_empresas, 1)
            st.success("✅ Distribución equitativa aplicada")
            st.rerun()
        
        # Botón para activar portfolio
        if st.button("✅ Activar Análisis de Portfolio", type="primary", use_container_width=True):
            # Verificar suma exacta
            suma_final = sum(st.session_state.portfolio_dist.values())
            
            if abs(suma_final - 100) <= 0.1:
                st.session_state.portfolio_configured = True
                st.success("✅ Portfolio configurado exitosamente!")
                st.rerun()
            else:
                # Intentar ajustar automáticamente
                factor = 100 / suma_final
                for ticker in st.session_state.portfolio_dist:
                    st.session_state.portfolio_dist[ticker] *= factor
                
                st.session_state.portfolio_configured = True
                st.warning("⚠️ Distribución ajustada automáticamente para sumar 100%")
                st.rerun()

# ============================================================================
# PIE DE PÁGINA
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("**© 2024 ROSTADINA EIRL**")
st.sidebar.markdown("Dashboard de Análisis Financiero")
st.sidebar.markdown("Versión 2.2 | Datos: Yahoo Finance")
st.sidebar.markdown("**Uso:** Análisis general + Portfolio opcional")

# Footer principal
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <small>
        <strong>ROSTADINA EIRL - Dashboard de Análisis Financiero</strong> | 
        Datos históricos proporcionados por Yahoo Finance | 
        Última actualización: {date}
    </small>
</div>
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)