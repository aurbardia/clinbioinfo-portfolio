#!/usr/bin/env python3
"""
Dashboard Interactivo - Vigilancia Epidemiológica
Demo para ClinBioinfo - Visualización en Tiempo Real
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Configuración de página
st.set_page_config(
    page_title="ClinBioinfo Dashboard",
    page_icon="🧬",
    layout="wide"
)

@st.cache_data
def generar_datos_dashboard():
    """Genera datos sintéticos para el dashboard"""
    
    # Datos temporales (últimos 30 días)
    fechas = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='D'
    )
    
    hospitales = ["H. Virgen del Rocío", "H. Reina Sofía", "H. Puerta del Mar"]
    
    # Generar casos diarios
    casos_data = []
    for fecha in fechas:
        for hospital in hospitales:
            casos = np.random.poisson(15)  # Media de 15 casos por día
            casos_data.append({
                'fecha': fecha,
                'hospital': hospital,
                'casos_nuevos': casos,
                'tasa_positividad': np.random.uniform(0.05, 0.20)
            })
    
    df_casos = pd.DataFrame(casos_data)
    
    # Datos de variantes
    variantes = ['Alpha', 'Delta', 'Omicron', 'BA.1', 'BA.2']
    variantes_data = []
    
    for _ in range(200):
        variantes_data.append({
            'variante': np.random.choice(variantes),
            'hospital': np.random.choice(hospitales),
            'fecha': np.random.choice(fechas[-14:])  # Últimas 2 semanas
        })
    
    df_variantes = pd.DataFrame(variantes_data)
    
    return df_casos, df_variantes

def main():
    """Función principal del dashboard"""
    
    # Título
    st.title("🧬 ClinBioinfo Dashboard")
    st.markdown("**Plataforma Andaluza de Medicina Computacional**")
    
    # Cargar datos
    df_casos, df_variantes = generar_datos_dashboard()
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_casos = df_casos['casos_nuevos'].sum()
        st.metric("📊 Total Casos", f"{total_casos:,}")
    
    with col2:
        casos_ultima_semana = df_casos[df_casos['fecha'] >= datetime.now() - timedelta(days=7)]['casos_nuevos'].sum()
        st.metric("📈 Última Semana", f"{casos_ultima_semana:,}")
    
    with col3:
        tasa_promedio = df_casos['tasa_positividad'].mean()
        st.metric("🎯 Tasa Positividad", f"{tasa_promedio:.1%}")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Evolución Temporal")
        casos_diarios = df_casos.groupby('fecha')['casos_nuevos'].sum().reset_index()
        fig_temporal = px.line(casos_diarios, x='fecha', y='casos_nuevos', 
                              title="Casos Nuevos por Día")
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Casos por Hospital")
        casos_hospital = df_casos.groupby('hospital')['casos_nuevos'].sum().reset_index()
        fig_hospital = px.bar(casos_hospital, x='hospital', y='casos_nuevos',
                             title="Total Casos por Hospital")
        st.plotly_chart(fig_hospital, use_container_width=True)
    
    # Análisis de variantes
    st.subheader("🦠 Distribución de Variantes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        variantes_count = df_variantes['variante'].value_counts().reset_index()
        fig_variantes = px.pie(variantes_count, values='count', names='variante',
                              title="Distribución de Variantes")
        st.plotly_chart(fig_variantes, use_container_width=True)
    
    with col2:
        st.subheader("📋 Datos Recientes")
        datos_recientes = df_casos.tail(10)[['fecha', 'hospital', 'casos_nuevos', 'tasa_positividad']]
        st.dataframe(datos_recientes, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Última actualización**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()