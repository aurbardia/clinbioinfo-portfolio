#!/usr/bin/env python3
"""
iRWD - Análisis de Datos Clínicos con SQL
Demo para ClinBioinfo - Real World Data Analysis
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

def crear_base_datos_sintetica():
    """Crea base de datos SQLite con datos clínicos sintéticos"""
    print(" Creando base de datos clínica sintética...")
    
    # Conectar a SQLite
    conn = sqlite3.connect("clinica_synthetic.db")
    cursor = conn.cursor()
    
    # Crear tabla pacientes
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pacientes (
        id INTEGER PRIMARY KEY,
        edad INTEGER,
        sexo TEXT,
        hospital TEXT,
        diagnostico TEXT,
        dias_hospitalizacion INTEGER
    )
    """)
    
    # Generar datos sintéticos
    hospitales = ["H. Virgen del Rocío", "H. Reina Sofía", "H. Puerta del Mar"]
    diagnosticos = ["Diabetes", "Hipertensión", "EPOC", "Cardiopatía", "Nefropatía"]
    
    pacientes_data = []
    for i in range(1, 501):  # 500 pacientes
        pacientes_data.append((
            i,
            random.randint(18, 85),
            random.choice(["M", "F"]),
            random.choice(hospitales),
            random.choice(diagnosticos),
            random.randint(1, 30)
        ))
    
    cursor.executemany("""
    INSERT OR REPLACE INTO pacientes 
    (id, edad, sexo, hospital, diagnostico, dias_hospitalizacion)
    VALUES (?, ?, ?, ?, ?, ?)
    """, pacientes_data)
    
    conn.commit()
    conn.close()
    print(" Base de datos creada: clinica_synthetic.db")

def analizar_cohortes():
    """Realiza análisis de cohortes clínicas"""
    print("\n ANÁLISIS DE COHORTES")
    print("=" * 40)
    
    conn = sqlite3.connect("clinica_synthetic.db")
    
    # Análisis 1: Distribución por edad y sexo
    query1 = """
    SELECT 
        CASE 
            WHEN edad < 30 THEN '18-29'
            WHEN edad < 50 THEN '30-49'
            WHEN edad < 70 THEN '50-69'
            ELSE '70+'
        END as grupo_edad,
        sexo,
        COUNT(*) as total_pacientes
    FROM pacientes 
    GROUP BY grupo_edad, sexo
    ORDER BY grupo_edad, sexo
    """
    
    df_demo = pd.read_sql_query(query1, conn)
    print("Distribución demográfica:")
    print(df_demo)
    
    # Análisis 2: Top diagnósticos
    query2 = """
    SELECT diagnostico, COUNT(*) as frecuencia
    FROM pacientes
    GROUP BY diagnostico
    ORDER BY frecuencia DESC
    """
    
    df_diag = pd.read_sql_query(query2, conn)
    print("\nTop diagnósticos:")
    print(df_diag)
    
    # Crear visualizaciones
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Distribución por diagnóstico
    df_diag.plot(x='diagnostico', y='frecuencia', kind='bar', ax=axes[0])
    axes[0].set_title('Distribución por Diagnóstico')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Distribución por hospital
    query3 = "SELECT hospital, COUNT(*) as pacientes FROM pacientes GROUP BY hospital"
    df_hosp = pd.read_sql_query(query3, conn)
    df_hosp.plot(x='hospital', y='pacientes', kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Pacientes por Hospital')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('analisis_iRWD.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    conn.close()
    print("\n✅ Análisis completado")

def main():
    """Función principal"""
    crear_base_datos_sintetica()
    analizar_cohortes()

if __name__ == "__main__":
    main()