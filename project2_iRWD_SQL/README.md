# 🏥 Proyecto 2: iRWD - Datos Clínicos con SQL

## Descripción
Simulación de **integración y análisis de datos de mundo real (Real-World Data)** inspirado en la infraestructura **iRWD** de ClinBioinfo para investigación clínica.

## Tecnologías
- **Python 3.8+**
- **SQLite** (base de datos)
- **pandas** (análisis de datos)
- **matplotlib/seaborn** (visualización)

## Instalación y ejecución

```bash
# Instalar dependencias
pip install pandas matplotlib seaborn

# Crear base de datos sintética y ejecutar análisis
python consulta.py

# Salida esperada:
# - Base de datos SQLite con 500 pacientes
# - Análisis de cohortes por edad/sexo
# - Gráficos de distribuciones clínicas
```

## Conexión con ClinBioinfo
Este proyecto simula el flujo de trabajo de **iRWD** para:
- Estudios observacionales retrospectivos
- Análisis de cohortes clínicas
- Integración de datos hospitalarios