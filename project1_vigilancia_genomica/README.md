# Proyecto 1: Vigilancia Genómica

## Descripción
Simulación de pipeline para detección de mutaciones en secuencias viricas, inspirado en el sistema **SIEGA** de ClinBioinfo para vigilancia epidemiológica.

## Objetivo
Demostrar capacidades en:
- Procesamiento de secuencias FASTA
- Alineamiento de secuencias
- Detección automática de variantes
- Generación de reportes epidemiológicos

## Tecnologías
- **Python 3.8+**
- **Biopython** (manejo de secuencias)
- **pandas** (análisis de datos)
- **matplotlib** (visualización)

## Instalación y ejecución

```bash
# Instalar dependencias
pip install biopython pandas matplotlib

# Ejecutar análisis
python script_align.py

# Salida esperada:
# - Reporte de mutaciones detectadas
# - Gráfico de distribución de variantes
# - Archivo CSV con resultados
```

## Conexión con ClinBioinfo
Este proyecto simula el flujo de trabajo del **sistema SIEGA** para:
- Vigilancia de variantes SARS-CoV-2
- Trazabilidad de bacterias resistentes
- Análisis epidemiológico automatizado