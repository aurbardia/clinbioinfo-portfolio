#!/usr/bin/env python3
"""
Pipeline de Vigilancia Genómica - ClinBioinfo Demo
Detección automática de mutaciones en secuencias virales
"""

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import matplotlib.pyplot as plt
import os

def generar_datos_sinteticos():
    """Genera archivo FASTA sintético para demo"""
    # Secuencia de referencia (simulando fragmento viral)
    referencia = "ATGCGATCGTAGCTAGCTAGCTAGCGATCGATCGTAGCTAGCTAGCTAGCGATCGATCGTAGCTAGCTAGCTAGC"
    
    secuencias_demo = [
        ("REF_001", referencia),
        ("VAR_002", referencia.replace("ATGC", "TTGC")),  # Mutación A>T
        ("VAR_003", referencia.replace("GATC", "CATC")),  # Mutación G>C
        ("VAR_004", referencia.replace("CGTA", "CGAA")),  # Mutación T>A
        ("VAR_005", referencia.replace("GCTA", "ACTA")),  # Mutación G>A
        ("VAR_006", referencia[:30] + "AAAA" + referencia[34:]),  # Inserción
        ("VAR_007", referencia.replace("TAGC", "AAGC")),  # Mutación T>A
        ("VAR_008", referencia.replace("ATCG", "ATAG")),  # Mutación C>A
        ("VAR_009", referencia.replace("GCGC", "ACGC")),  # Mutación G>A
        ("VAR_010", referencia.replace("CTAG", "CTGG"))   # Mutación A>G
    ]
    
    with open("datos_sinteticos.fasta", "w") as f:
        for seq_id, secuencia in secuencias_demo:
            f.write(f">{seq_id}\n{secuencia}\n")
            print(f"ID_sec: {seq_id}\n Secuencia:{secuencia}")
    
    print("Datos sintéticos generados: datos_sinteticos.fasta")
    

def cargar_secuencias(archivo_fasta):
    """Carga secuencias desde archivo FASTA"""
    secuencias = []
    for record in SeqIO.parse(archivo_fasta, "fasta"): #Lector archivo FASTA
        secuencias.append({
            'id': record.id,
            'secuencia': str(record.seq),
            'longitud': len(record.seq)
        })
    print(f"Cargadas {len(secuencias)} secuencias")
    return secuencias

def detectar_mutaciones(secuencias, referencia_idx=0):
    """Detecta mutaciones comparando con secuencia de referencia"""
    referencia = secuencias[referencia_idx]['secuencia']
    mutaciones = []
    
    for i, seq_data in enumerate(secuencias):
        if i == referencia_idx:
            continue
            
        secuencia = seq_data['secuencia']
        seq_id = seq_data['id']
        
        # Comparación posición por posición
        for pos, (ref_base, seq_base) in enumerate(zip(referencia, secuencia)):
            if ref_base != seq_base:
                mutaciones.append({
                    'secuencia_id': seq_id,
                    'posicion': pos + 1,
                    'referencia': ref_base,
                    'mutacion': seq_base,
                    'tipo': f"{ref_base}{pos+1}{seq_base}"
                })
    
    print(f"Detectadas {len(mutaciones)} mutaciones")
    return mutaciones

def generar_reporte(mutaciones):
    """Genera reporte de mutaciones en CSV y visualización"""
    if not mutaciones:
        print("No se encontraron mutaciones")
        return
    
    # DataFrame con mutaciones
    df_mut = pd.DataFrame(mutaciones)
    
    # Guardar CSV
    df_mut.to_csv('reporte_mutaciones.csv', index=False)
    print("Reporte guardado: reporte_mutaciones.csv")
    
    # Visualización
    plt.figure(figsize=(12, 6))
    
    # Gráfico 1: Distribución de posiciones
    plt.subplot(1, 2, 1)
    plt.hist(df_mut['posicion'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Posición en genoma')
    plt.ylabel('Número de mutaciones')
    plt.title('Distribución de mutaciones por posición')
    
    # Gráfico 2: Tipos de mutación más frecuentes
    plt.subplot(1, 2, 2)
    top_mutaciones = df_mut['tipo'].value_counts().head(10)
    top_mutaciones.plot(kind='bar', color='lightcoral')
    plt.xlabel('Tipo de mutación')
    plt.ylabel('Frecuencia')
    plt.title('Top 10 mutaciones detectadas')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('analisis_mutaciones.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Resumen estadístico
    print("\n RESUMEN EPIDEMIOLÓGICO")
    print(f"Total mutaciones: {len(mutaciones)}")
    print(f"Secuencias analizadas: {df_mut['secuencia_id'].nunique()}")
    print(f"Posiciones afectadas: {df_mut['posicion'].nunique()}")

def main():
    """Pipeline principal"""
    print("PIPELINE DE VIGILANCIA GENÓMICA - ClinBioinfo Demo")
    print("=" * 60)
    
    # Verificar archivo de datos
    archivo_fasta = "datos_sinteticos.fasta"
    if not os.path.exists(archivo_fasta):
        print(f"No se encuentra {archivo_fasta}")
        print("Generando datos sintéticos...")
        generar_datos_sinteticos()
    
    # Ejecutar pipeline
    secuencias = cargar_secuencias(archivo_fasta)
    mutaciones = detectar_mutaciones(secuencias)
    generar_reporte(mutaciones)
    
    print("\n Pipeline completado exitosamente")
    print("Archivos generados:")
    print("  - reporte_mutaciones.csv")
    print("  - analisis_mutaciones.png")

if __name__ == "__main__":
    main()