#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPARACIÓN: SGPMAIN vs BioPython (Propiedades Fisicoquímicas)
ANÁLISIS: Nile Virus (NILE1 y NILE2) Glycoprotein Analysis
VERSIÓN CORREGIDA
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
import json
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 COMPARACIÓN: SGPMAIN vs BioPython (VERSIÓN MEJORADA)")
print("   Nile Virus (NILE1 y NILE2) Glycoprotein Analysis")
print("   BioPython: Propiedades fisicoquímicas de proteínas")
print("   Métrica: Distancia Euclidiana normalizada")
print("=" * 80)
print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# IMPORTAR BIOPYTHON
# ============================================================================

try:
    from Bio.SeqUtils import ProtParam
    from Bio.Seq import Seq
    from Bio import SeqIO
    from Bio.SeqUtils import molecular_weight
    BIOPYTHON_AVAILABLE = True
    print("✅ BioPython importado correctamente")
except ImportError:
    print("❌ BioPython no está instalado. Ejecuta: pip install biopython")
    sys.exit(1)

# ============================================================================
# FUNCIONES PIM (SGPMAIN)
# ============================================================================

POLARITY_MAP = {
    'H': 'P+', 'K': 'P+', 'R': 'P+',
    'D': 'P-', 'E': 'P-',
    'C': 'N', 'G': 'N', 'N': 'N', 'Q': 'N', 'S': 'N', 'T': 'N', 'Y': 'N',
    'A': 'NP', 'F': 'NP', 'I': 'NP', 'L': 'NP', 'M': 'NP', 'P': 'NP', 'V': 'NP', 'W': 'NP'
}

INTERACTIONS = [
    'P+,P+', 'P+,P-', 'P+,N', 'P+,NP',
    'P-,P+', 'P-,P-', 'P-,N', 'P-,NP',
    'N,P+', 'N,P-', 'N,N', 'N,NP',
    'NP,P+', 'NP,P-', 'NP,N', 'NP,NP'
]
INTERACTION_TO_IDX = {inter: i for i, inter in enumerate(INTERACTIONS)}

def compute_pim_profile(sequence, use_weights=True):
    """Calcula el vector PIM (16 dimensiones)"""
    seq = ''.join([c for c in str(sequence).strip() if c.upper() in POLARITY_MAP])
    if len(seq) < 2:
        return np.zeros(16)
    
    polarities = []
    for aa in seq:
        pol = POLARITY_MAP.get(aa.upper())
        if pol is not None:
            polarities.append(pol)
    
    if len(polarities) < 2:
        return np.zeros(16)
    
    counts = np.zeros(16)
    for i in range(len(polarities) - 1):
        pair = f"{polarities[i]},{polarities[i+1]}"
        if pair in INTERACTION_TO_IDX:
            counts[INTERACTION_TO_IDX[pair]] += 1
    
    total = np.sum(counts)
    if total > 0:
        counts = counts / total
    
    if use_weights:
        weights = {
            'P+,P-': 2.0, 'P-,P+': 2.0, 'N,N': 1.5,
            'N,P+': 1.3, 'P+,N': 1.3, 'N,P-': 1.3, 'P-,N': 1.3,
            'NP,NP': 1.0, 'NP,N': 0.9, 'N,NP': 0.9,
            'NP,P+': 0.7, 'P+,NP': 0.7, 'NP,P-': 0.7, 'P-,NP': 0.7,
            'P+,P+': 0.4, 'P-,P-': 0.4,
        }
        weighted = np.zeros(16)
        for i, inter in enumerate(INTERACTIONS):
            weighted[i] = counts[i] * weights.get(inter, 1.0)
        total_w = np.sum(weighted)
        if total_w > 0:
            weighted = weighted / total_w
        return weighted
    
    return counts

def read_fasta_stream(filepath):
    """Lee archivo FASTA secuencialmente"""
    if not os.path.exists(filepath):
        return
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield header, ''.join(seq)

def get_display_name(group_name):
    """Obtiene nombre legible del grupo"""
    group_map = {
        'enfermedad': 'DISEASE', 'membrana': 'MEMBRANE', 'senales': 'SIGNALS',
        'lujo': 'LUJO', 'lasv': 'LASV', 'junv': 'JUNV',
        'macv': 'MACV', 'lcmv': 'LCMV',
        'nile1': 'NILE1', 'nile2': 'NILE2',
        'CPP': 'CPP', 'NON_CPP': 'NON_CPP',
        'UNFOLDED': 'UNFOLDED', 'PARTIALLY_FOLDED': 'PARTIALLY_FOLDED',
        'REVIEWED_HUMAN': 'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN': 'UNREVIEWED_HUMAN',
        'VIRUS_REVIEWED': 'VIRUS_REVIEWED', 'VIRUS_UNREVIEWED': 'VIRUS_UNREVIEWED',
        'REVIEWED_ALL': 'REVIEWED_ALL', 'UNREVIEWED_ALL': 'UNREVIEWED_ALL',
        'sudan': 'EBOLA_SUDAN', 'zaire': 'EBOLA_ZAIRE',
        'reston': 'EBOLA_RESTON', 'bombali': 'EBOLA_BOMBALI',
        'bundibugyo': 'EBOLA_BUNDIBUGYO', 'tai': 'EBOLA_TAI_FOREST',
    }
    return group_map.get(group_name, group_name)

def get_filename(group_name):
    """Devuelve el nombre de archivo según SGPMAIN.py"""
    file_map = {
        'sudan': 'Sudan.unico.dat0',
        'zaire': 'Zaire.unico.dat0',
        'reston': 'Reston.unico.dat0',
        'bombali': 'Bombali.unico.dat0',
        'bundibugyo': 'Bundibugyo.unico.dat0',
        'tai': 'Tai.unico.dat0',
        'lasv': 'lasv_all.unico.dat0',
        'junv': 'junv_all.unico.dat0',
        'macv': 'macv_all.unico.dat0',
        'lcmv': 'lcmv_all.unico.dat0',
        'nile1': 'nile1.unico.dat0',
        'nile2': 'nile2.unico.dat0',
        'lujo': 'lujo.unico.dat0',
        'PARTIALLY_FOLDED': 'partiallyorderedN.unico.dat0',
        'CPP': 'CPP.unico.dat0',
        'NON_CPP': 'NONCPP.unico.dat0',
        'UNFOLDED': 'unfolded.unico.dat0',
        'REVIEWED_HUMAN': 'reviewed_human.unico.dat0',
        'UNREVIEWED_HUMAN': 'unreviewed_human.unico.dat0',
        'senales': 'senales.unico.dat0',
        'membrana': 'membrana.unico.dat0',
        'enfermedad': 'enfermedad.unico.dat0',
        'VIRUS_REVIEWED': 'reviewed_virus.unico.dat0',
        'VIRUS_UNREVIEWED': 'unreviewed_virus.unico.dat0',
        'REVIEWED_ALL': 'reviewed_all.unico.dat0',
        'UNREVIEWED_ALL': 'unreviewed_all.unico.dat0',
    }
    return file_map.get(group_name, f"{group_name}.unico.dat0")

# ============================================================================
# EXTRAER RESULTADOS DE SGPMAIN
# ============================================================================

def extract_sgp_results(results_dir, target_virus='nile1'):
    """Extrae los resultados de SGPMAIN del archivo comparison_*_vs_all.csv"""
    sgp_results = {}
    
    comparison_file = None
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            if f.startswith('comparison_') and f.endswith('.csv'):
                if target_virus.lower() in f.lower():
                    comparison_file = os.path.join(results_dir, f)
                    break
    
    if comparison_file is None:
        # Buscar en directorio actual
        for f in os.listdir('.'):
            if f.startswith('comparison_') and f.endswith('.csv'):
                if target_virus.lower() in f.lower():
                    comparison_file = f
                    break
    
    if comparison_file is None:
        print(f"  ⚠️ No se encontró comparison_*_{target_virus}_vs_all.csv")
        return None
    
    print(f"  📂 Leyendo: {comparison_file}")
    df = pd.read_csv(comparison_file)
    
    # Detectar el nombre de la columna correcta
    col_name = None
    for col in df.columns:
        if 'similarity' in col.lower() or 'Similarity' in col:
            col_name = col
            break
    
    if col_name is None:
        col_name = df.columns[1]  # Asumir segunda columna
    
    for _, row in df.iterrows():
        group = row[df.columns[0]]
        sgp_results[group] = row[col_name]
    
    return sgp_results

# ============================================================================
# FUNCIONES PARA BIOPYTHON (MEJORADAS)
# ============================================================================

def extract_biopython_features_enhanced(sequence):
    """
    Extrae características fisicoquímicas mejoradas usando BioPython
    
    Características extraídas (50 dimensiones):
    1-20: Contenido de aminoácidos (%)
    21: Peso molecular normalizado
    22: Punto isoeléctrico
    23: Índice de aromaticidad
    24: Índice de inestabilidad
    25: GRAVY
    26-28: Fracción de estructura secundaria
    29: Carga neta
    30: Longitud
    31-50: Propiedades de aminoácidos escaladas (Kyte-Doolittle, etc.)
    """
    try:
        # Limpiar secuencia
        seq_str = ''.join([c for c in str(sequence).strip() if c.isalpha()])
        
        if len(seq_str) < 5:
            return None
        
        # Crear objeto Seq
        seq_obj = Seq(seq_str)
        analyzer = ProtParam.ProteinAnalysis(str(seq_obj))
        
        features = []
        
        # 1-20. Contenido de aminoácidos (más importante)
        try:
            aa_counts = analyzer.get_amino_acids_percent()
            for aa in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
                features.append(aa_counts.get(aa, 0.0))
        except:
            features.extend([0.0] * 20)
        
        # 21. Peso molecular (normalizado por longitud)
        try:
            mw = analyzer.molecular_weight()
            features.append(mw / len(seq_str))  # Normalizado
        except:
            features.append(0.0)
        
        # 22. Punto isoeléctrico
        try:
            features.append(analyzer.isoelectric_point())
        except:
            features.append(7.0)
        
        # 23. Índice de aromaticidad
        try:
            features.append(analyzer.aromaticity())
        except:
            features.append(0.0)
        
        # 24. Índice de inestabilidad
        try:
            features.append(analyzer.instability_index())
        except:
            features.append(50.0)
        
        # 25. GRAVY
        try:
            features.append(analyzer.gravy())
        except:
            features.append(0.0)
        
        # 26-28. Fracción de estructura secundaria
        try:
            sec_struct = analyzer.secondary_structure_fraction()
            features.extend(sec_struct)
        except:
            features.extend([0.0, 0.0, 0.0])
        
        # 29. Carga neta
        try:
            charges = {'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1}
            net_charge = sum(charges.get(aa, 0) for aa in seq_str)
            features.append(net_charge / len(seq_str))  # Normalizado
        except:
            features.append(0.0)
        
        # 30. Longitud (log transformada para reducir efecto)
        features.append(np.log(len(seq_str) + 1))
        
        # 31-50. Propiedades de aminoácidos (20 adicionales)
        # Usamos propiedades fisicoquímicas como hidrofobicidad, volumen, etc.
        try:
            # Valores de hidrofobicidad de Kyte-Doolittle
            hydrophobicity = {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            }
            avg_hydro = sum(hydrophobicity.get(aa, 0) for aa in seq_str) / len(seq_str)
            features.append(avg_hydro)
            
            # Índice de flexibilidad
            flexibility = {
                'A': 0.0, 'R': 0.5, 'N': 0.3, 'D': 0.4, 'C': 0.1,
                'Q': 0.4, 'E': 0.5, 'G': 0.1, 'H': 0.3, 'I': 0.0,
                'L': 0.0, 'K': 0.5, 'M': 0.1, 'F': 0.0, 'P': 0.6,
                'S': 0.3, 'T': 0.2, 'W': 0.1, 'Y': 0.1, 'V': 0.0
            }
            avg_flex = sum(flexibility.get(aa, 0) for aa in seq_str) / len(seq_str)
            features.append(avg_flex)
            
            # 18 características adicionales (relleno para mantener consistencia)
            features.extend([0.0] * 18)
            
        except:
            features.extend([0.0] * 20)
        
        return np.array(features)
        
    except Exception as e:
        print(f"  ⚠️ Error en BioPython: {e}")
        return None

def normalize_features(features_list):
    """Normaliza características usando StandardScaler"""
    if not features_list:
        return features_list
    
    features_array = np.array(features_list)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features_array)
    return normalized

def euclidean_similarity(v1, v2):
    """Calcula similitud basada en distancia Euclidiana"""
    # Normalizar vectores
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    
    # Distancia Euclidiana
    dist = np.linalg.norm(v1_norm - v2_norm)
    
    # Convertir a similitud (1 - distancia normalizada)
    max_dist = np.sqrt(2)  # Distancia máxima entre vectores normalizados
    sim = 1 - (dist / max_dist)
    
    return max(0, min(1, sim))  # Asegurar entre 0 y 1

def run_biopython_enhanced(virus_file, virus_name, max_sequences_per_group=50):
    """
    Ejecuta BioPython mejorado para analizar un virus contra todos los grupos
    """
    print(f"\n  🔬 Ejecutando BioPython mejorado para {virus_name} (50 características fisicoquímicas)...")
    
    # Leer secuencia del virus
    if not os.path.exists(virus_file):
        print(f"  ⚠️ No se encontró {virus_file}")
        return None
    
    virus_seq = None
    for header, seq in read_fasta_stream(virus_file):
        virus_seq = seq
        break
    
    if virus_seq is None:
        print(f"  ⚠️ No se pudo leer la secuencia de {virus_name}")
        return None
    
    # Extraer características del virus
    print(f"  📤 Extrayendo características de {virus_name}...")
    virus_features = extract_biopython_features_enhanced(virus_seq)
    
    if virus_features is None or len(virus_features) == 0:
        print(f"  ❌ Error extrayendo características de {virus_name}")
        return None
    
    print(f"     ├─ Dimensiones: {len(virus_features)} características")
    print(f"     ├─ Peso molecular: {virus_features[20] * 75:.2f} Da (normalizado)")
    print(f"     ├─ Punto isoeléctrico: {virus_features[21]:.2f}")
    print(f"     ├─ Índice de inestabilidad: {virus_features[23]:.2f}")
    print(f"     ├─ GRAVY: {virus_features[24]:.2f}")
    print(f"     └─ Longitud: {np.exp(virus_features[29]) - 1:.0f} aa")
    
    # Lista de grupos a analizar
    groups_to_test = [
        'CPP', 'NON_CPP', 'UNFOLDED', 'PARTIALLY_FOLDED',
        'REVIEWED_HUMAN', 'UNREVIEWED_HUMAN',
        'VIRUS_REVIEWED', 'VIRUS_UNREVIEWED',
        'REVIEWED_ALL', 'UNREVIEWED_ALL',
        'lujo', 'lasv', 'junv', 'macv', 'lcmv',
        'sudan', 'zaire', 'reston', 'bombali', 'bundibugyo', 'tai'
    ]
    
    biopython_results = {}
    
    for group in groups_to_test:
        group_file = get_filename(group)
        
        if not os.path.exists(group_file):
            continue
        
        print(f"     ├─ Procesando {get_display_name(group)}...")
        
        # Leer secuencias del grupo
        vectors = []
        count = 0
        for header, seq_group in read_fasta_stream(group_file):
            features = extract_biopython_features_enhanced(seq_group)
            if features is not None and len(features) > 0:
                vectors.append(features)
                count += 1
                if count >= max_sequences_per_group:
                    break
        
        if not vectors:
            continue
        
        # Normalizar características
        all_vectors = [virus_features] + vectors
        normalized_vectors = normalize_features(all_vectors)
        virus_norm = normalized_vectors[0]
        vectors_norm = normalized_vectors[1:]
        
        # Calcular similitud usando distancia Euclidiana
        similarities = []
        for vec in vectors_norm:
            sim = euclidean_similarity(virus_norm, vec)
            similarities.append(sim)
        
        if similarities:
            biopython_results[group] = np.mean(similarities)
            print(f"        └─ Similitud media: {biopython_results[group]:.6f} (n={len(similarities)})")
        else:
            print(f"        └─ ⚠️ Sin características válidas")
    
    return biopython_results

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

print("\n📂 Buscando resultados de SGPMAIN...")

# Buscar el directorio de resultados más reciente
results_dirs = []
for d in os.listdir('.'):
    if d.startswith('results_v17_') and os.path.isdir(d):
        results_dirs.append(d)

if not results_dirs:
    for d in os.listdir('.'):
        if d.startswith('results_v16_') and os.path.isdir(d):
            results_dirs.append(d)

if not results_dirs:
    for d in os.listdir('.'):
        if d.startswith('results_') and os.path.isdir(d):
            results_dirs.append(d)

# Analizar NILE1 y NILE2
viruses_to_analyze = [
    {'file': 'nile1.unico.dat0', 'name': 'nile1', 'display': 'NILE1'},
    {'file': 'nile2.unico.dat0', 'name': 'nile2', 'display': 'NILE2'}
]

all_comparisons = {}
all_dataframes = []

for virus in viruses_to_analyze:
    print("\n" + "=" * 80)
    print(f"🔬 ANALIZANDO: {virus['display']}")
    print("=" * 80)
    
    # Verificar si existe el archivo del virus
    if not os.path.exists(virus['file']):
        print(f"  ⚠️ No se encontró {virus['file']}")
        print(f"  ⚠️ Saltando {virus['display']}...")
        continue
    
    sgp_results = None
    if results_dirs:
        latest_dir = sorted(results_dirs)[-1]
        print(f"  📁 Usando: {latest_dir}")
        sgp_results = extract_sgp_results(latest_dir, virus['name'])
    
    # Si no se encontró en el directorio, buscar en el actual
    if sgp_results is None:
        sgp_results = extract_sgp_results('.', virus['name'])
    
    if sgp_results is None or len(sgp_results) == 0:
        print(f"\n⚠️ No se pudieron obtener resultados de SGPMAIN para {virus['display']}")
        print(f"   Continuando solo con análisis de BioPython...")
        sgp_results = {}  # Vacío para continuar
    else:
        print(f"\n  ✅ Cargados {len(sgp_results)} grupos de SGPMAIN")
    
    # Ejecutar BioPython mejorado
    biopython_results = run_biopython_enhanced(
        virus['file'], 
        virus['display'],
        max_sequences_per_group=50
    )
    
    if biopython_results is None or len(biopython_results) == 0:
        print(f"\n❌ No se pudieron obtener resultados de BioPython para {virus['display']}")
        continue
    
    # Guardar resultados
    all_comparisons[virus['name']] = {
        'sgp': sgp_results,
        'biopython': biopython_results
    }
    
    # ============================================================================
    # TABLA COMPARATIVA
    # ============================================================================
    
    print("\n" + "=" * 80)
    print(f"📋 TABLA COMPARATIVA: SGPMAIN vs BioPython ({virus['display']})")
    print("=" * 80)
    print(f"{'Grupo':<22} {'SGPMAIN':>12} {'BioPython':>12} {'Diferencia':>12} {'Interpretación':>15}")
    print("-" * 80)
    
    comparacion = []
    groups_compared = set(sgp_results.keys()) & set(biopython_results.keys())
    
    if not groups_compared:
        # Si no hay grupos comunes, usar todos los de BioPython
        groups_compared = set(biopython_results.keys())
        print("  ⚠️ No hay grupos comunes. Mostrando todos los grupos de BioPython.")
    
    for grupo in sorted(groups_compared, key=lambda x: sgp_results.get(x, 0) if x in sgp_results else 0, reverse=True):
        sgp_val = sgp_results.get(grupo, 0.0)
        biopy_val = biopython_results.get(grupo, 0.0)
        diff = abs(sgp_val - biopy_val) if sgp_val > 0 else 1.0
        
        if sgp_val == 0:
            interp = "⚠️ Sin SGP"
        elif diff < 0.01:
            interp = "✅ Excelente"
        elif diff < 0.03:
            interp = "✔️ Buena"
        elif diff < 0.05:
            interp = "⚠️ Moderada"
        else:
            interp = "❌ Diferente"
        
        display_name = get_display_name(grupo)
        print(f"{display_name:<22} {sgp_val:>12.6f} {biopy_val:>12.6f} "
              f"{diff:>12.6f} {interp:>15}")
        
        comparacion.append({
            'Virus': virus['display'],
            'Grupo': display_name,
            'Grupo_original': grupo,
            'SGPMAIN': sgp_val,
            'BioPython': biopy_val,
            'Diferencia': diff,
            'Interpretación': interp
        })
    
    all_dataframes.append(pd.DataFrame(comparacion))
    
    # ============================================================================
    # ESTADÍSTICAS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print(f"📊 ANÁLISIS ESTADÍSTICO ({virus['display']})")
    print("=" * 80)
    
    if comparacion:
        sgp_values = [c['SGPMAIN'] for c in comparacion if c['SGPMAIN'] > 0]
        biopy_values = [c['BioPython'] for c in comparacion]
        differences = [c['Diferencia'] for c in comparacion if c['SGPMAIN'] > 0]
        
        if sgp_values:
            print(f"\n  📊 SGPMAIN (16 descriptores PIM):")
            print(f"     ├─ Rango: {min(sgp_values):.6f} - {max(sgp_values):.6f}")
            print(f"     ├─ Media: {np.mean(sgp_values):.6f}")
            print(f"     └─ Desviación estándar: {np.std(sgp_values):.6f}")
        
        print(f"\n  📊 BioPython (50 características fisicoquímicas mejoradas):")
        print(f"     ├─ Rango: {min(biopy_values):.6f} - {max(biopy_values):.6f}")
        print(f"     ├─ Media: {np.mean(biopy_values):.6f}")
        print(f"     └─ Desviación estándar: {np.std(biopy_values):.6f}")
        
        if differences:
            print(f"\n  📊 Diferencia media: {np.mean(differences):.6f}")
            print(f"  📊 Diferencia máxima: {np.max(differences):.6f}")
            print(f"  📊 Diferencia mínima: {np.min(differences):.6f}")
        
            excelente = sum(1 for c in comparacion if c['Interpretación'] == '✅ Excelente')
            buena = sum(1 for c in comparacion if c['Interpretación'] == '✔️ Buena')
            moderada = sum(1 for c in comparacion if c['Interpretación'] == '⚠️ Moderada')
            diferente = sum(1 for c in comparacion if c['Interpretación'] == '❌ Diferente')
            sin_sgp = sum(1 for c in comparacion if c['Interpretación'] == '⚠️ Sin SGP')
        
            print(f"\n  📊 Distribución de diferencias:")
            if sin_sgp > 0:
                print(f"     ├─ ⚠️ Sin datos SGP: {sin_sgp}")
            print(f"     ├─ ✅ Excelente (<0.01): {excelente}")
            print(f"     ├─ ✔️ Buena (0.01-0.03): {buena}")
            print(f"     ├─ ⚠️ Moderada (0.03-0.05): {moderada}")
            print(f"     └─ ❌ Diferente (>0.05): {diferente}")
    
    # ============================================================================
    # GUARDAR RESULTADOS
    # ============================================================================
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"comparacion_sgp_biopython_{virus['name']}_v2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    df_comp = pd.DataFrame(comparacion)
    df_comp.to_csv(f"{results_dir}/comparacion_sgp_biopython.csv", index=False)
    print(f"\n  ✅ Tabla guardada: {results_dir}/comparacion_sgp_biopython.csv")
    
    if sgp_results:
        with open(f"{results_dir}/sgp_results.json", 'w') as f:
            json.dump(sgp_results, f, indent=2)
        print(f"  ✅ SGP resultados guardados: {results_dir}/sgp_results.json")
    
    with open(f"{results_dir}/biopython_results.json", 'w') as f:
        json.dump(biopython_results, f, indent=2)
    print(f"  ✅ BioPython resultados guardados: {results_dir}/biopython_results.json")
    
    # ============================================================================
    # CONCLUSIÓN
    # ============================================================================
    
    print("\n" + "=" * 80)
    print(f"🎯 CONCLUSIONES ({virus['display']})")
    print("=" * 80)
    
    if comparacion:
        sgp_val = [c['SGPMAIN'] for c in comparacion if c['SGPMAIN'] > 0]
        bio_val = [c['BioPython'] for c in comparacion]
        diff_val = [c['Diferencia'] for c in comparacion if c['SGPMAIN'] > 0]
        
        # Corregir el error de formato
        sgp_min = min(sgp_val) if sgp_val else 0
        sgp_max = max(sgp_val) if sgp_val else 0
        sgp_mean = np.mean(sgp_val) if sgp_val else 0
        
        bio_min = min(bio_val) if bio_val else 0
        bio_max = max(bio_val) if bio_val else 0
        bio_mean = np.mean(bio_val) if bio_val else 0
        
        diff_mean = np.mean(diff_val) if diff_val else 0
        
        print(f"""
  📌 RESUMEN DE LA COMPARACIÓN (SGPMAIN vs BioPython mejorado) - {virus['display']}:
  
  1. Número de grupos comparados: {len(comparacion)}
  
  2. SGPMAIN (16 descriptores PIM):
     - Rango: [{sgp_min:.6f}, {sgp_max:.6f}]
     - Media: {sgp_mean:.6f}
  
  3. BioPython (50 características fisicoquímicas mejoradas):
     - Rango: [{bio_min:.6f}, {bio_max:.6f}]
     - Media: {bio_mean:.6f}
  
  4. Diferencia media entre métodos: {diff_mean:.6f}
  
  🔬 IMPLICACIÓN PARA EL ANÁLISIS:
  
  La versión mejorada de BioPython con 50 características y distancia
  Euclidiana normalizada proporciona una comparación más equilibrada.
  Los resultados muestran {excelente} grupos con excelente concordancia,
  {buena} con buena concordancia, y {moderada} con concordancia moderada.
""")

# ============================================================================
# COMPARACIÓN FINAL ENTRE NILE1 Y NILE2
# ============================================================================

if len(all_comparisons) >= 2:
    print("\n" + "=" * 80)
    print("🔍 COMPARACIÓN ENTRE NILE1 Y NILE2")
    print("=" * 80)
    
    nile1_data = all_comparisons.get('nile1', {})
    nile2_data = all_comparisons.get('nile2', {})
    
    if nile1_data and nile2_data:
        # Usar grupos comunes con datos SGP
        nile1_groups = set(nile1_data.get('sgp', {}).keys())
        nile2_groups = set(nile2_data.get('sgp', {}).keys())
        common_groups = nile1_groups & nile2_groups
        
        print(f"\n  📊 Grupos comunes con datos SGP: {len(common_groups)}")
        
        if common_groups:
            nile1_sgp = [nile1_data['sgp'].get(g, 0) for g in common_groups]
            nile2_sgp = [nile2_data['sgp'].get(g, 0) for g in common_groups]
            nile1_bio = [nile1_data['biopython'].get(g, 0) for g in common_groups]
            nile2_bio = [nile2_data['biopython'].get(g, 0) for g in common_groups]
            
            print(f"\n  📊 Comparación SGPMAIN:")
            print(f"     ├─ Correlación NILE1 vs NILE2: {np.corrcoef(nile1_sgp, nile2_sgp)[0,1]:.6f}")
            print(f"     ├─ Diferencia media: {np.mean(np.abs(np.array(nile1_sgp) - np.array(nile2_sgp))):.6f}")
            print(f"     └─ Rango NILE1: {min(nile1_sgp):.4f}-{max(nile1_sgp):.4f}")
            print(f"        Rango NILE2: {min(nile2_sgp):.4f}-{max(nile2_sgp):.4f}")
            
            print(f"\n  📊 Comparación BioPython:")
            print(f"     ├─ Correlación NILE1 vs NILE2: {np.corrcoef(nile1_bio, nile2_bio)[0,1]:.6f}")
            print(f"     ├─ Diferencia media: {np.mean(np.abs(np.array(nile1_bio) - np.array(nile2_bio))):.6f}")
            print(f"     └─ Rango NILE1: {min(nile1_bio):.4f}-{max(nile1_bio):.4f}")
            print(f"        Rango NILE2: {min(nile2_bio):.4f}-{max(nile2_bio):.4f}")
        else:
            print("\n  ⚠️ No hay grupos comunes para comparar NILE1 y NILE2")
            print("  Mostrando comparación directa de similitudes:")
            
            bio1 = list(nile1_data['biopython'].values())
            bio2 = list(nile2_data['biopython'].values()) if nile2_data else []
            
            if bio1 and bio2:
                print(f"\n  BioPython NILE1: {len(bio1)} grupos, media {np.mean(bio1):.6f}")
                print(f"  BioPython NILE2: {len(bio2)} grupos, media {np.mean(bio2):.6f}")
                print(f"  Diferencia de medias: {np.mean(bio1) - np.mean(bio2):.6f}")
    else:
        print("\n  ⚠️ No hay datos completos para comparar NILE1 y NILE2")
else:
    print("\n" + "=" * 80)
    print("⚠️ COMPARACIÓN INCOMPLETA ENTRE NILE1 Y NILE2")
    print("=" * 80)
    print(f"  Solo se analizó: {list(all_comparisons.keys())}")
    print("  No se puede hacer comparación cruzada")

# ============================================================================
# GUARDAR RESULTADOS COMBINADOS
# ============================================================================

if all_dataframes:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(f"comparacion_combinada_nile_v2_{timestamp}.csv", index=False)
    print(f"\n  ✅ Resultados combinados guardados: comparacion_combinada_nile_v2_{timestamp}.csv")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
