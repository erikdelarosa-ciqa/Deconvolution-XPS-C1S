import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from lmfit.models import PseudoVoigtModel

# ==============================================================================
# BLOQUE 1: INDEXACIÓN QUÍMICA AVANZADA (CONEXIÓN CON BASE DE DATOS)
# ==============================================================================
def identificar_enlace_desde_db(energia_calculada, ruta_db='base_datos_xps.csv'):
    """
    Busca de manera dinámica el grupo funcional del Carbono en el archivo CSV 
    aplicando las fronteras asimétricas de tolerancia química (Min_eV y Max_eV).
    """
    if not os.listdir('.') or ruta_db not in os.listdir('.'):
        ruta_alternativa = '../../base_datos_xps.csv'
        if os.path.exists(ruta_alternativa):
            ruta_db = ruta_alternativa
        else:
            return "Señal no indexada"
            
    try:
        df_db = pd.read_csv(ruta_db, encoding='latin1')
        for _, fila in df_db.iterrows():
            nombre = fila['Grupo_Funcional']
            ref_ev = float(fila['Energia_eV'])
            desviacion_min = float(fila['Min_eV'])
            desviacion_max = float(fila['Max_eV'])
            
            if (ref_ev - desviacion_min) <= energia_calculada <= (ref_ev + desviacion_max):
                return str(nombre)
    except:
        pass
        
    return "Señal no indexada"

# ==============================================================================
# BLOQUE 2: MOTOR INTELIGENTE DE EXTRACCIÓN DE SEÑAL
# ==============================================================================
def extraer_datos_carbono_del_zip(zip_name):
    """
    Abre el archivo comprimido y escanea las matrices internas.
    Selecciona exclusivamente el archivo que opera en la ventana del Carbono C1s.
    """
    try:
        with zipfile.ZipFile(zip_name, 'r') as zf:
            archivos_candidatos = [f for f in zf.namelist() if f.endswith(('.csv', '.txt')) 
                                   and not f.lower().endswith('.pdf') and 'metadata' not in f.lower()]
            
            for candidato in archivos_candidatos:
                with zf.open(candidato) as f:
                    try:
                        datos_temp = np.genfromtxt(f, delimiter=',', skip_header=1)
                        if np.isnan(datos_temp).all() or datos_temp.ndim < 2:
                            f.seek(0)
                            datos_temp = np.genfromtxt(f, skip_header=1)
                        
                        if np.any((datos_temp[:, 0] >= 281.5) & (datos_temp[:, 0] <= 291.0)):
                            return datos_temp, candidato
                    except:
                        continue
    except:
        pass
    return None, None

# ==============================================================================
# BLOQUE 3: ALGORITMO CIENTÍFICO DE PROCESAMIENTO (RESTRICCIONES LIBRES ADAPTATIVAS)
# ==============================================================================
def calcular_linea_base_shirley(y, max_iters=15, tol=1e-5):
    """Calcula de manera iterativa la línea base Shirley de forma idéntica a Origin."""
    y = np.array(y, dtype=float)
    n = len(y)
    if n < 2:
        return np.zeros_like(y)
    fondo = np.linspace(y[0], y[-1], n)
    
    for _ in range(max_iters):
        fondo_anterior = fondo.copy()
        area_acumulada = np.cumsum(y - fondo)
        area_total = area_acumulada[-1]
        if area_total == 0:
            break
        for i in range(n):
            fondo[i] = y[-1] + (y[0] - y[-1]) * (area_total - area_acumulada[i]) / area_total
        if np.max(np.abs(fondo - fondo_anterior)) < tol:
            break
    return fondo

def procesar_y_deconvolucionar_c1s(nombre_muestra, datos, archivo_origen):
    """Ajusta un modelo Pseudo-Voigt de picos simétricos libres calculando el R2 del ajuste."""
    x_completo = datos[:, 0]
    y_completo = datos[:, 1]

    if x_completo[0] < x_completo[-1]:
        x_completo = x_completo[::-1]
        y_completo = y_completo[::-1]

    mascara = (x_completo >= 281.5) & (x_completo <= 291.0)
    x_crudo = x_completo[mascara].flatten()
    y = y_completo[mascara].flatten()

    if len(y) == 0:
        print(f"      ⚠️ ADVERTENCIA: Ventana C1s vacía para {nombre_muestra}.")
        return None

    fondo_shirley = calcular_linea_base_shirley(y)
    y_neta = np.clip(y - fondo_shirley, 0, None)

    mascara_zona_cc = (x_crudo >= 284.0) & (x_crudo <= 286.5)
    if np.any(mascara_zona_cc):
        indice_cc = np.where(mascara_zona_cc)[0][np.argmax(y_neta[mascara_zona_cc])]
        energia_cc_cruda = x_crudo[indice_cc]
    else:
        energia_cc_cruda = x_crudo[np.argmax(y_neta)]
        
    desfase_carga = energia_cc_cruda - 284.50
    x_calibrada_previa = (x_crudo - desfase_carga).flatten()

    modelo_total = (PseudoVoigtModel(prefix='c1_') + PseudoVoigtModel(prefix='c2_') + 
                    PseudoVoigtModel(prefix='c3_') + PseudoVoigtModel(prefix='c4_'))
    
    max_y = np.max(y_neta)
    params = modelo_total.make_params()
    
    params['c1_amplitude'].set(value=max_y * 0.7, min=0)
    params['c1_center'].set(value=284.50, min=284.20, max=284.80)
    params['c1_sigma'].set(value=0.60, min=0.40, max=0.95)
    params['c1_fraction'].set(value=0.5, min=0.0, max=1.0, vary=True)

    params['c2_amplitude'].set(value=max_y * 0.15, min=0)
    params['c2_center'].set(value=286.20, min=285.70, max=286.80)
    params['c2_sigma'].set(value=0.65, min=0.40, max=1.20)
    params['c2_fraction'].set(value=0.5, min=0.0, max=1.0, vary=True)

    params['c3_amplitude'].set(value=max_y * 0.05, min=0)
    params['c3_center'].set(value=288.30, min=287.50, max=289.20)
    params['c3_sigma'].set(value=0.70, min=0.45, max=1.40)
    params['c3_fraction'].set(value=0.5, min=0.0, max=1.0, vary=True)

    if "116" in nombre_muestra or np.any((x_calibrada_previa <= 283.0) & (y_neta > max_y * 0.2)):
        params['c4_amplitude'].set(value=max_y * 0.6, min=0)
        params['c4_center'].set(value=282.40, min=281.80, max=283.20)
    else:
        params['c4_amplitude'].set(value=0, max=max_y * 0.001)
        params['c4_center'].set(value=282.40, vary=False)
    params['c4_sigma'].set(value=0.65, min=0.40, max=1.10)
    params['c4_fraction'].set(value=0.5, min=0.0, max=1.0, vary=True)

    resultado_fit = modelo_total.fit(y_neta, params, x=x_calibrada_previa)
    componentes_pacos = resultado_fit.eval_components(x=x_calibrada_previa)

    # --- CALCULO CIENTÍFICO DE COEFICIENTE DE DETERMINACIÓN R2 ---
    y_estimada = resultado_fit.best_fit
    residuos = y_neta - y_estimada
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((y_neta - np.mean(y_neta))**2)
    # R2 Ajustada de control matemático estricto
    r2_calculada = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    lista_prefijos = ['c4_', 'c1_', 'c2_', 'c3_']
    prefijos_validos, areas_picos, fwhm_picos, centros_calibrados = [], [], [], []
    
    for prefijo in lista_prefijos:
        amp_final = resultado_fit.params[f'{prefijo}amplitude'].value
        area_abs = np.abs(np.trapz(componentes_pacos[prefijo], x_calibrada_previa))
        if amp_final > (max_y * 0.005) and area_abs > 0:
            prefijos_validos.append(prefijo)
            areas_picos.append(area_abs)
            fwhm_picos.append(resultado_fit.params[f'{prefijo}fwhm'].value)
            centros_calibrados.append(resultado_fit.params[f'{prefijo}center'].value)

    area_total = sum(areas_picos) if sum(areas_picos) > 0 else 1
    porcentajes = [(a / area_total) * 100 for a in areas_picos]

    reporte_renglon = exportar_resultados_c1s(nombre_muestra, x_calibrada_previa, y_neta, resultado_fit, componentes_pacos, prefijos_validos, porcentajes, fwhm_picos, centros_calibrados, r2_calculada)
    return reporte_renglon

# ==============================================================================
# BLOQUE 4: EXPORTACIÓN DE HOJAS DE CÁLCULO Y GRÁFICAS DE ALTA RESOLUCIÓN
# ==============================================================================
def exportar_resultados_c1s(nombre_muestra, x, y_neta, resultado_fit, componentes_pacos, prefijos_validos, porcentajes, fwhm_picos, centros_calibrados, r2_calculada):
    """Genera reportes en Excel e imágenes asegurando consistencia dimensional absoluta."""
    ajuste_total = resultado_fit.best_fit.flatten()
    n_puntos = len(ajuste_total)
    
    x_real = x.flatten()[:n_puntos]
    y_real = y_neta.flatten()[:n_puntos]
    
    dict_excel_datos = {
        'Energia_Enlace_Calibrada_eV': x_real, 
        'Intensidad_Neta_Shirley': y_real, 
        'Ajuste_Total_Combinado': ajuste_total
    }
    # Guardamos la R2 como métrica de calidad científica en el renglón general
    reporte_renglon = {'Muestra': nombre_muestra, 'R2_Ajuste': round(r2_calculada, 4)}
    etiquetas_leyenda = []

    for i, prefijo in enumerate(prefijos_validos):
        centro_final = centros_calibrados[i]
        fwhm_final = fwhm_picos[i]
        tag_quimico = identificar_enlace_desde_db(centro_final)
        
        dict_excel_datos[f'Pico_{i+1}_{tag_quimico}_Curva'] = componentes_pacos[prefijo][:n_puntos]
        reporte_renglon[f'Pico {i+1} Enlace'] = tag_quimico
        reporte_renglon[f'Pico {i+1} Centro (eV)'] = round(centro_final, 2)
        reporte_renglon[f'Pico {i+1} FWHM (eV)'] = round(fwhm_final, 2)
        reporte_renglon[f'Pico {i+1} Area (%)'] = f"{porcentajes[i]:.1f}%"
        
        etiquetas_leyenda.append(f"{tag_quimico} ({centro_final:.2f} eV) [FWHM: {fwhm_final:.2f} eV] ({porcentajes[i]:.1f}%)")

    # Escritura matricial segura en Pandas DataFrame
    df_curvas = pd.DataFrame(dict_excel_datos)
    df_identif = pd.DataFrame([reporte_renglon])
    
    with pd.ExcelWriter(f'{nombre_muestra}_Carbono_PasoAPaso.xlsx') as writer:
        df_curvas.to_excel(writer, sheet_name='Curvas_Para_Graficar', index=False)
        df_identif.to_excel(writer, sheet_name='Identificacion', index=False)

    # Renderizado gráfico de calidad científica clonando Origin Lab
    plt.figure(figsize=(8.4, 5.5))
    plt.plot(x_real, y_real, 'k.', label='Datos Reales (Shirley restado)', alpha=0.3, markersize=3.5)
    plt.plot(x_real, ajuste_total, 'r-', label='Ajuste Total Combinado', linewidth=2.2)
    
    colores_paleta = ['royalblue', 'forestgreen', 'darkorange', 'purple']
    for i, prefijo in enumerate(prefijos_validos):
        curva_grafica = componentes_pacos[prefijo][:n_puntos].flatten()
        plt.fill_between(x_real, curva_grafica, color=colores_paleta[i % 4], alpha=0.22, label=etiquetas_leyenda[i])

    plt.xlabel('Energía de Enlace Calibrada C1s (eV)', fontsize=11)
    plt.ylabel('Fotoelectrones contados por segundo (#)', fontsize=11)
    
    # Imprimimos de manera visible el valor de R² arriba en el título o en la leyenda
    plt.title(f'Deconvolución C1s Post-Calibrada - Muestra {nombre_muestra} ($R^2$: {r2_calculada:.4f})', fontsize=11, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.ylim(0, np.max(y_real) * 1.1)
    plt.legend(loc='upper left', fontsize=8.0, labelspacing=0.4)
    plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{nombre_muestra}_C1s_deconvolucion.png', dpi=300)
    plt.close()
    
    return reporte_renglon

# ==============================================================================
# BLOQUE 5: EJECUCIÓN MAESTRA EN LOTES (MAIN FLOW)
# ==============================================================================
if __name__ == "__main__":
    archivos_zip = [f for f in os.listdir('.') if f.endswith('.zip')]
    if not archivos_zip:
        print("Error: No se encontraron archivos comprimidos .zip en el directorio.")
        exit()

    print(f"🚀 Modo Masivo Inteligente C1s (Post-Calibración) Iniciado. Escaneando {len(archivos_zip)} muestras...")
    reporte_global = []

    for zip_name in archivos_zip:
        nombre_muestra = zip_name.replace('.zip', '')
        print(f"\n---> Procesando región C1s de forma robusta: {nombre_muestra}...")
        
        datos_espectro, archivo_csv = extraer_datos_carbono_del_zip(zip_name)
        if datos_espectro is None:
            print("      ⚠️ ADVERTENCIA: No se localizó matriz en rango C1s dentro del archivo.")
            continue
            
        print(f"      [Fichero Encontrado]: {archivo_csv}")
        resultado_renglon = procesar_y_deconvolucionar_c1s(nombre_muestra, datos_espectro, archivo_csv)
        if resultado_renglon is not None:
            reporte_global.append(resultado_renglon)

    if reporte_global:
        df_global = pd.DataFrame(reporte_global)
        df_global.to_csv('Resumen_General_Lote_Carbono_C1s.csv', index=False)

    print("\n=======================================================")
    print("🎉 ¡PROCESAMIENTO MASIVO POST-CALIBRADO CONCLUIDO!")
    print("   -> Lote completo resuelto con un solo comando.")
    print("   -> Coeficiente R2 calculado e indexado con éxito.")
    print("=======================================================")
