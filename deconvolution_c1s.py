import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import PseudoVoigtModel

def identificar_enlace_desde_db(energia_calculada, ruta_db='base_datos_xps.csv'):
    """Busca en el archivo CSV aplicando las desviaciones restadas al Min y sumadas al Max"""
    if not os.listdir('.') or ruta_db not in os.listdir('.'):
        ruta_alternativa = '../../base_datos_xps.csv'
        if os.path.exists(ruta_alternativa):
            ruta_db = ruta_alternativa
        else:
            return f"Señal no indexada"
            
    try:
        df_db = pd.read_csv(ruta_db, encoding='latin1')
        for _, fila in df_db.iterrows():
            nombre = fila['Grupo_Funcional']
            ref_ev = float(fila['Energia_eV'])
            desviacion_min = float(fila['Min_eV'])  # Lee el 0.3 de tu tabla
            desviacion_max = float(fila['Max_eV'])  # Lee el 0.4 de tu tabla
            
            # NUEVA FÓRMULA DINÁMICA: Calcula los límites reales en base al promedio
            limite_inferior = ref_ev - desviacion_min
            limite_superior = ref_ev + desviacion_max
            
            # Evalúa si la energía real calculada cae dentro del rango de tolerancia
            if limite_inferior <= energia_calculada <= limite_superior:
                return f"{nombre}"
    except Exception as e:
        pass
        
    return f"Señal no indexada"

# 1. Buscar TODOS los archivos .zip en la carpeta actual
archivos_zip = [f for f in os.listdir('.') if f.endswith('.zip')]
if not archivos_zip:
    print("Error: No se encontraron archivos .zip en esta carpeta.")
    exit()

print(f"¡Modo masivo activado! Se detectaron {len(archivos_zip)} muestras para procesar.")
reporte_global = []

# ==============================================================================
# BUCLE MAESTRO: PROCESAR CADA ZIP DE FORMA AUTOMÁTICA
# ==============================================================================
for zip_name in archivos_zip:
    nombre_muestra = zip_name.replace('.zip', '')
    print(f"\n---> Deconvolucionando muestra: {nombre_muestra}...")
    
    try:
        with zipfile.ZipFile(zip_name, 'r') as zf:
            lista_archivos = zf.namelist()
            archivo_datos_lista = [f for f in lista_archivos if ('C1s' in f or 'c1s' in f or f.endswith('.csv') or f.endswith('.txt')) 
                                   and not f.lower().endswith('.pdf') and 'metadata' not in f.lower()]
            
            target_file = archivo_datos_lista[0]
            with zf.open(target_file) as f:
                datos = np.genfromtxt(f, delimiter=',', skip_header=1)
                if np.isnan(datos).all() or datos.shape[1] < 2:
                    f.seek(0)
                    datos = np.genfromtxt(f, skip_header=1)
    except Exception as e:
        print(f"      Saltando muestra por error de lectura: {e}")
        continue

    x_completo = datos[:, 0]
    y_completo = datos[:, 1]

    if x_completo[0] < x_completo[-1]:
        x_completo = x_completo[::-1]
        y_completo = y_completo[::-1]

    # --- RECORTE ESTRICTO A LA VENTANA ---
    mascara = (x_completo >= 281.5) & (x_completo <= 291.0)
    x_crudo = x_completo[mascara]
    y = y_completo[mascara]

    # --- RESTAR LÍNEA BASE ---
    fondo = np.linspace(y[-1], y[0], len(y)) 
    y_neta = y - fondo
    y_neta = np.clip(y_neta, 0, None)

    # ==============================================================================
    # NUEVO: CORRECCIÓN AUTOMÁTICA POR ALTA CONDUCTIVIDAD (AUTO-CALIBRACIÓN)
    # ==============================================================================
    # Localizamos en qué celda de energía cayó el punto real más alto de tu espectro
    indice_maximo = np.argmax(y_neta)
    energia_maxima_real = x_crudo[indice_maximo]
    
    # Calculamos el desfase respecto a la referencia teórica estándar (284.50 eV)
    desfase_carga = energia_maxima_real - 284.50
    
    # Movemos el eje X completo en bloque para corregir el corrimiento instrumental
    x = x_crudo - desfase_carga
    # ==============================================================================

    # --- CONFIGURACIÓN DEL MODELO DE 2 PICOS SIMÉTRICOS CON INICIALIZACIÓN FUERTE ---
    modelo = PseudoVoigtModel(prefix='cc_') + PseudoVoigtModel(prefix='tailing_')
    max_y = np.max(y_neta)
    params = modelo.make_params()

    # Pico 1: Tu C-C perfecto intocable (~284.5 eV)
    params['cc_amplitude'].set(value=max_y * 0.6, min=0)
    params['cc_center'].set(value=284.5, min=284.3, max=284.8) 
    params['cc_sigma'].set(value=0.45, min=0.3, max=0.55)  
    params['cc_fraction'].set(value=0.3, vary=False)

    # Pico 2: Tailing / Hombro de Oxígeno residual (~285.5 eV)
    params['tailing_amplitude'].set(value=max_y * 0.45, min=0)
    params['tailing_center'].set(value=285.4, min=285.0, max=286.5) 
    params['tailing_sigma'].set(value=0.7, min=0.4, max=1.4)        
    params['tailing_fraction'].set(value=0.3, vary=False)

    # Ajuste matemático con pesos
    pesos = np.sqrt(y_neta + 1)
    resultado_fit = modelo.fit(y_neta, params, x=x, weights=pesos)
    pacos = resultado_fit.eval_components(x=x)

    centro_cc = resultado_fit.params['cc_center'].value
    centro_tail = resultado_fit.params['tailing_center'].value

    # --- IDENTIFICACIÓN DESDE LA DB ---
    tag_p1 = identificar_enlace_desde_db(centro_cc)
    tag_p2 = identificar_enlace_desde_db(centro_tail)
    
    label_p1 = f"{tag_p1} ({centro_cc:.2f} eV)"
    label_p2 = f"{tag_p2} ({centro_tail:.2f} eV)"

    a_cc = np.trapz(pacos['cc_'], x)
    a_tail = np.trapz(pacos['tailing_'], x)
    a_tot = a_cc + a_tail
    pct_cc = (a_cc / a_tot) * 100
    pct_tail = (a_tail / a_tot) * 100

    # --- GUARDAR EXCEL DE ESTA MUESTRA INDIVIDUAL ---
    df_datos = pd.DataFrame({
        'Energia_Enlace_Calibrada_eV': x,  # Refleja el eje corregido en tu Excel
        'Intensidad_Neta_SinFondo': y_neta,
        'Ajuste_Total_Combinado': resultado_fit.best_fit,
        'Pico_1_CC_Curva': pacos['cc_'],
        'Pico_2_Tailing_Curva': pacos['tailing_']
    })
    
    df_reporte = pd.DataFrame([{
        'Muestra': nombre_muestra,
        'Enlace Pico 1': tag_p1, 'Centro Calibrado 1 (eV)': round(centro_cc, 2), 'Área 1 (%)': f"{pct_cc:.1f}%",
        'Enlace Pico 2': tag_p2, 'Centro Calibrado 2 (eV)': round(centro_tail, 2), 'Área 2 (%)': f"{pct_tail:.1f}%"
    }])

    with pd.ExcelWriter(f'{nombre_muestra}_Datos_PasoAPaso.xlsx') as writer:
        df_datos.to_excel(writer, sheet_name='Curvas_Para_Graficar', index=False)
        df_reporte.to_excel(writer, sheet_name='Identificacion', index=False)

    # Acumular en la lista del reporte de Excel global
    reporte_global.append({
        'Muestra': nombre_muestra,
        'Pico 1': tag_p1, 'Centro 1': round(centro_cc, 2), 'Área 1': f"{pct_cc:.1f}%",
        'Pico 2': tag_p2, 'Centro 2': round(centro_tail, 2), 'Área 2': f"{pct_tail:.1f}%"
    })

    # --- GUARDAR GRÁFICA LIMPIA ---
    plt.figure(figsize=(7.5, 5))
    plt.plot(x, y_neta, 'k.', label='Datos Reales (Fondo restado)', alpha=0.4, markersize=3.5)
    plt.plot(x, resultado_fit.best_fit, 'r-', label='Ajuste Total Combinado', linewidth=2.2)

    plt.fill_between(x, pacos['cc_'], color='forestgreen', alpha=0.25, label=f'{label_p1} ({pct_cc:.1f}%)')
    plt.fill_between(x, pacos['tailing_'], color='darkorange', alpha=0.25, label=f'{label_p2} ({pct_tail:.1f}%)')

    plt.xlabel('Energía de Enlace Calibrada (eV)', fontsize=11)  # Título de eje modificado
    plt.ylabel('Fotoelectrones contados por segundo (#)', fontsize=11)
    plt.title(f'Deconvolución C1s Calibrada - Muestra {nombre_muestra}', fontsize=12, fontweight='bold')
    plt.gca().invert_xaxis()
    plt.ylim(0, max_y * 1.08)
    plt.legend(loc='upper left', fontsize=8, labelspacing=0.3)
    plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{nombre_muestra}_deconvolucion.png', dpi=300)
    plt.close()

# 2. Guardar el Gran Resumen de todas las muestras juntas
df_global = pd.DataFrame(reporte_global)
df_global.to_csv('Resumen_General_Lote_XPS.csv', index=False)

print("\n=======================================================")
print("¡PROCESAMIENTO MULTI-MUESTRA TERMINADO CON AUTO-CALIBRACIÓN!")
print("-> Se generaron las imágenes de deconvolución de CADA muestra.")
print("-> Se creó el archivo maestro central: Resumen_General_Lote_XPS.csv")
print("=======================================================")