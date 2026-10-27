# Deconvolution-XPS-C1S

# XPS-C1s-Deconvolution-Script

Este repositorio contiene un script general de computación numérica en Python desarrollado para la semiautomatización, calibración electrostática y deconvolución multicomponente de espectros de Fotoemisión de Rayos X (**XPS**) enfocados en la región del **Carbono 1s (C1s)**.

El motor de ajuste matemático está diseñado bajo un enfoque de restricciones interdependientes y deltas rígidas de energía, imitando las rutinas convencionales pero eliminando por completo el sesgo visual del operador humano.

# XPS-C1s-Deconvolution-Script

Este repositorio contiene un script general de computación numérica desarrollado colaborativamente mediante asistencia de Inteligencia Artificial (IA) y programación humana. Su objetivo es la semiautomatización, calibración electrostática de carga y deconvolución multicomponente de espectros de Fotoemisión de Rayos X (**XPS**) enfocados en la región del **Carbono 1s (C1s)**.

El motor de ajuste implementa restricciones matemáticas interdependientes y vínculos de parámetros para mitigar el sobreajuste (*overfitting*) numérico, garantizando soluciones espectrales con estricta coherencia física.

## 🔬 Ecosistema de Librerías y Dependencias

La arquitectura del script se fundamenta en un entorno de software libre compuesto por las siguientes librerías científicas especializadas:
*   **NumPy (v<2.0):** Vectorización de tensores, álgebra lineal y manipulación robusta de arreglos numéricos crudos.
*   **SciPy (v<1.4):** Integración numérica iterativa para el fondo y algoritmos de procesamiento digital de señales.
*   **Pandas:** Estructuración, indexación y alineación dimensional segura de curvas espectrales independientes.
*   **LMFIT:** Motor central de optimización no lineal basado en mínimos cuadrados ponderados para el ajuste de picos Pseudo-Voigt 1.
*   **Matplotlib:** Renderizado automatizado de gráficas con calidad de publicación científica de alta resolución.
*   **OpenPyXL:** Escritura directa y automatización de reportes matriciales en formato Excel nativo.

## 🛠️ Arquitectura de Bloques Modular

El script está diseñado en 5 bloques lógicos de alta cohesión:
1.  **Bloque 1:** Indexación química dinámica mediante emparejamiento con bases de datos de referencia XPS.
2.  **Bloque 2:** Motor de extracción inyectiva en lote desde archivos comprimidos `.zip` con purga activa de celdas vacías (`NaN`).
3.  **Bloque 3:** Algoritmo iterativo para línea base Shirley e inicialización de perfiles simétricos Pseudo-Voigt 1 con restricciones relativas de energía.
4.  **Bloque 4:** Exportador automatizado a hojas de cálculo matriciales paso a paso (`.xlsx`) y renderizado gráfico científico a 300 DPI.
5.  **Bloque 5:** Flujo maestro de ejecución automatizada en lotes (*Main Flow*).

## 🚀 Requisitos e Instalación

Para preparar tu entorno local, ejecuta el siguiente comando en tu terminal:

```bash
pip3 install "numpy<2.0" "scipy<1.4" pandas matplotlib lmfit openpyxl
```

## 📂 Modo de Uso

1. Coloca tus archivos comprimidos `.zip` con los espectros crudos en el mismo directorio del script.
2. Asegúrate de tener el archivo de referencias químicas `base_datos_xps.csv` en la carpeta.
3. Ejecuta el procesamiento masivo desde tu terminal de comandos:

```bash
python3 deconvolution_c1s.py
```

## 📊 Entregables Automatizados

Por cada muestra procesada en el lote, el script genera de forma autónoma:
*   **`Muestra_Carbono_PasoAPaso.xlsx`:** Tablas con los vectores de energía calibrada, intensidad neta y el desglose de cada pico por separado.
*   **`Muestra_C1s_deconvolucion.png`:** Gráfica científica a 300 DPI con el ajuste global combinado y las áreas sombreadas correspondientes a cada entorno químico, incluyendo el cálculo del parámetro estadístico $R^2$.
*   **`Resumen_General_Lote_Carbono_C1s.csv`:** Archivo maestro que consolida las métricas numéricas del lote completo.
