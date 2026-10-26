# Clasificación de Paisajes mediante Redes Neuronales Convolucionales (CNN)

Este repositorio contiene el notebook y la documentación técnica correspondiente al desarrollo de un sistema de Visión por Computadora basado en Redes Neuronales Convolucionales (CNN) y Perceptrón Multicapa (MLP) para la clasificación automática de imágenes de paisajes en plataformas turísticas y bancos de imágenes.

---

## Integrantes del Equipo

* **Christian Pastran**
* **Alessandro Rosell**
* **Diego Mancilla**
* **Sección:** 001D

---

## Descripción del Proyecto

En el sector turístico y digital, la catalogación manual de recursos fotográficos resulta ineficiente, costosa y susceptible a errores humanos. Este proyecto implementa un modelo de Inteligencia Artificial capaz de automatizar el etiquetado y categorización de imágenes mediante la extracción de características visuales (texturas, bordes, colores) para clasificarlas en entornos urbanos y naturales.

---

## Objetivos y KPIs

### Objetivos Específicos
1. Configurar una variante controlada y balanceada del dataset para asegurar reproducibilidad e irrestricto control de memoria.
2. Diseñar e implementar una arquitectura de CNN que extraiga patrones visuales clave combinada con un clasificador denso (MLP).
3. Evaluar el desempeño global y por categoría empleando métricas estandarizadas de clasificación.

### Metas Deseadas (KPIs)
* **Recall Promedio ($\ge 85\%$):** Minimizar los Falsos Negativos para evitar la omisión o traspapelo de imágenes dentro del catálogo fotográfico.
* **Precision Global ($\ge 80\%$):** Garantizar exactitud en las galerías mostradas al usuario final.
* **F1-Score Ponderado ($\ge 0.82$):** Validar el equilibrio armónico global del clasificador.

---

## Especificaciones del Dataset

El modelo trabaja sobre una variante personalizada del dataset **Intel Image Classification**:

* **Clases Seleccionadas (4):** `buildings`, `forest`, `sea`, `street`.
* **Límite de Muestras por Clase:** Máximo de 2,000 imágenes por categoría para el conjunto de entrenamiento.
* **Distribución de Datos:**
  * **Entrenamiento:** 6,400 imágenes (80% de la variante).
  * **Validación:** 1,600 imágenes (20% de la variante).
  * **Prueba (Test):** 1,922 imágenes independientes.
* **Resolución de Entrada:** Redimensionado uniforme a $64 \times 64$ píxeles.
* **Semilla de Reproducibilidad (Seed):** `42`

---

## Estructura del Proyecto

El código genera y gestiona de forma automática la siguiente estructura de carpetas en Google Colab (`/content/`):

```text
/
├── data/
│   ├── raw/                  # Dataset original comprimido y descompresionado
│   └── processed/            # Dataset filtrado con la variante de 4 clases
│       └── dataset_variante/
│           ├── train/        # Datos balanceados para entrenamiento y validación
│           └── test/         # Datos de evaluación independiente
├── models/                   # Artefactos y pesos del modelo exportado (.h5 / SavedModel)
├── images/                   # Gráficos de entrenamiento y matrices de confusión
└── informe_cnn.ipynb         # Notebook principal de ejecución y análisis
```

---

##️ Tecnologías y Librerías

* **Lenguaje:** Python 3.x
* **Framework Principal:** TensorFlow / Keras
* **Procesamiento y Métricas:** NumPy, Scikit-Learn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`)
* **Visualización:** Matplotlib
* **Gestión de Entorno:** OS, Shutil, Subprocess, Pathlib

---

## Guía de Ejecución

1. **Entorno de Ejecución:** Cargar el notebook `informe_cnn.ipynb` en **Google Colab**.
2. **Carga del Dataset:** Subir el archivo `Intel_Image_Classification.zip` dentro del directorio `/content/data/raw/`.
3. **Ejecución del Flujo:**
   * **Celda 1 (Extracción):** Descompresión segura y verificación determinista de carpetas.
   * **Celda 2 (Variante):** Submuestreo balanceado a 4 clases con semilla `42`.
   * **Celda 3 (Data Pipeline):** Carga de datos mediante `tf.keras.utils.image_dataset_from_directory` y optimización con `cache()` y `prefetch()`.
   * **Análisis EDA & Entrenamiento:** Ejecutar el flujo secuencial para visualizar muestras, entrenar la CNN y evaluar las métricas clave.
