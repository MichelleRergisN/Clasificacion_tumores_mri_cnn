# Clasificación de tumores cerebrales en resonancias magnéticas con CNN

Proyecto de Deep Learning con **Keras + TensorFlow** que clasifica cortes de resonancia magnética (MRI) cerebral en cuatro categorías: `glioma`, `meningioma`, `pituitary` y `notumor`.


[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/TU_REPOSITORIO/blob/main/clasificacion_tumores_mri_cnn.ipynb)

<!-- Reemplaza TU_USUARIO y TU_REPOSITORIO por los de tu repositorio para que el botón funcione. -->

## Contenido

- [Problema](#problema)
- [Dataset](#dataset)
- [Modelos](#modelos)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo ejecutarlo](#cómo-ejecutarlo)
- [Resultados](#resultados)
- [Uso del modelo entrenado](#uso-del-modelo-entrenado)
- [Limitaciones](#limitaciones)
- [Posibles mejoras](#posibles-mejoras)
- [Créditos y licencia](#créditos-y-licencia)

## Problema

Dada una imagen de resonancia magnética cerebral, el modelo debe predecir una de cuatro clases:

| Clase | Descripción |
|---|---|
| `glioma` | Tumor originado en las células gliales |
| `meningioma` | Tumor originado en las meninges |
| `pituitary` | Tumor en la glándula pituitaria (hipófisis) |
| `notumor` | Cerebro sin tumor visible |

Detectar y distinguir el tipo de tumor es relevante porque cada uno implica un pronóstico y tratamiento distintos. Como no detectar un tumor es un error más costoso que una falsa alarma, el análisis da prioridad al *recall* por clase y a las matrices de confusión, no solo al *accuracy*.

## Dataset

- **Nombre:** Brain Tumor MRI Dataset
- **Autor:** Msoud Nickparvar
- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (también en [Zenodo](https://zenodo.org/doi/10.5281/zenodo.12735692), licencia CC BY 4.0)
- **Contenido:** imágenes de MRI cerebral en carpetas `Training/` y `Testing/`, una subcarpeta por clase.

El notebook descarga el dataset automáticamente con `kagglehub`. Se combinan las particiones originales y se vuelven a separar de forma estratificada:

| Conjunto | Proporción | Uso |
|---|---|---|
| Entrenamiento | 70 % | Ajuste de pesos |
| Validación | 15 % | Comparar modelos, ajustar hiperparámetros, detención temprana |
| Prueba | 15 % | Evaluación final (una sola vez) |

Preprocesamiento: redimensionado a 160 × 160 px, replicación a 3 canales y normalización dentro del propio modelo (así la función de predicción acepta directamente una imagen).

## Modelos

Se desarrollan tres modelos de forma iterativa. Todos usan capas convolucionales.

| Modelo | Descripción |
|---|---|
| **v1 – CNN base** | 3 bloques `Conv2D` + `MaxPooling2D` y una capa densa. Línea base sin regularización. |
| **v2 – CNN regularizada** | Bloques con doble convolución, `BatchNormalization`, `GlobalAveragePooling2D`, `Dropout`, regularización L2, aumento de datos, pesos de clase, búsqueda de *learning rate*, `EarlyStopping` y `ReduceLROnPlateau`. |
| **v3 – MobileNetV2** | Transferencia de aprendizaje desde ImageNet: primero se entrena la cabeza con la base congelada y luego se hace *fine-tuning* de las últimas capas con *learning rate* bajo. |

El modelo final se elige **solo con el conjunto de validación**. El notebook documenta qué cambios mejoraron el desempeño y por qué.

## Estructura del repositorio

```
.
├── clasificacion_tumores_mri_cnn.ipynb   # Notebook con todo el proyecto
├── README.md
├── modelos/                              # Se genera al ejecutar (checkpoints .keras)
└── modelo_final.keras                    # Se genera al ejecutar (modelo seleccionado)
```

El notebook está organizado en: **1. Introducción · 2. Datos · 3. Desarrollo del modelo · 4. Ajuste del modelo · 5. Resultados · 6. Conclusiones · 7. Aplicación**.

## Cómo ejecutarlo

### Opción A: Google Colab (recomendada)

1. Abre el notebook con el botón de arriba o sube el `.ipynb` a Colab.
2. Activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*.
3. Ejecuta todo: *Entorno de ejecución → Ejecutar todo*.

Si Colab pide credenciales de Kaggle para descargar el dataset, configura tu token de la API de Kaggle o descarga el dataset a mano y asigna la ruta a `DATA_DIR` en la primera celda de configuración.

### Opción B: entorno local

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install tensorflow scikit-learn pandas matplotlib pillow kagglehub jupyter
jupyter notebook clasificacion_tumores_mri_cnn.ipynb
```

Para la descarga con `kagglehub` necesitas credenciales de Kaggle (`~/.kaggle/kaggle.json`). Se recomienda una GPU; en CPU el entrenamiento es considerablemente más lento.

### Parámetros ajustables

Se concentran en la primera celda de configuración: `IMG_SIZE`, `BATCH_SIZE`, `DATA_DIR`, `USE_PRETRAINED` y el número de épocas de cada fase.

## Resultados

<!-- Ejecuta el notebook y copia aquí los valores de la tabla "Comparación de modelos" (sección 5). -->

| Modelo | Parámetros | Val. accuracy | Test accuracy |
|---|---|---|---|
| v1 – CNN base | _pendiente_ | _pendiente_ | _pendiente_ |
| v2 – CNN regularizada | _pendiente_ | _pendiente_ | _pendiente_ |
| v3 – MobileNetV2 | _pendiente_ | _pendiente_ | _pendiente_ |

El notebook también incluye el reporte de clasificación por clase, la matriz de confusión, las confusiones más frecuentes y ejemplos de imágenes mal clasificadas.

## Uso del modelo entrenado

La sección 7 del notebook define `predecir(ruta_imagen)`. Recibe la ruta de una imagen y devuelve la clase predicha, su confianza y las `top_k` clases más probables:

```python
resultado = predecir("ruta/a/imagen_mri.jpg", top_k=3)
print(resultado["clase"], resultado["confianza"])
```

La función carga `modelo_final.keras`, por lo que funciona de forma independiente del entrenamiento mientras el archivo exista.

## Limitaciones

- Se clasifican **cortes 2D individuales**, no estudios completos. Cortes del mismo paciente pueden caer en entrenamiento y prueba a la vez, lo que puede inflar las métricas; el dataset no incluye identificador de paciente.
- Posibles imágenes duplicadas o con etiquetas dudosas en el dataset original.
- Los datos provienen de pocas fuentes; el modelo puede degradarse con otros escáneres, protocolos o poblaciones.
- Reducción a 160 × 160 px: se pierde detalle fino.
- Una sola partición y una sola semilla, sin medir variabilidad entre ejecuciones.

## Posibles mejoras

- Mayor resolución y arquitecturas más grandes (EfficientNet, ResNet, ConvNeXt).
- Eliminar duplicados por *hashing* y separar por paciente si se dispone de esa información.
- Validación cruzada estratificada y varias semillas.
- Explicabilidad con **Grad-CAM** para verificar que el modelo atiende a la región del tumor.
- Calibración de probabilidades y rechazo de predicciones de baja confianza.
- Segmentación del tumor (por ejemplo, U-Net) en lugar de solo clasificación.

## Créditos y licencia

- Dataset: Msoud Nickparvar, *Brain Tumor MRI Dataset* (Kaggle / Zenodo, CC BY 4.0). Consulta la página del dataset para los términos de uso y la atribución a sus fuentes originales.
- Modelo preentrenado: MobileNetV2 con pesos de ImageNet (incluida en `keras.applications`).
- Autoría del proyecto: Michelle Rergis Novelo | A01798576
