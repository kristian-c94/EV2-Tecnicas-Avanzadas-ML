# EV2-Tecnicas-Avanzadas-ML

## Reproducir el entrenamiento

El notebook utiliza el dataset [Intel Image Classification de Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). El archivo ZIP no se versiona porque ocupa aproximadamente 363 MB. Para obtener la misma entrada usada en este proyecto:

1. Configura la autenticación de [Kaggle API](https://github.com/Kaggle/kaggle-api) con una cuenta que tenga acceso al dataset.
2. Descarga el archivo en la ruta esperada por el notebook:

   ```bash
   mkdir -p data/raw
   kaggle datasets download -d puneet6060/intel-image-classification -p data/raw
   mv data/raw/intel-image-classification.zip data/raw/Intel_Image_Classification.zip
   ```

3. Comprueba que el archivo corresponde al usado en este proyecto:

   ```bash
   sha256sum data/raw/Intel_Image_Classification.zip
   ```

   SHA-256 esperado: `3921405607c3e9d67a3459b6c629894c3a723aa1647f9310b6793da306e89337`.

4. Ejecuta las celdas del notebook [CNN_EV2_Tecnicas_Avanzadas.ipynb](./models/CNN_EV2_Tecnicas_Avanzadas.ipynb) en orden. El notebook extrae el ZIP y crea la variante del dataset con las clases y semilla configuradas.

Los datos descargados y procesados se mantienen localmente dentro de `data/` y no se incluyen en Git.
