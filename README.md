# 🌱 FertiIA — Análisis de fertilidad del suelo

FertiIA es una aplicación en Python con una interfaz web de Streamlit que estima la clase de fertilidad de una muestra de suelo. Utiliza un modelo de aprendizaje automático **Random Forest**, entrenado con el CSV del proyecto.

## ¿Qué hace el programa?

1. Carga automáticamente `dataset1.csv`, valida su contenido y entrena el modelo.
2. Permite seleccionar cantidades bajas, medias o altas para los 12 componentes: nitrógeno, fósforo, potasio, pH, conductividad eléctrica, carbono orgánico, azufre, zinc, hierro, cobre, manganeso y boro.
3. Al presionar **Analizar fertilidad**, muestra la clase estimada y las probabilidades de clasificación.
4. Presenta gráficas para comparar la muestra con los datos de entrenamiento y explorar cada componente.
5. Permite descargar el resultado en CSV, consultar las métricas del modelo y cargar otro dataset compatible.

Se ejecuta en tu computador y se utiliza desde el navegador. No necesitas una cuenta ni un servicio externo de predicción.

## Requisitos

Necesitas:

- **Python 3.10 o superior**, con `pip` y soporte para entornos virtuales. Se recomienda Python 3.12 para facilitar la instalación de las bibliotecas.
- Un navegador actualizado.
- Conexión a Internet para instalar las dependencias la primera vez.

Puedes descargar Python desde [python.org](https://www.python.org/downloads/). En Windows, marca **Add Python to PATH** durante la instalación.

## Instalar y ejecutar

### 1. Descargar y localizar la aplicación

Descarga el proyecto y descomprime el ZIP si corresponde. Si descargaste el repositorio completo, entra en:

```text
Proyectos/Desafio de IA/interfaz_fertilidad/
```

Abre una terminal en esa carpeta. Debes encontrar `app.py`, `requirements.txt` y `dataset1.csv` juntos. Si la descarga contiene solamente la carpeta de la aplicación, entra directamente en ella.

### 2. Crear un entorno e instalar las dependencias

El entorno virtual mantiene las bibliotecas del proyecto separadas de las de otros programas. Créalo en tu equipo: una carpeta `venv` incluida en una descarga puede pertenecer a otro sistema y no funcionar.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux o macOS:**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Estos comandos usan directamente el Python del entorno; no es necesario activarlo.

### 3. Iniciar la aplicación

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Linux o macOS:**

```bash
./.venv/bin/python -m streamlit run app.py
```

Espera a que termine el entrenamiento inicial. Streamlit suele abrir el navegador automáticamente; si no lo hace, visita **http://localhost:8501** o la dirección `Local URL` que aparezca en la terminal.

Mantén la terminal abierta mientras uses el programa. Para detenerlo, presiona **Ctrl+C** en ella.

### 4. Abrirlo nuevamente

Las siguientes veces, abre una terminal en la misma carpeta y ejecuta únicamente el comando de inicio del paso 3. No necesitas reinstalar las dependencias ni crear el entorno otra vez.

## Cómo usar FertiIA

1. Comprueba que el panel lateral indique **Dataset cargado y modelo listo**. El CSV incluido tiene **880 registros y 12 variables de entrada**.
2. En **Predicción → Analiza una muestra de suelo**, selecciona una cantidad para cada componente. El número entre paréntesis es el valor que recibirá el modelo.
3. Presiona **Analizar fertilidad**.
4. Revisa primero el resultado y luego las probabilidades y comparaciones. En el explorador puedes elegir un componente para ver su distribución.
5. Usa **Descargar resultado en CSV** para guardar el análisis.

La pestaña **Gráficas y análisis** contiene las métricas y visualizaciones generales del modelo; **Ayuda al usuario** explica su funcionamiento. Si cambias una cantidad, debes volver a analizar para obtener un resultado actualizado.

### Usar otro CSV

En el panel lateral, utiliza **Sube dataset1.csv** para seleccionar un archivo `.csv` (puede tener otro nombre). Debe conservar los componentes y el significado de las clases del original, con este encabezado:

```csv
N,P,K,pH,EC,OC,S,Zn,Fe,Cu,Mn,B,Output
```

Usa comas para separar columnas y punto para los decimales. Todas las celdas deben contener números finitos, sin valores vacíos. `Output` es la clase conocida de cada registro de entrenamiento y debe incluir las tres clases originales, `0`, `1` y `2`, con suficientes registros para entrenar.

La aplicación actualiza el modelo y las opciones al cambiar el archivo. La subida no sobrescribe el CSV original. Para regresar a él, retira el archivo del control de subida. Sin un dataset válido no se habilita la predicción.

## Solución de problemas

| Problema | Qué hacer |
|---|---|
| `python`, `python3` o `py` no se reconoce | Instala Python y vuelve a abrir la terminal. En Windows verifica que esté añadido al PATH. |
| No se puede crear el entorno en Ubuntu/Debian | Instala el soporte con `sudo apt install python3-venv` y repite el paso 2. |
| No se encuentra `app.py` o `requirements.txt` | Abre la terminal en la carpeta `interfaz_fertilidad`, donde están esos archivos. |
| `No module named streamlit` | Instala las dependencias con el Python de `.venv` y usa ese mismo ejecutable para iniciar. |
| No se encuentra el dataset | Coloca `dataset1.csv` junto a `app.py` o sube un CSV compatible desde el panel lateral. |
| El CSV se rechaza | Revisa el mensaje mostrado: columnas, celdas vacías, tipos numéricos y clases de `Output`. |
| El puerto 8501 está ocupado | Añade `--server.port 8517` al comando de inicio y abre `http://localhost:8517`. |
| El navegador no se abre automáticamente | Copia la dirección `Local URL` de la terminal en el navegador. |

## ¿Qué incluye?

- `app.py`: aplicación web.
- `dataset1.csv`: 880 registros originales con 12 variables y `Output`.
- `requirements.txt`: dependencias.
- `test_app.py`: pruebas automatizadas del dataset y la interfaz.
- `README.md`: descripción, instalación y uso.

Al iniciar, la aplicación entrena automáticamente el clasificador con el CSV incluido. No hace falta subir otro archivo. Si se desea, se puede subir un CSV alternativo desde la barra lateral.

## Metodología

Separación estratificada 80/20; SMOTE sobre entrenamiento; StandardScaler; RandomForestClassifier de 100 árboles con `random_state=42`. Las métricas se calculan con los datos de prueba.

> La salida conserva las clases originales 0, 1 y 2. Los nombres y colores se muestran únicamente cuando una fuente documentada está vinculada a la versión exacta del CSV; no constituyen diagnóstico agronómico certificado.

## Interfaz y datos

- Los 12 componentes tienen tres opciones calculadas con `quantile([0.25, 0.5, 0.75])` sobre el CSV activo, con interpolación lineal. Se muestra la representación decimal mínima que recupera exactamente el valor `float64` usado por el modelo, sin truncarlo a 12 cifras. Baja/Media/Alta no son umbrales agronómicos.
- Entrenamiento y muestra usan el orden `N,P,K,pH,EC,OC,S,Zn,Fe,Cu,Mn,B` y `float64`, para conservar percentiles fraccionarios incluso en columnas originalmente enteras. Output conserva sus códigos enteros.
- El CSV se busca junto a `app.py`, independientemente de la carpeta desde donde se ejecute. Se rechazan columnas faltantes, adicionales o duplicadas, celdas vacías, texto, infinitos, objetivos distintos de 0/1/2 y clases insuficientes para partición estratificada/SMOTE.
- Cada contenido CSV tiene una huella SHA-256. Cambiarlo reinicia menús y resultados, y obtiene un modelo entrenado con esos mismos bytes. Quitar la subida restaura el CSV local. Volver a entrenar elimina el resultado anterior. Cambiar un componente también elimina la predicción anterior.
- El resultado precede a las probabilidades, comparación porcentual y explorador de distribuciones por componente. Se conservan descarga CSV, navegación, métricas, importancia de variables, correlaciones y matriz de confusión. Las gráficas no prueban causalidad.

## Procedencia de las clases y límites de evidencia

La presente versión incorpora `clases.json` **solo para la huella SHA-256 exacta** del `dataset1.csv` incluido. Esta configuración habilita los nombres y colores en la interfaz sin alterar `Output`, el entrenamiento ni el modelo:

| Código | Interpretación publicada | Color |
|---|---|---|
| 0 | SUELO MENOS FÉRTIL | Rojo `#DC3545` |
| 1 | SUELO FÉRTIL | Amarillo `#EAB308` |
| 2 | SUELO ALTAMENTE FÉRTIL | Verde `#16A34A` |

**Fuentes complementarias de la procedencia:**

- [Dataset Kaggle Soil Fertility](https://www.kaggle.com/datasets/rahuljaiswalonkaggle/soil-fertility-dataset).
- [Reproducción pública de registros y clases](https://www.scribd.com/document/1074655890/Vertopal-com-Notebook-3-Soil-Fetility-and-Texture-Ml): exhibe las mismas cinco filas iniciales (valores de las 12 variables y códigos 0,0,0,0,1) y los conteos 401/440/39.
- [AgriWiseGP — Soil Quality (README)](https://github.com/AgriWiseGP/ML---Soil-Quality): indica explícitamente `0 Less Fertile`, `1 Fertile`, `2 Highly Fertile` y enlaza el Soil Fertility Dataset de Kaggle.
- [Rao y Reddy, *Scientific Reports* (2026)](https://www.nature.com/articles/s41598-026-50366-9): utiliza los tres niveles y los mismos conteos.

**Salvedad importante:** la tabla de estadísticos descriptivos del artículo de *Scientific Reports* difiere de los rangos del CSV local (p. ej. N y pH). Por ello, no se afirma que sea exactamente el archivo analizado por ese artículo, ni se transfieren automáticamente las unidades de su tabla. La identificación de las etiquetas se sustenta conjuntamente en la coincidencia de filas del dataset público y el diccionario explícito de la implementación que enlaza el dataset original. Es una interpretación documentada del conjunto, no una validación agronómica de muestras concretas.

Huella SHA-256 de `dataset1.csv`: `930e0731ce747590aba71ca6d6ffd2f6e77202f812e01fa3e9c19a5b9e5fe5a4`.

Al cargar cualquier CSV distinto, su huella deja de coincidir: la interfaz **vuelve a las etiquetas numéricas neutrales**. Solo debe crearse un nuevo `clases.json` cuando se documente adecuadamente la correspondencia de ese nuevo archivo. Esta protección impide asignar clasificaciones sin respaldo a datos ajenos.

Las probabilidades proceden de `RandomForestClassifier.predict_proba()`, no de valores decorativos. La clase 2 tiene solo 39 muestras originales; interpreta su resultado con precaución. Ninguna salida sustituye un análisis profesional del suelo.

## Evaluación del Random Forest local

Resultados reproducidos con el CSV original y partición estratificada 80/20 (`random_state=42`). No son las métricas del modelo de aprendizaje profundo del artículo.

| Clase | Precisión | Recall | F1 | Casos de prueba |
|---|---:|---:|---:|---:|
| 0 | 92,59% | 93,75% | 93,17% | 80 |
| 1 | 90,48% | 86,36% | 88,37% | 88 |
| 2 | 36,36% | 50,00% | 42,11% | 8 |

Exactitud global: **88,07%**. F1 macro: **74,55%**. Matriz de confusión (filas = clase real, columnas = clase predicha, orden 0/1/2):

```text
          Pred. 0  Pred. 1  Pred. 2
Real 0       75        5        0
Real 1        5       76        7
Real 2        1        3        4
```

La interfaz recalcula la tabla por clase y la matriz para cada dataset. La clase 2 tiene 39 registros totales, 31 de entrenamiento y solo 8 de prueba: acierta 4 de esos 8 y produce 7 falsos positivos desde la clase 1. La exactitud global oculta esta debilidad; SMOTE no añade observaciones reales ni garantiza generalización.

## Interpretación y limitaciones

- Random Forest combina 100 árboles. Se conserva el procedimiento del proyecto: división estratificada, SMOTE solo sobre entrenamiento y StandardScaler ajustado en entrenamiento. No se reproduce el modelo AMIFP del artículo.
- La probabilidad mostrada corresponde a la clase predicha usando `clf.classes_`. Las barras relacionan cada probabilidad con su código real, sin suponer un orden. No son probabilidades calibradas ni garantías de acierto.
- El explorador presenta una variable por vez con la muestra y su distribución. La comparación porcentual indica qué fracción de registros tiene un valor igual o menor; no mide porcentaje de fertilidad.
- Los menús discretizan cada variable en tres valores y pueden generar combinaciones poco representadas en los datos. No reemplazan una medición real.
- El dataset es pequeño, desbalanceado y contiene valores extremos. Se conservan los datos originales; no se eliminaron valores para forzar coincidencia con el artículo. No hay validación externa ni geográfica.
- Importancias, correlaciones y probabilidades no demuestran causalidad. **FertiIA es una herramienta académica y no reemplaza un análisis profesional de laboratorio ni asesoría agronómica.**

## Verificaciones

Desde la carpeta de la aplicación, después de instalar las dependencias:

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

**Linux o macOS:**

```bash
./.venv/bin/python -m unittest -v
```

Ocho pruebas automatizadas: CSV real y contrato numérico; rechazo de datos inválidos; 12 menús y predicción; ausencia/reemplazo del dataset; cambio de CSV en una misma sesión; precisión decimal y soporte de clase minoritaria; metadatos vinculados por huella y probabilidades con clases reordenadas; recuperación de errores al predecir. Los colores se prueban también con metadatos ficticios; la versión entregada habilita además las etiquetas documentadas solo para el CSV original. No requieren dependencias nuevas.
