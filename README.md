# 🧠 Fundamentos de Clasificación de Imágenes: ML clásico vs CNN


## 1. Redes Neuronales Convolucionales (CNN) para Clasificación
**¿Qué son?**
Imagina que enseñamos a un niño a reconocer animales:
1. **Primero** ve manchas de color (bordes/texturas) -> **Capas convolucionales iniciales**
2. **Luego** identifica orejas, patas -> **Capas medias**
3. **Finalmente** distingue "perro" vs "gato" -> **Capas finales**

**Partes clave de una CNN**:
- **Capa Convolucional**:
    ```python
    Conv2D(32, (3,3), activation='relu') # 32 filtros de 3x3
    ```

    # Cada filtro detecta patrones distintos (bordes, colores, texturas).
- **Capa Pooling** (ej: MaxPooling):
    ```python
    MaxPooling2D(2,2) # Reduce dimensiones a la mitad

    # Como hacer zoom hacia afuera para ver la imagen mas general.

**Ejemplo completo**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)), # Entrada: imagen 100x100 RGB
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(), # Aplana los features para la capa densa
    Dense(128, activation='relu'),
    Dense(2, activation='softmax') # 2 clases: probabilidades
])
```

## 2. Dataset de imagenes
**Estructura ideal**:
```python
/dataset
    /manzanas
        img1.jpg # [Foto de manzana roja]
        img2.jpg # [Foto de manzana verde]
    /platanos
        img1.jpg # [Foto plátano maduro]
```

**Buenas prácticas**:
- **Balance**: 50 imagenes de manzanas y 50 imagenes de plátanos (no 10 vs 90)-
- **Variabilidad**:
    - Diferentes angulos
    - Distintas iluminaciones
    

## 3. Operaciones morfologicas en procesamiento de imagenes
**¿Qué son?**

Técnicas para modificar la estructura geométrica de objetos en imagenes binarias o en escala de grises, basadas en la teoría de conjuntos.
Son fundamentales tanto para el preprocesamiento en ML clásico como para capas iniciales en CNNs. 

### Operaciones básicas (Kernel: `np.ones((3,3))`)

| Operación      | Efecto Visual                               | Representación | Código OpenCV                                  | Aplicación típica          |
|----------------|---------------------------------------------|----------------|-----------------------------------------------|----------------------------|
| **Erosión**    | Adelgaza los objetos, reduce su tamaño      | ⬛→▪️        | `cv2.erode(img, kernel, iterations=1)`        | Eliminar ruido pequeño     |
| **Dilatación** | Engrosa los objetos, aumenta su tamaño      | ▪️→⬛        | `cv2.dilate(img, kernel, iterations=1)`       | Unir partes rotas          |
| **Apertura**   | Elimina salientes finos y ruido exterior    | ⚫→🔘        | `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)` | Limpieza de fondos         |
| **Cierre**     | Rellena entrantes finos y huecos pequeños   | 🔘→⚫        | `cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)` | Rellenar huecos           |

### Operaciones avanzadas
```python
# Gradiente morfologico (Bordes)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Top Hat (Detectar objetos claros sobre fondo oscuro)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Black Hat (Detectar objetos oscuros sobre fondo claro)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```

### **Cómo se usan en ML/ CNN?**
- **1.Para ML clásico**:
- Preprocesamiento obligatorio antes de extraer HOG/LBP
- Ejemplo de detección de texto:
```python
# Binarización  + apertura para mejorar OCR
_, binary = cv2.treshold(img, 0, 255, cv2.THRESHOLD_BINARY_INV+cv2.TRESH_OTSU)
processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
```
- **2.En CNNs**:
- Las primeras capas convolucionales aprenden operaciones similares
- Pueden inicializarse kernels para simular efectos morfológicos
```python
# Kernel de erosión aprendible en una CNN
Conv2D(32, (3x3), activation='relu', kernel_initializer='he_normal')
```
- Ejemplo completo: **Limpieza de huellas dactilares**
```python
import cv2
import numpy as np
# 1. Binarización
_, binary = cv2.treshold(img, 128, 255, cv2.THRESHOLD_BINARY_INV)

# 2. Operaciones morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # Rellena huecos
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel) # Elimina ruido

# 3. Extracción de características
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
**Parámetros Clave:**

* kernel_size: Mayor tamaño afecta más la imagen (típico 3x3 o 5x5)

* iterations: Número de aplicaciones (usar 1-3 normalmente)

* kernel_shape:

    * cv2.MORPH_RECT (rectangular)

    * cv2.MORPH_ELLIPSE (elíptico)

    * cv2.MORPH_CROSS (cruz)

## 4. Descriptores de imágenes en OpenCV
**¿Qué son?**
Los descriptores son métodos matemáticos para extraer caracterísiticas numéricas clave de una imagen, 
permitiendo representar su contenido de forma compacta. Son esenciales para ML clásico, donde el feature engineering es manual.

### Tipos principales de Descriptores
#### 1. **HOG (Histogram of Oriented Gradients)**
**Concepto**:
Analiza la dirección e intensidad de los gradientes (cambios de color/intensidad) en regiones locales de la imagen.

**Código**:
```python
import cv2
hog = cv2.HOGDescriptor(
    win_size=(64, 64),  # Tamaño de la imagen de entrada
    block_size=(16, 16),  # Tamaño del bloque sobre el que se calcula
    block_stride=(8, 8),  # Desplazamiento entre bloques
    cell_size=(8, 8),     # Tamaño de la celda para histogramas
    nbins=9               # Número de bins del histograma
)
features = hog.compute(img)  # Vector de características (3780 valores para 64x64)
```
#### 2. **LBP (Local Binary Patterns)**
**Concepto**
Describe texturas comparando cada píxel con sus vecinos, generando un código binario.

**Código**:
```python
from skimage.feature import local_binary_pattern
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(
    image=gray_img, 
    P=n_points,  # Puntos vecinos a considerar
    R=radius,    # Radio
    method='uniform'  # Patrones uniformes
)
hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
```
**Ejemplo binario**:
```python
Píxeles vecinos: [50, 60, 70]  
               [55, 58, 75]  
               [52, 65, 80]  
Umbral (58):    [0, 1, 1]  
                [0, 0, 1]  
                [0, 1, 1]  
LBP Decimal: 01110011 → 115
``` 
#### 3. **SIFT y SURF (Detectores de puntos clave)**
**Concepto**:
Identifican puntos invariantes a rotación y escala, útiles para objetos con patrones distintivos. 

**Ejemplo**:
```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
```
#### 4. **Color histogramas**
**Concepto**:
Describe la distribución de colores en la imagen, útil para objetos con colores característicos.
```python
hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])  # Canal Azul
hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])  # Canal Verde
hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])  # Canal Rojo
```
### Comparativa de descriptores

| Descriptor | ¿Qué es? | Ventajas | Limitaciones | Dimensionalidad Típica | Uso Recomendado |
|------------|----------|----------|--------------|------------------------|-----------------|
| **HOG** (Histograma de Gradientes Orientados) | Captura la distribución de direcciones de bordes en la imagen | • Robustez a cambios de iluminación<br>• Bueno para detectar formas | • Sensible a oclusión<br>• No invariante a rotación | 3780 (para imagen 64x64) | Detección de personas, objetos con forma definida |
| **LBP** (Patrones Binarios Locales) | Analiza la textura comparando cada píxel con sus vecinos | • Cómputo muy rápido<br>• Excelente para texturas<br>• Robusto a iluminación uniforme | • Poco discriminativo para objetos complejos<br>• Sensible al ruido | 256 (histograma básico) | Reconocimiento facial, clasificación de texturas |
| **SIFT** (Transformación de Características Invariante a Escala) | Detecta puntos clave en la imagen y los describe independientemente de escala y rotación | • Invariante a rotación y escala<br>• Robusto a cambios de perspectiva<br>• Altamente distintivo | • Computacionalmente costoso<br>• Anteriormente patentado<br>• Memoria intensiva | 128 por cada punto clave | Reconocimiento de objetos, stitching de imágenes |
| **Color Hist** (Histograma de Color) | Cuenta la frecuencia de cada valor de color en la imagen | • Simple de implementar<br>• Intuitivo<br>• Rápido de calcular | • Ignora información espacial<br>• Muy sensible a cambios de iluminación<br>• No captura formas | 768 (256 bins × 3 canales RGB) | Búsqueda por contenido, clasificación por color | 

## ¿Cómo Usarlos en ML Clásico?

**Pipeline Típico:**

**Preprocesamiento:**
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
```

**Extracción:**
```python
hog_features = hog.compute(equalized)
lbp_features = local_binary_pattern(equalized, 8, 1, 'uniform')
```
**Concatenación:**
```python
final_features = np.concatenate([hog_features.flatten(), lbp_hist])
```
**Ejemplo con SVM:**
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X_train, y_train)  # X_train = [hog_features, lbp_features, ...]
```
## ¿Y en las CNNs?
Las redes neuronales modernas reemplazan los descriptores manuales por:

* Capas convolucionales: Aprenden features automáticamente
* Global Average Pooling: Reduce la dimensionalidad al final

**Ventaja**: No requiere ingeniería manual de características, pero necesita más datos.

```python
# Capas iniciales de una CNN actúan como "descriptores aprendidos"
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    GlobalAveragePooling2D(),  # Reemplaza a HOG/LBP
    Dense(2, activation='softmax')
])
```
## 5. Machine learning VS CNN

### Machine Learning Clásico (como con una SVM): El Artesano de las Características

Piensen en el Machine Learning clásico como si nosotros fuéramos artesanos creando las herramientas que el modelo va a usar.

**Preprocesamiento: El Preparado Manual.** Primero, tomábamos la imagen y la preparábamos un poco, por ejemplo, convirtiéndola a blanco y negro para simplificarla. Luego, venía la parte crucial: **la extracción manual de características**. Era como si tuviéramos un conjunto de reglas o recetas (algoritmos como HOG o LBP) que aplicábamos a la imagen para identificar patrones importantes. Por ejemplo, HOG se enfoca en las direcciones de los bordes, y LBP analiza las texturas locales. Nosotros, como ingenieros, decidíamos qué características eran relevantes para el problema.

**Entrenamiento: Aprendizaje con Herramientas Hechas a Mano.** Una vez que teníamos estas "herramientas" (los vectores de características HOG o LBP), se las dábamos a un modelo de aprendizaje automático, como una Máquina de Vectores de Soporte (SVM). La SVM aprendía a clasificar las imágenes basándose en estos descriptores que nosotros habíamos creado. Este proceso de entrenamiento solía ser relativamente rápido, incluso en computadoras normales (CPUs), especialmente si no teníamos muchísimos datos.

**Rendimiento: Limitado por Nuestras Herramientas.** El rendimiento que podíamos obtener dependía mucho de qué tan buenas fueran las características que habíamos diseñado manualmente. Con pocos datos, podíamos obtener resultados decentes, alrededor del 80% de precisión en algunos casos. Sin embargo, si las imágenes eran muy complejas o variadas, nuestras "herramientas" manuales podían quedarse cortas, limitando la precisión que podíamos alcanzar.

### Redes Neuronales Convolucionales (CNNs): El Aprendiz que Aprende Solo

Ahora, las CNNs son como tener un aprendiz muy inteligente que aprende a identificar las características importantes por sí mismo, sin que nosotros tengamos que decirle exactamente qué buscar.

**Preprocesamiento: Una Preparación Sencilla.** Con las CNNs, la preparación inicial de la imagen es mucho más simple. A menudo, solo normalizamos los valores de los píxeles (los hacemos estar en un rango específico, como entre 0 y 1). La idea es darle a la red la imagen "casi cruda".

**Entrenamiento: Un Aprendizaje Profundo y Computacionalmente Intensivo.** El "aprendizaje" en las CNNs ocurre a través de muchas capas de procesamiento (las capas convolucionales). Estas capas aprenden automáticamente a detectar patrones cada vez más complejos a partir de los píxeles de la imagen. Es como si la red construyera sus propias "herramientas" de detección de características. Este proceso requiere muchísimos datos y mucha potencia de cálculo, por lo que normalmente necesitamos usar tarjetas gráficas especializadas (GPUs) para que el entrenamiento no tarde una eternidad. Puede llevar horas o incluso días entrenar una CNN potente.

**Rendimiento: Potencialmente Mucho Mayor con Suficientes Datos.** La gran ventaja de las CNNs es que, si les damos suficientes datos, pueden aprender representaciones de las imágenes mucho más ricas y complejas de lo que nosotros podríamos diseñar manualmente. Esto les permite alcanzar niveles de precisión muy altos, a menudo superando el 95% en tareas de clasificación de imágenes complejas. Sin embargo, esta alta precisión generalmente viene de la mano de la necesidad de tener una gran cantidad de datos para que la red aprenda correctamente.

## 6. Métricas de Evaluación

Cuando construimos un modelo de clasificación (por ejemplo, para distinguir entre gatos y perros), necesitamos formas de medir qué tan bien está funcionando. Las **métricas de evaluación** nos proporcionan estas medidas. La **matriz de confusión** es la base para calcular muchas de estas métricas.

### Matriz de Confusión

La matriz de confusión es una tabla que resume el rendimiento de un modelo de clasificación al comparar las predicciones del modelo con las etiquetas reales de los datos de prueba. Para un problema de clasificación binaria (dos clases, como "gato" y "perro"), la matriz de confusión tiene la siguiente estructura:

|                     | Predicho: Positivo | Predicho: Negativo |
|---------------------|--------------------|--------------------|
| **Real: Positivo** | Verdadero Positivo (TP) | Falso Negativo (FN) |
| **Real: Negativo** | Falso Positivo (FP)    | Verdadero Negativo (TN) |

### 1. Exactitud (Accuracy)

**Descripción:** La exactitud mide la proporción de todas las predicciones que fueron correctas. Es la métrica más intuitiva, pero puede ser engañosa en conjuntos de datos con clases desbalanceadas.

**Fórmula:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Código:**
```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(f"Exactitud (Accuracy): {accuracy:.2f}")

cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)
```
### 2.Sensibilidad (Recall o Tasa de Verdaderos Positivos - TPR)

**Descripción:** Mide la proporción de las instancias positivas reales que fueron correctamente identificadas por el modelo. Es importante cuando queremos minimizar los falsos negativos (casos positivos que el modelo predijo como negativos). Por ejemplo, en la detección de enfermedades, queremos un alto recall para no pasar por alto casos positivos.

**Fórmula:**
$$\text{Recall} = \frac{TP}{TP + FN}$$
donde:
* TP (True Positives): Número de instancias positivas que fueron correctamente predichas como positivas.
* FN (False Negatives): Número de instancias positivas que fueron incorrectamente predichas como negativas.

**Código:**
```python
from sklearn.metrics import recall_score

# Etiquetas reales (y_true) y predicciones del modelo (y_pred)
# Ejemplo: 1 representa la clase positiva
y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

# Calcular el recall para la clase positiva (por defecto, etiqueta 1)
recall_positivo = recall_score(y_true, y_pred)
print(f"Sensibilidad (Recall) para la clase positiva (1): {recall_positivo:.2f}")

# Calcular el recall para una clase específica (ej. clase 0)
recall_clase_0 = recall_score(y_true, y_pred, pos_label=0)
print(f"Sensibilidad (Recall) para la clase 0: {recall_clase_0:.2f}")

# Calcular el recall para la clase específica (ej. clase 1)
recall_clase_1 = recall_score(y_true, y_pred, pos_label=1)
print(f"Sensibilidad (Recall) para la clase 1: {recall_clase_1:.2f}")
```
### 3.Especificidad (Tasa de Verdaderos Negativos - TNR)

**Descripción:** Mide la proporción de las instancias negativas reales que fueron correctamente identificadas por el modelo. Es importante cuando queremos minimizar los falsos positivos (casos negativos que el modelo predijo como positivos). Por ejemplo, en un sistema de alarma de incendios, queremos alta especificidad para evitar falsas alarmas.

**Fórmula:**
$$\text{Specificity} = \frac{TN}{TN + FP}$$
donde:
* TN (True Negatives): Número de instancias negativas que fueron correctamente predichas como negativas.
* FP (False Positives): Número de instancias negativas que fueron incorrectamente predichas como positivas.

**Código:**
```python
from sklearn.metrics import confusion_matrix

# Etiquetas reales (y_true) y predicciones del modelo (y_pred)
# Ejemplo: 1 representa la clase positiva, 0 la clase negativa
y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

# Calcular la matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)
# La matriz de confusión para clasificación binaria es:
# [[TN FP]
#  [FN TP]]

# Extraer los valores de la matriz de confusión
TN = cm[0, 0]
FP = cm[0, 1]

# Calcular la especificidad
if (TN + FP) == 0:
    specificity = 0  # Evitar división por cero
else:
    specificity = TN / (TN + FP)

print(f"Especificidad: {specificity:.2f}")
```
### 4.Precisión (Precision)

**Descripción:** Mide la proporción de las instancias que el modelo predijo como positivas que realmente fueron positivas. En otras palabras, de todas las veces que el modelo dijo que era "positivo", ¿cuántas veces acertó? La precisión es importante cuando queremos minimizar los falsos positivos (casos que el modelo clasificó como positivos pero eran negativos). Por ejemplo, en un sistema de recomendación, queremos alta precisión para no recomendar muchos elementos irrelevantes al usuario.

**Fórmula:**
$$\text{Precision} = \frac{TP}{TP + FP}$$
donde:
* TP (True Positives): Número de instancias positivas que fueron correctamente predichas como positivas.
* FP (False Positives): Número de instancias negativas que fueron incorrectamente predichas como positivas.

**Código:**
```python
from sklearn.metrics import precision_score

# Etiquetas reales (y_true) y predicciones del modelo (y_pred)
# Ejemplo: 1 representa la clase positiva
y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

# Calcular la precisión para la clase positiva (por defecto, etiqueta 1)
precision_positivo = precision_score(y_true, y_pred)
print(f"Precisión para la clase positiva (1): {precision_positivo:.2f}")

# Calcular la precisión para una clase específica (ej. clase 0)
precision_clase_0 = precision_score(y_true, y_pred, pos_label=0)
print(f"Precisión para la clase 0: {precision_clase_0:.2f}")

# Calcular la precisión para la clase específica (ej. clase 1)
precision_clase_1 = precision_score(y_true, y_pred, pos_label=1)
print(f"Precisión para la clase 1: {precision_clase_1:.2f}")
```
### 5.F1-Score

**Descripción:** El F1-score es la media armónica de la precisión y el recall. Proporciona una medida equilibrada del rendimiento del modelo, especialmente cuando hay un desequilibrio entre precisión y recall. A diferencia de la media aritmética, la media armónica penaliza los valores extremos; un F1-score alto solo se logra si tanto la precisión como el recall son relativamente altos.

**Fórmula:**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times TP}{2 \times TP + FP + FN}$$
donde:
* TP (True Positives): Número de instancias positivas que fueron correctamente predichas como positivas.
* FP (False Positives): Número de instancias negativas que fueron incorrectamente predichas como positivas.
* FN (False Negatives): Número de instancias positivas que fueron incorrectamente predichas como negativas.

**Código:**
```python
from sklearn.metrics import f1_score

# Etiquetas reales (y_true) y predicciones del modelo (y_pred)
# Ejemplo: 1 representa la clase positiva
y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

# Calcular el F1-score para la clase positiva (por defecto, etiqueta 1)
f1_positivo = f1_score(y_true, y_pred)
print(f"F1-Score para la clase positiva (1): {f1_positivo:.2f}")

# Calcular el F1-score para una clase específica (ej. clase 0)
f1_clase_0 = f1_score(y_true, y_pred, pos_label=0)
print(f"F1-Score para la clase 0: {f1_clase_0:.2f}")

# Calcular el F1-score para la clase específica (ej. clase 1)
f1_clase_1 = f1_score(y_true, y_pred, pos_label=1)
print(f"F1-Score para la clase 1: {f1_clase_1:.2f}")
```
### 6.Área Bajo la Curva ROC (AUC - ROC)

**Descripción:** El Área Bajo la Curva Característica Operativa del Receptor (AUC-ROC) es una métrica de evaluación para clasificadores binarios que producen una probabilidad de pertenencia a una clase como salida. La Curva ROC (Receiver Operating Characteristic) se crea graficando la **Tasa de Verdaderos Positivos (TPR o Recall)** en el eje Y contra la **Tasa de Falsos Positivos (FPR)** en el eje X a diferentes umbrales de clasificación. El AUC mide el área bajo esta curva.

* Un AUC de **0.5** indica que el clasificador no es mejor que una predicción aleatoria.
* Un AUC de **1.0** indica un clasificador perfecto.
* Generalmente, cuanto mayor sea el AUC, mejor será el rendimiento del modelo para distinguir entre las dos clases.

**Fórmulas:**
$$\text{TPR (Recall)} = \frac{TP}{TP + FN}$$
$$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Especificidad}$$

**Código:**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Probabilidades predichas por el modelo para la clase positiva
y_prob = np.array([0.1, 0.8, 0.3, 0.9, 0.2, 0.6, 0.4, 0.7, 0.5, 0.95])
# Etiquetas reales (0 para negativo, 1 para positivo)
y_true = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 1])

# Calcular el AUC-ROC
auc = roc_auc_score(y_true, y_prob)
print(f"AUC-ROC: {auc:.2f}")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal de referencia (clasificador aleatorio)
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
```
### 7.Puntuación Logarítmica (Log Loss o Cross-Entropy Loss)

**Descripción:** La puntuación logarítmica, también conocida como Log Loss o Cross-Entropy Loss, es una métrica de evaluación fundamental para clasificadores probabilísticos. A diferencia de métricas como la exactitud que evalúan las predicciones discretas finales, el Log Loss tiene en cuenta la **incertidumbre** de las predicciones del modelo. Penaliza las predicciones incorrectas con mayor severidad cuanto más confiado esté el modelo en su predicción errónea. Un valor de Log Loss más bajo indica un mejor ajuste del modelo a los datos.

**Fórmula (para clasificación binaria):**
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$
donde:
* $N$ es el número total de muestras.
* $y_i$ es la etiqueta real de la $i$-ésima muestra (0 o 1).
* $p_i$ es la probabilidad predicha por el modelo de que la $i$-ésima muestra pertenezca a la clase 1.

**Para clasificación multiclase:**
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \log(p_{ij})$$
donde:
* $M$ es el número de clases.
* $y_{ij}$ es 1 si la $i$-ésima muestra pertenece a la clase $j$, y 0 en caso contrario (codificación one-hot).
* $p_{ij}$ es la probabilidad predicha por el modelo de que la $i$-ésima muestra pertenezca a la clase $j$.

**Código:**
```python
from sklearn.metrics import log_loss
import numpy as np

# Ejemplo de clasificación binaria
y_true_binary = np.array([0, 1, 0, 1, 0, 0, 1, 1])
y_prob_binary = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3, 0.6, 0.95])

logloss_binary = log_loss(y_true_binary, y_prob_binary)
print(f"Log Loss (Binario): {logloss_binary:.2f}")

# Ejemplo de clasificación multiclase
y_true_multiclass = np.array([0, 1, 2, 0, 1, 2])
y_prob_multiclass = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.3, 0.5],
    [0.6, 0.3, 0.1],
    [0.05, 0.9, 0.05],
    [0.3, 0.4, 0.3]
])

logloss_multiclass = log_loss(y_true_multiclass, y_prob_multiclass)
print(f"Log Loss (Multiclase): {logloss_multiclass:.2f}")
```
### 8.Kappa de Cohen

**Descripción:** El Kappa de Cohen ($\kappa$) es una métrica estadística que mide la concordancia entre las predicciones de un clasificador y las etiquetas reales, corrigiendo la concordancia esperada por azar. En otras palabras, evalúa cuánto mejor es el rendimiento del modelo en comparación con lo que se esperaría si las predicciones se hicieran completamente al azar. El Kappa de Cohen es útil cuando se trabaja con datos desbalanceados o cuando existe un componente subjetivo en el etiquetado.

**Rango de Valores:**
* $\kappa = 1$: Concordancia perfecta entre el clasificador y las etiquetas.
* $\kappa = 0$: Concordancia igual a la que se esperaría por azar.
* $\kappa < 0$: Concordancia menor de la esperada por azar (lo cual es raro y generalmente indica un problema con el modelo o los datos).

**Fórmula:**
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
donde:
* $p_o$ es la proporción de concordancia observada (que es simplemente la exactitud del clasificador).
* $p_e$ es la proporción de concordancia esperada por azar. Se calcula considerando las probabilidades marginales de cada clase en las etiquetas reales y las predicciones.

Para un problema de clasificación binaria, $p_e$ se puede calcular como:
$$p_e = P(\text{clase 1 real}) \times P(\text{clase 1 predicha}) + P(\text{clase 0 real}) \times P(\text{clase 0 predicha})$$

**Código:**
```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Etiquetas reales y predicciones del modelo
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])

# Calcular el Kappa de Cohen
kappa = cohen_kappa_score(y_true, y_pred)
print(f"Kappa de Cohen: {kappa:.2f}")

# Ejemplo con mayor desacuerdo
y_pred_malo = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kappa_malo = cohen_kappa_score(y_true, y_pred_malo)
print(f"Kappa de Cohen (mayor desacuerdo): {kappa_malo:.2f}")

# Ejemplo con concordancia perfecta
y_pred_perfecto = y_true
kappa_perfecto = cohen_kappa_score(y_true, y_pred_perfecto)
print(f"Kappa de Cohen (concordancia perfecta): {kappa_perfecto:.2f}")

# Ejemplo con concordancia al azar (aproximada)
y_pred_azar = np.random.randint(0, 2, size=len(y_true))
kappa_azar = cohen_kappa_score(y_true, y_pred_azar)
print(f"Kappa de Cohen (azar aproximado): {kappa_azar:.2f}")
```
## 7. Normalización vs Estandarización

En el preprocesamiento de datos para Machine Learning, tanto la normalización como la estandarización son técnicas comunes para escalar los valores de las características. El objetivo es asegurar que diferentes características con diferentes rangos no afecten desproporcionadamente el rendimiento del modelo.

**Normalización:**

* **Objetivo:** Escalar los valores de las características a un rango específico, generalmente entre 0 y 1.
* **Método Común:** Dividir cada valor por el valor máximo de la característica (o el rango si se conoce). Para imágenes con valores de píxeles entre 0 y 255, la normalización a \[0, 1] se logra dividiendo cada valor de píxel por 255.
* **Resultado:** Los datos normalizados estarán confinados dentro de un intervalo fijo.
* **Sensibilidad a Outliers:** Es sensible a los valores atípicos (outliers). Si hay un valor máximo muy grande, la mayoría de los otros valores se comprimirán en un rango pequeño.

**Estandarización (o Z-score Normalization):**

* **Objetivo:** Escalar los valores de las características para que tengan una media de aproximadamente 0 y una desviación estándar de aproximadamente 1.
* **Método Común:** Para cada valor $x$ de una característica, se calcula $(x - \mu) / \sigma$, donde $\mu$ es la media de la característica y $\sigma$ es su desviación estándar.
* **Resultado:** Los datos estandarizados no tienen un rango fijo. Pueden tener valores positivos y negativos, y no están necesariamente confinados a un intervalo específico.
* **Menos Sensibilidad a Outliers:** Es menos sensible a los outliers en comparación con la normalización basada en el rango, ya que la media y la desviación estándar son menos drásticamente afectadas por valores extremos. Sin embargo, los outliers aún pueden influir en la media y la desviación estándar.

**¿Cuándo usar cuál?**

* **Normalización (img/255): Siempre para CNNs.**
    * Las Redes Neuronales Convolucionales (CNNs) a menudo funcionan mejor con datos de entrada normalizados al rango \[0, 1] o \[-1, 1]. La división por 255 es una forma común y efectiva de normalizar los valores de píxeles de imágenes al rango \[0, 1]. Esto ayuda a que el proceso de aprendizaje sea más estable y rápido, ya que los valores de entrada se mantienen dentro de un rango manejable para las funciones de activación y los cálculos de gradiente.

* **Estandarización: Para SVM/Random Forest con features HOG.**
    * Algoritmos como las Máquinas de Vectores de Soporte (SVM) y los Bosques Aleatorios (Random Forest) pueden beneficiarse de la estandarización cuando se utilizan características como Histogram of Oriented Gradients (HOG). HOG produce vectores de características con valores en un rango que puede variar significativamente entre diferentes imágenes y celdas. La estandarización asegura que cada componente del vector de características tenga una escala similar, lo que puede mejorar el rendimiento de estos modelos.
    * SVM es sensible a la escala de las características, ya que busca maximizar el margen basado en las distancias. La estandarización evita que las características con rangos más grandes dominen el cálculo de la distancia.
    * Si bien los Bosques Aleatorios son menos sensibles a la escala de las características que SVM, la estandarización de las características HOG puede, en algunos casos, conducir a una convergencia más rápida o un rendimiento ligeramente mejor.

**En resumen:** La elección entre normalización y estandarización depende del algoritmo de Machine Learning que se esté utilizando y de la naturaleza de los datos. La normalización al rango \[0, 1] es una práctica estándar para la entrada de imágenes en CNNs, mientras que la estandarización puede ser preferible para otros algoritmos como SVM y Random Forest, especialmente cuando se utilizan descriptores de características como HOG.


# Desglose de términos
## 1.`Conv2D`
Es una capa convolucional 2D. Se usa para procesar imagenes (que son datos bidimensionales con 1 o más canales, como RGB). Esta capa:
* Extrae características espaciales de la imagen (como bordes, texturas, formas).
* Usa unos "filtros" o "kernels" que van desplazandose (convolucionando) sobre la imagen original.

**Ejemplo**: `Conv2D(32, (3,3), activation='relu')`
- `32` (número de filtros)
* Significa que esta capa aplicará 32 filtros distintos
* Cada filtro va a detectar un tipo de patrón distinto (un filtro puede detectar bordes verticales, otro horizontales, texturas etc)
* **Resultado**: Se generan 32 mapas de activación, uno por cada filtro. Si tu imagen era `(100, 100, 3)`, la salida de esta capa será algo como
`(98, 98, 32)` que ahora es una representación abstracta de características aprendidas. (esto depende del tamaño del filtro y si hay padding o no).
- `(3,3)` (tamaño del filtro o kernel)
* Cada filtro es una matriz de tamaño 3x3 que "lee" una pequeña parte de la imagen.
* El filtro se va desplazando (como una ventana) por toda la imagen.
* Se calcula un valor de salida por cada posición, generando un mapa. 
* Esto se llama **operación de convolución**.
`activation='relu'`
* Después de aplicar la convolución, se pasa el resultado por una función de activación.
* `ReLU` (Rectified Linear Unit) es la activación más común en CNNs.
* Su formular es simple: `f(x) = max(0, x)`
    * Todos los valores negativos se convierten en 0
    * Se conservan los positivos
Esto introduce **no linealidad**, lo que permite a la red aprender patrones complejos.

## 2.`MaxPooling2D(2,2)`
- Max Pooling es una operación que:
    * **Reduce el tamaño espacial** (alto y ancho) de los mapas de activación.
    * Se aplica **independientemente por cada canal**.
    * Su función principal es:
        * **Reducir la cantidad de datos**(y por ende, el cómputo).
        * **Conservar las características más importantes**(como bordes, texturas clave).
        * Hacer que la red sea más **robusta a pequeñas variaciones** en la imagen (como movimientos o ruidos).
    * Reduce el overfitting: Al reducir la cantidad de parámetros que la red necesira procesar.
    * Hace que la red generalice mejor: Porque conserva solo los aspectos más prominentes (los valores máximos).
    * Reduce el tiempo de entrenamiento y mejora la invariancia espacial, es decir si un objeto se mueve un poco, igual puede ser detectado.

**Explicación no técnica del ejemplo completo en punto 1**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)), # Entrada: imagen 100x100 RGB
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(), # Aplana los features para la capa densa
    Dense(128, activation='relu'),
    Dense(2, activation='softmax') # 2 clases: probabilidades
])
```
Imagina que estás entrenando a un robot a reconocer imágenes de manzanas y plátanos. Este es el recorrido que hace paso a paso:

🔹 Paso 1: Mira la imagen por partes pequeñas
🧩 `Conv2D`
El robot examina pequeños bloques de 3x3 píxeles, buscando formas básicas como bordes y líneas.

🔹 Paso 2: Se queda solo con lo importante
🔍 `MaxPooling2D`
En cada zona, se queda con lo más representativo (el valor más fuerte), resumiendo la imagen sin perder lo esencial.

🔹 Paso 3: Vuelve a analizar con más experiencia
🎯 `Conv2D` con más filtros
Ya con una idea general, observa con mayor profundidad y más filtros. Ahora reconoce formas completas como la curvatura de un plátano o el tallo de una manzana.

🔹 Paso 4: Junta todo lo que aprendió
📦 `Flatten`
Convierte toda esa información visual en una lista lineal para tomar decisiones.

🔹 Paso 5: Toma una decisión
🧠 `Dense`
Con todo lo aprendido, el robot pasa por una etapa de "reflexión" (una capa densa) y finalmente decide:
→ “Estoy 90% seguro de que es una manzana y 10% de que es un plátano”.

