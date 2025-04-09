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
    ```

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