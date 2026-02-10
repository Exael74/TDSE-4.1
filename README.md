# Implementación: Capítulo 2 - Trabajando con Datos de Texto

Este repositorio contiene la implementación práctica y explicada de los conceptos del **Capítulo 2** del libro *"Build a Large Language Model (From Scratch)"* de Sebastian Raschka.

El objetivo principal es preparar datos de texto para el entrenamiento de un LLM tipo GPT, cubriendo desde la tokenización hasta la generación de embeddings.

## 📂 Archivos del Proyecto

*   **`embeddings.ipynb`**: Notebook principal creado. Contiene:
    *   Código core para tokenización (BPE) y creación de datasets.
    *   Explicaciones teóricas sobre por qué dividimos el texto, usamos *sliding windows* y qué representan los embeddings.
    *   Resultados del experimento de *overlap*.
*   **`experiment.py`**: Script de Python utilizado para realizar experimentos cuantitativos sobre el impacto del *stride* en la cantidad de muestras de entrenamiento.
*   **`ch02.ipynb`**: Notebook original de referencia del libro.
*   **`the-verdict.txt`**: Corpus de texto utilizado (cuento corto de Edith Wharton).
*   **`the veredict.txt`**: (Archivo original renombrado a `the-verdict.txt` para consistencia).

## 🚀 Proceso de Implementación

### 1. Preparación del Entorno
Se descargaron los archivos necesarios y se renombró el archivo de texto a `the-verdict.txt` para asegurar la compatibilidad con el código estándar del libro.

### 2. Tokenización (Byte Pair Encoding)
Implementamos el tokenizador BPE utilizando la librería `tiktoken` (modelo `gpt2`). BPE es crucial porque permite manejar palabras fuera del vocabulario dividiéndolas en subpalabras, equilibrando el tamaño del vocabulario y la longitud de la secuencia.

### 3. Dataset y Sliding Window
Creamos la clase `GPTDatasetV1` y un `DataLoader` que utiliza una ventana deslizante (*sliding window*) para generar pares de `(input, target)` para el entrenamiento autoregresivo.

### 4. Experimento: Impacto del Stride
Utilizamos el script `experiment.py` para analizar cómo el parámetro `stride` (paso de la ventana) afecta la cantidad de datos generados.

**Resultados (para `max_length=4`):**
*   **Stride 4 (Sin solapamiento):** 1286 muestras.
*   **Stride 1 (Alto solapamiento):** 5141 muestras.

**Conclusión:** Un `stride` menor (mayor solapamiento) actúa como una forma de *data augmentation* natural, generando casi 4 veces más datos de entrenamiento del mismo corpus, lo cual es vital para evitar overfitting en datasets pequeños.

### 5. Embeddings
Finalmente, implementamos capas de embeddings (`torch.nn.Embedding`) para transformar los IDs de los tokens en vectores densos continuos, sumando embeddings posicionales para retener la información del orden de las palabras.

## 🛠️ Cómo Ejecutar

1.  **Instalar dependencias:**
    ```bash
    pip install torch tiktoken
    ```

2.  **Ejecutar el Notebook:**
    Abrir `embeddings.ipynb` en Jupyter Lab o VS Code y ejecutar las celdas secuencialmente.

3.  **Reproducir el Experimento:**
    EJecutar el script desde la terminal:
    ```bash
    python experiment.py
    ```

## 🧠 Conceptos Clave para Agentes
*   **Embeddings:** Son la base de la "comprensión" semántica. Permiten que un agente relacione instrucciones de usuario con conceptos aprendidos.
*   **Context Window:** La gestión eficiente de la ventana de contexto (vía sliding windows durante el entrenamiento) permite al agente mantener la coherencia en conversaciones largas.
