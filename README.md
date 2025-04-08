
### Descripción de la implementación del programa

#### Objetivo
El programa tiene como objetivo calcular la similitud de cosenos entre dos cadenas de texto ingresadas por el usuario, utilizando embeddings generados por un modelo de lenguaje (`all-MiniLM-L6-v2` de `sentence-transformers`) y realizando el cálculo en C++ para mayor eficiencia. La interfaz de usuario está en Python, mientras que la lógica pesada (generación de embeddings y cálculo de similitud) se trasladó a C++.

#### Estructura general
1. **Interfaz en Python (`test1.py`)**:
   - Solicita al usuario dos cadenas de texto.
   - Importa un módulo C++ (`cosine_module`) creado con `pybind11`.
   - Llama a una clase `CosineSimilarity` definida en C++ para obtener la similitud y muestra el resultado.

2. **Módulo en C++ (`bindings.cpp`)**:
   - Define la clase `CosineSimilarity` que:
     - Carga el modelo `sentence-transformers` en su constructor.
     - Genera embeddings para las cadenas de texto.
     - Calcula la similitud de cosenos usando `Eigen`.
   - Expone esta funcionalidad a Python mediante `pybind11`.

3. **Configuración de compilación (`CMakeLists.txt`)**:
   - Utiliza CMake para compilar el módulo C++ como una biblioteca compartida (`cosine_module.so`), vinculando las dependencias necesarias (`pybind11` y `Eigen`).

#### Evolución de la implementación
La implementación pasó por varias iteraciones debido a errores y ajustes:

#### Detalles técnicos
- **Generación de embeddings**:
  - En C++, `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` genera embeddings de 384 dimensiones para cada texto.
  - Se usa `encode` con `convert_to_numpy=True` para obtener un array de NumPy (`2 x 384` para dos textos).
  - El array se convierte a `py::array_t<double>` y se accede a su buffer (`buf.ptr`) para extraer los valores.

- **Cálculo de similitud**:
  - Dos vectores `Eigen::VectorXd` (cada uno de 384 elementos) se construyen copiando los datos de NumPy.
  - La similitud de cosenos se calcula como:
    \![imagen](https://github.com/user-attachments/assets/11701dcc-2030-4411-9a58-c492dce8483a)

    usando `vec1.dot(vec2)` para el producto punto y `vec1.norm()` para las normas L2.

- **Interfaz Python-C++**:
  - `pybind11` permite embeber Python en C++ para usar `sentence-transformers` y exponer la clase `CosineSimilarity` a Python.

- **Compilación**:
  - `CMakeLists.txt` vincula `pybind11` (para la interfaz Python) y `Eigen3` (para cálculos vectoriales).

#### Resultado final
El programa ahora:
- Carga el modelo `all-MiniLM-L6-v2` en C++ al instanciar `CosineSimilarity`.
- Genera embeddings correctos para dos cadenas de texto ingresadas.
- Calcula la similitud de cosenos en C++ con `Eigen`
- Es robusto y eficiente, evitando dependencias pesadas como `libtorch`.

### Resumen
La implementación final es una solución híbrida que combina la facilidad de Python para la entrada/salida con la eficiencia de C++ para el procesamiento.
