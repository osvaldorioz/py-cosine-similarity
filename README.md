
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
 
La instrucción:

```cpp
double dot_product = vec1.dot(vec2);
```

en el contexto de este programa, calcula el **producto punto** (o producto escalar) entre dos vectores representados como objetos `Eigen::VectorXd` (`vec1` y `vec2`) y asigna el resultado a la variable `dot_product` de tipo `double`. Vamos a desglosar qué hace exactamente esta línea y por qué es importante.

---

### Contexto
- **`vec1` y `vec2`**: Son instancias de `Eigen::VectorXd`, una clase de la biblioteca `Eigen` que representa vectores de longitud dinámica con elementos de tipo `double`. En este caso, cada vector tiene 384 elementos, correspondientes a los embeddings generados por el modelo `all-MiniLM-L6-v2` para las dos cadenas de texto.
- **`dot`**: Es un método proporcionado por `Eigen` para calcular el producto punto entre dos vectores.

### ¿Qué hace el producto punto?
El producto punto entre dos vectores \(\mathbf{v_1}\) y \(\mathbf{v_2}\) de longitud \(n\) se define matemáticamente como:

![imagen](https://github.com/user-attachments/assets/a24a3c0a-c4e0-4be8-9043-4de2975893af)


Es decir:
- Multiplica cada elemento de `vec1` por el elemento correspondiente de `vec2`.
- Suma todos esos productos para obtener un único valor escalar.

En este caso:
![imagen](https://github.com/user-attachments/assets/9d6937e7-8c76-43f3-9ee8-9dec927b7cdd)


### Ejemplo práctico
Supongamos que tenemos vectores más cortos para ilustrarlo (digamos, \(n = 3\)):
- `vec1 = [1.0, 2.0, 3.0]`
- `vec2 = [4.0, 5.0, 6.0]`

El cálculo sería:
![imagen](https://github.com/user-attachments/assets/b8fe6eb1-f8b1-43f6-b62d-4a4d1c72055d)


Entonces, `double dot_product = vec1.dot(vec2);` asignaría `32.0` a `dot_product`.

### Rol en el programa
Esta instrucción es parte del cálculo de la **similitud de cosenos**, que se define como:

![imagen](https://github.com/user-attachments/assets/304f3f56-1386-4627-9b9a-89063df485b3)


Donde:
![imagen](https://github.com/user-attachments/assets/84b0804b-4f72-4287-baf9-9cede8597d5a)


En el código:
```cpp
double dot_product = vec1.dot(vec2);
double norm1 = vec1.norm();
double norm2 = vec2.norm();
return dot_product / (norm1 * norm2);
```

- `dot_product` es el numerador de la fórmula.
- El resultado final es la similitud de cosenos, un valor entre -1 y 1 que indica cuán similares son los dos embeddings (y, por ende, las cadenas de texto).

### Propiedades del producto punto
- **Simetría**: `vec1.dot(vec2) == vec2.dot(vec1)`.
- **Relación con el ángulo**: El producto punto está relacionado con el ángulo \(\theta\) entre los vectores por:
  ![imagen](https://github.com/user-attachments/assets/37677152-0c97-4c40-8775-c546bc438173)

  Por eso, dividir por las normas da \(\cos(\theta)\), que es la similitud de cosenos.
- **Eficiencia**: `Eigen` implementa este cálculo de forma optimizada, aprovechando operaciones vectorizadas si el hardware lo permite.

### En el contexto de los embeddings
- `vec1` y `vec2` representan los embeddings de "Osvaldo Rivera Zamudio" y "OSVALDO Rios Zambrano" (384 dimensiones cada uno).
- El producto punto mide la "alineación" entre estos vectores en el espacio de embeddings. Un valor positivo alto (como en `0.7413`) indica que los textos son semánticamente similares.

La instrucción `double dot_product = vec1.dot(vec2);`:
- Calcula el producto escalar entre los embeddings de las dos cadenas.
- Es un paso crítico para determinar la similitud de cosenos.
- Usa la biblioteca `Eigen` para realizar la operación de manera eficiente en C++.

#### Resultado final
El programa ahora:
- Carga el modelo `all-MiniLM-L6-v2` en C++ al instanciar `CosineSimilarity`.
- Genera embeddings correctos para dos cadenas de texto ingresadas.
- Calcula la similitud de cosenos en C++ con `Eigen`
- Es robusto y eficiente, evitando dependencias pesadas como `libtorch`.

### Resumen
La implementación final es una solución híbrida que combina la facilidad de Python para la entrada/salida con la eficiencia de C++ para el procesamiento.
