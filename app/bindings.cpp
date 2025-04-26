#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

class CosineSimilarity {
private:
    py::object sentence_transformers;
    py::object model;
    py::object util;

public:
    CosineSimilarity() {
        try {
            // Configurar el directorio de caché
            py::module_ os = py::module::import("os");
            os.attr("environ")["HF_HOME"] = "/opt/app-root/src/cache";
            // Configurar modo offline
            os.attr("environ")["TRANSFORMERS_OFFLINE"] = "1";
            os.attr("environ")["HF_HUB_OFFLINE"] = "1";
            // char* path = "/home/hadoop/Documentos/cpp_programs/pybind/py-cosine-similarity/all-MiniLM-L12-v2";
            char* path = "/opt/app-root/src/models/all-MiniLM-L12-v2";
            // Importar sentence_transformers y cargar el modelo desde un directorio local
            sentence_transformers = py::module::import("sentence_transformers");
            model = sentence_transformers.attr("SentenceTransformer")(path);
            util = sentence_transformers.attr("util");
            std::cout << "Modelo SentenceTransformer cargado exitosamente en C++ desde directorio local" << std::endl;
        } catch (const py::error_already_set& e) {
            std::cerr << "Error inicializando los módulos Python: " << e.what() << std::endl;
            throw std::runtime_error("Error inicializando los módulos Python: " + std::string(e.what()));
        }
    }

    double get_cosine_similarity(const std::string& text1, const std::string& text2) {
        try {
            // Crear lista de textos
            py::list texts;
            texts.append(text1);
            texts.append(text2);

            // Obtener embeddings como tensor
            py::object embeddings_obj = model.attr("encode")(texts, py::arg("convert_to_tensor") = true);

            // Acceder a los embeddings como objetos Python
            py::object emb1 = embeddings_obj[py::int_(0)];  // Primer embedding
            py::object emb2 = embeddings_obj[py::int_(1)];  // Segundo embedding

            // Calcular similitud de cosenos usando sentence_transformers.util.cos_sim
            py::object similarity_obj = util.attr("cos_sim")(emb1, emb2);
            double similarity = similarity_obj.attr("item")().cast<double>();

            return similarity;
        } catch (const py::cast_error& e) {
            throw std::runtime_error("Error convirtiendo resultado a double: " + std::string(e.what()));
        } catch (const py::error_already_set& e) {
            throw std::runtime_error("Error generando embeddings o calculando similitud: " + std::string(e.what()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Error en get_cosine_similarity: " + std::string(e.what()));
        }
    }
};

PYBIND11_MODULE(cosine_module, m) {
    m.doc() = "Módulo para calcular similitud de cosenos con SentenceTransformer en C++";
    py::class_<CosineSimilarity>(m, "CosineSimilarity")
        .def(py::init<>())
        .def("get_cosine_similarity", &CosineSimilarity::get_cosine_similarity, 
             "Calcula la similitud de cosenos entre dos cadenas de texto",
             py::arg("text1"), py::arg("text2"));
}
