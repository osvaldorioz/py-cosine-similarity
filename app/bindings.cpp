#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

class CosineSimilarity {
private:
    py::object sentence_transformers;
    py::object model;

    double compute_cosine_similarity(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
        if (vec1.size() != vec2.size()) {
            throw std::runtime_error("Los vectores deben tener el mismo tamaño");
        }
        double dot_product = vec1.dot(vec2);
        double norm1 = vec1.norm();
        double norm2 = vec2.norm();
        if (norm1 == 0 || norm2 == 0) {
            throw std::runtime_error("Una de las normas es cero, no se puede calcular la similitud");
        }
        return dot_product / (norm1 * norm2);
    }

public:
    CosineSimilarity() {
        try {
            
            sentence_transformers = py::module::import("sentence_transformers");
            model = sentence_transformers.attr("SentenceTransformer")("all-MiniLM-L6-v2");
            std::cout << "Modelo SentenceTransformer cargado exitosamente en C++" << std::endl;
        } catch (const py::error_already_set& e) {
            throw std::runtime_error("Error inicializando los módulos Python: " + std::string(e.what()));
        }
    }

    double get_cosine_similarity(const std::string& text1, const std::string& text2) {
        try {
            py::list texts;
            texts.append(text1);
            texts.append(text2);

            py::object embeddings_obj = model.attr("encode")(texts, py::arg("convert_to_numpy") = true);
            py::array_t<double> embeddings_np = embeddings_obj.cast<py::array_t<double>>();

            auto buf = embeddings_np.request();
            double* ptr = static_cast<double*>(buf.ptr);
            Eigen::VectorXd vec1(buf.shape[1]);
            Eigen::VectorXd vec2(buf.shape[1]);
            for (int i = 0; i < buf.shape[1]; ++i) {
                vec1[i] = ptr[i];
                vec2[i] = ptr[buf.shape[1] + i];
            }

            return compute_cosine_similarity(vec1, vec2);
        } catch (const py::cast_error& e) {
            throw std::runtime_error("Error convirtiendo embeddings a NumPy: " + std::string(e.what()));
        } catch (const py::error_already_set& e) {
            throw std::runtime_error("Error generando embeddings: " + std::string(e.what()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Error en get_cosine_similarity: " + std::string(e.what()));
        }
    }
};

PYBIND11_MODULE(cosine_module, m) {
    m.doc() = "Módulo para calcular similitud de cosenos con SentenceTransformer y Eigen en C++";
    py::class_<CosineSimilarity>(m, "CosineSimilarity")
        .def(py::init<>())
        .def("get_cosine_similarity", &CosineSimilarity::get_cosine_similarity, 
             "Calcula la similitud de cosenos entre dos cadenas de texto",
             py::arg("text1"), py::arg("text2"));
}