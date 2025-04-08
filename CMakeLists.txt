cmake_minimum_required(VERSION 3.10)
project(CosineSimilarity)

# Encontrar pybind11
find_package(pybind11 REQUIRED)
message(STATUS "pybind11 found: ${pybind11_INCLUDE_DIRS}")

# Encontrar Eigen
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen found: ${EIGEN3_INCLUDE_DIR}")

# Crear el módulo
add_library(cosine_module MODULE bindings.cpp)
target_link_libraries(cosine_module PRIVATE 
    pybind11::module 
    Eigen3::Eigen
)
set_target_properties(cosine_module PROPERTIES 
    PREFIX "" 
    SUFFIX ".so"
)

# Incluir directorios
include_directories(${pybind11_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})