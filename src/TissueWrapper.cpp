#include "Tissue.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_Tissue(py::module &m){
    py::class_<DPM::Tissue3D> (m, "Tissue3D")
        .def(py::init<std::vector<DPM::Cell3D>, float>())
        .def_readwrite("Kre", &DPM::Tissue3D::Kre)
        .def_readwrite("Cells", &DPM::Tissue3D::Cells)
        .def_readonly("NCELLS", &DPM::Tissue3D::NCELLS)
        .def("CLEulerUpdate", &DPM::Tissue3D::CLEulerUpdate)
        ;
}