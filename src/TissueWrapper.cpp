#include "Tissue.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_Tissue3D(py::module &m){
    py::class_<DPM::Tissue3D> (m, "Tissue3D")
        .def(py::init<std::vector<DPM::Cell3D>, float>())
        .def_readwrite("Kre", &DPM::Tissue3D::Kre)
        .def_readwrite("Cells", &DPM::Tissue3D::Cells)
        .def_readonly("NCELLS", &DPM::Tissue3D::NCELLS)
        .def_readonly("L", &DPM::Tissue3D::L)
        .def_readonly("PBC", &DPM::Tissue3D::PBC)
        .def("CLEulerUpdate", &DPM::Tissue3D::CLEulerUpdate)
        .def("Disperse2D", &DPM::Tissue3D::Disperse2D)
        ;
}
void init_Tissue2D(py::module &m){
    py::class_<DPM::Tissue2D> (m, "Tissue2D")
        .def(py::init<std::vector<DPM::Cell2D>, float>())
        .def_readwrite("Cells", &DPM::Tissue2D::cells)
        .def_readonly("NCELLS", &DPM::Tissue2D::NCELLS)
        .def_readonly("L", &DPM::Tissue2D::L)
        .def_readonly("PBC", &DPM::Tissue2D::PBC)
        .def_readwrite("Kre",&DPM::Tissue2D::Kre)
        .def_readwrite("Kat",&DPM::Tissue2D::Kat)
        .def("CLEulerUpdate", &DPM::Tissue2D::CLEulerUpdate)
        .def("Disperse", &DPM::Tissue2D::Disperse)
        ;
}
