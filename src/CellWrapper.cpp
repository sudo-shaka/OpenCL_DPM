#include "cell.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_Cell(py::module &m){
    py::class_<DPM::Cell3D> (m, "Cell3D")
        .def(py::init<std::array<float,3>, float, float>())
        .def_readwrite("Kv", &DPM::Cell3D::Kv)
        .def_readwrite("Ka", &DPM::Cell3D::Ka)
        .def_readwrite("Ks", &DPM::Cell3D::Ks)
        .def_readwrite("Verts", &DPM::Cell3D::Verts)
        .def("CLShapeEuler", &DPM::Cell3D::CLShapeEuler)
        .def("GetVolume", &DPM::Cell3D::GetVolume)
        .def("GetPositions", &DPM::Cell3D::GetPositions)
        .def("GetFaces",&DPM::Cell3D::GetFaces)
        .def("GetForces",&DPM::Cell3D::GetForces)
        .def("GetVolume",&DPM::Cell3D::GetVolume)
        ;
}
