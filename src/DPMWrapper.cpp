#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_Cell2D(py::module &);
void init_Tissue2D(py::module &);
void init_Cell3D(py::module &);
void init_Tissue3D(py::module &);

namespace dpmmodule {
    PYBIND11_MODULE(clDPM, m) {
        m.doc() = "OpenCL Deformable Partical Model";
        init_Cell2D(m);
        init_Tissue2D(m);
        init_Cell3D(m);
        init_Tissue3D(m);
    }
}
