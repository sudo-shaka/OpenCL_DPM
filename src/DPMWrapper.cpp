#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_Cell(py::module &);
void init_Tissue(py::module &);

namespace dpmmodule {
    PYBIND11_MODULE(clDPM, m) {
        m.doc() = "OpenCL Deformable Partical Model";
        init_Cell(m);
        init_Tissue(m);
    }
}
