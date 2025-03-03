PYTHON_VERSION := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('VERSION'))")
PYTHON_INCLUDE := $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PYBIND11_INCLUDE := $(shell python3 -m pybind11 --includes | cut -d' ' -f2)

CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC -lOpenCL -Wl,-v
INCLUDES := -I$(PYTHON_INCLUDE) -I$(PYBIND11_INCLUDE) -I include/
SRC := $(wildcard src/*.cpp)
OUT := clDPM$(shell python3-config --extension-suffix)

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $@

clean:
	rm -f $(OUT)

