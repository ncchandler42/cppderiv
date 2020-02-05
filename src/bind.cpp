
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cppderiv.hpp"

namespace ei = Eigen;
namespace py = pybind11;

const std::size_t ORDER = 2;
const std::size_t DIM = 3;

PYBIND11_MODULE(cppderiv, m)
{
	py::class_<NDeriv<ORDER, DIM>>(m, "NDeriv")
		.def(py::init<const StrArr<ORDER,DIM>&, const Dict&, double, double>())

		.def("euler", &NDeriv<ORDER, DIM>::euler)
		.def("rk2", &NDeriv<ORDER, DIM>::rk2)
		.def("rk4", &NDeriv<ORDER, DIM>::rk4)
		.def("leapfrog", &NDeriv<ORDER, DIM>::leapfrog)

		.def("get_plot_data", &NDeriv<ORDER, DIM>::get_plot_data, py::return_value_policy::reference_internal)
	;
}