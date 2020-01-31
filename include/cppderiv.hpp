#ifndef CPPDERIV_HPP
#define CPPDERIV_HPP

#include <vector>
#include <functional>
#include <chrono>
#include <stdexcept>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace ei = Eigen;
namespace py = pybind11;

class Timer {
public:
	Timer() { reset(); }

	void reset() { last = std::chrono::high_resolution_clock::now(); }
	auto elapsed() 
	{	
		auto now = std::chrono::high_resolution_clock::now();
		return (now - last).count();
	}
	
private:
	std::chrono::high_resolution_clock::time_point last;
};

////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
class NDeriv {
	using ValArr = ei::Array<double, ORDER, DIM>;

	using PyFunc = std::function<py::object(py::EigenDRef<ValArr>)>;
	using Alg = std::function<void()>;

public:	
	NDeriv(std::vector<PyFunc>& derivs, PyFunc& stop_c, double timeout_s): 
		derivs(derivs), stop_c(stop_c), timeout_s(timeout_s) {}

	void euler(const ValArr& init, double dt_);
	void rk2(const ValArr& init, double dt_);
	void rk4(const ValArr& init, double dt_);
	void leapfrog(const ValArr& init, double dt_);

	const std::vector<std::array<double,DIM>>& get_plot_data();
	
private:
	void run(const ValArr& init, double dt_);

	void euler_alg();
	void rk2_alg();
	void rk4_alg();
	void leapfrog_alg();

	std::vector<PyFunc> derivs;
	PyFunc stop_c;
	double timeout_s;
	
	Alg alg;
	ValArr vals_n;
	ValArr k1, k2, k3, k4;
	ValArr vals_nhalf; // only used in leapfrog
	double dt;

	std::vector<std::array<double,DIM>>  plot_data;
};

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::run(const ei::Array<double, ORDER, DIM>& init, double dt_)
{
	if (init.size() != derivs.size())
		throw std::invalid_argument("Number of initial values must match number of equations");
	
	vals_n = init;
	dt = dt_;

	Timer t;
	while (true) {
		alg();

		py::object stop = stop_c(vals_n);
		if (stop.cast<bool>())
			break;

		if (t.elapsed() > timeout_s)
			throw std::runtime_error("Timeout exceeded");
	}
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::euler_alg()
{

}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk2_alg()
{

}
/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk4_alg()
{
	
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::leapfrog_alg()
{

}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::euler(const ei::Array<double, ORDER, DIM>& init, double dt_)
{
	alg = std::bind(&NDeriv<ORDER,DIM>::euler_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk2(const ei::Array<double, ORDER, DIM>& init, double dt_)
{
	alg = std::bind(&NDeriv<ORDER,DIM>::rk2_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk4(const ei::Array<double, ORDER, DIM>& init, double dt_)
{
	alg = std::bind(&NDeriv<ORDER,DIM>::rk4_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::leapfrog(const ei::Array<double, ORDER, DIM>& init, double dt_)
{
	alg = std::bind(&NDeriv<ORDER,DIM>::leapfrog_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
const std::vector<std::array<double,DIM>>& NDeriv<ORDER,DIM>::get_plot_data() { return plot_data; }

#endif // CPPDERIV_HPP