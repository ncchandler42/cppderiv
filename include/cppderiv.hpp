#ifndef CPPDERIV_HPP
#define CPPDERIV_HPP

#include <vector>
#include <functional>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <stdexcept>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace ei = Eigen;
namespace py = pybind11;

////////////////////////////////////////////////////////////////////////////

class Timer {
public:
	Timer() { reset(); }

	void reset() { last = std::chrono::high_resolution_clock::now(); }
	double elapsed() 
	{	
		const auto now = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count() / 1000.0;
	}
	
private:
	std::chrono::high_resolution_clock::time_point last;
};

////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
using ValArr = ei::Array<double, ORDER+1, DIM>;

template <const std::size_t ORDER, const std::size_t DIM> 
using ValArr_r = ei::Ref<ValArr<ORDER, DIM>>;

template <const std::size_t ORDER, const std::size_t DIM> 
using NumpyArr = py::EigenDRef<ValArr<ORDER, DIM>>;

using PyFunc = py::object;
using Alg = std::function<void()>;
using Vector2D = std::vector<std::vector<double>>;

////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
class NDeriv {

public:	
	NDeriv(PyFunc& pyderivs, PyFunc& stop_c, double timeout_s): 
		pyderivs(pyderivs), stop_c(stop_c), timeout_s(timeout_s) {}
		
	void euler(NumpyArr<ORDER, DIM> init, double dt_);
	void rk2(NumpyArr<ORDER, DIM> init, double dt_);
	void rk4(NumpyArr<ORDER, DIM> init, double dt_);
	void leapfrog(NumpyArr<ORDER, DIM> init, double dt_);

	const Vector2D& get_plot_data();
	
private:
	void run(NumpyArr<ORDER, DIM> init, double dt_);
	
	const ValArr<ORDER, DIM>& derivs(ValArr_r<ORDER, DIM> vals);

	void euler_alg();
	void rk2_alg();
	void rk4_alg();
	void leapfrog_alg();

	PyFunc pyderivs;
	py::object pyderivs_res;
	
	PyFunc stop_c;
	double timeout_s;
	
	Alg curr_alg;
	ValArr<ORDER, DIM> vals_n;
	ValArr<ORDER, DIM> vals_nhalf; // only used in leapfrog
	ValArr<ORDER, DIM> k1, k2, k3, k4;
	ValArr<ORDER, DIM> intrmd; // fixes bug in RK algs
	double dt;

	Vector2D plot_data;
};

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::run(NumpyArr<ORDER, DIM> init, double dt_)
{
	if (vals_n.size() != derivs(vals_n).size() && vals_n.rows() != derivs(vals_n).rows())
		throw std::invalid_argument("Initial conditions must be the same dimensions as given derivatives");

	Timer t;
	py::object stop;
	ei::Array<double, 1, DIM> X;
	plot_data.clear();
	while (true) {
		curr_alg();

		stop = stop_c(vals_n);
		if (stop.cast<bool>())
			break;

		if (t.elapsed() > timeout_s)
			throw std::runtime_error("Timeout exceeded");

		X = vals_n.row(0);
		plot_data.emplace_back(X.data(), X.data()+DIM);
	}
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
const ValArr<ORDER, DIM>& NDeriv<ORDER,DIM>::derivs(ValArr_r<ORDER,DIM> vals)
{
	pyderivs_res = pyderivs(vals);
	intrmd = pyderivs_res.cast<ValArr<ORDER,DIM>>();
	return intrmd;
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::euler_alg()
{
	vals_n += dt * derivs(vals_n);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk2_alg()
{
	k1 = dt * derivs(vals_n);
	intrmd = vals_n + k1/2;
	
	k2 = dt * derivs(intrmd);

	vals_n += k2;
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk4_alg()
{
	k1 = dt * derivs(vals_n);
	intrmd = vals_n + k1/2;
	
	k2 = dt * derivs(intrmd);
	intrmd = vals_n + k2/2;

	k3 = dt * derivs(intrmd);
	intrmd = vals_n + k3;

	k4 = dt * derivs(intrmd);
	vals_n += (k1 + 2*k2 + 2*k3 + k4)/6;
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::leapfrog_alg()
{
	k2 = dt * derivs(vals_nhalf);
	vals_n += k2;
	
	vals_nhalf += dt * derivs(vals_n);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::euler(NumpyArr<ORDER, DIM> init, double dt_)
{
	vals_n = init;
	dt = dt_;
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::euler_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk2(NumpyArr<ORDER, DIM> init, double dt_)
{
	vals_n = init;
	dt = dt_;
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::rk2_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk4(NumpyArr<ORDER, DIM> init, double dt_)
{
	vals_n = init;
	dt = dt_;
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::rk4_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::leapfrog(NumpyArr<ORDER, DIM> init, double dt_)
{
	vals_n = init;
	dt = dt_;
	
	// Get first vals_nhalf
	k1 = dt * derivs(vals_n);
	vals_nhalf = vals_n + k1/2;
	
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::leapfrog_alg, this);
	run(init, dt_);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
const Vector2D& NDeriv<ORDER,DIM>::get_plot_data() { return plot_data; }

#endif // CPPDERIV_HPP