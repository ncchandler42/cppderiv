#ifndef CPPDERIV_HPP
#define CPPDERIV_HPP

#include <vector>
#include <array>
#include <map>
#include <string>
#include <regex>
#include <functional>
#include <chrono>
#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "exprtk.hpp"

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
using ValArr = ei::Array<double, ORDER, DIM>;

template <const std::size_t ORDER, const std::size_t DIM> 
using NumpyArr = py::EigenDRef<ValArr<ORDER, DIM>>;

template <const std::size_t ORDER, const std::size_t DIM> 
using StrArr = std::array<std::array<std::string, DIM>, ORDER>;

template <const std::size_t ORDER, const std::size_t DIM> 
using ExprArr = std::array<std::array<exprtk::expression<double>, DIM>, ORDER>;
using Expr = exprtk::expression<double>;
using ExprTable = exprtk::symbol_table<double>;
using ExprParser = exprtk::parser<double>;

using Alg = std::function<void()>;

using Dict = std::map<std::string, double>;
using Vect2D = std::vector<std::vector<double>>;

////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
class NDeriv {
public:	
	NDeriv(const StrArr<ORDER,DIM>& pyderivs, const Dict& consts, double dt, double timeout_s);
		
	void euler(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c);
	void rk2(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c);
	void rk4(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c);
	void leapfrog(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c);

	const Vect2D& get_plot_data();
	
private:
	void init_alg(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c);
	void run_alg();
	
	const ValArr<ORDER, DIM>& derivs(const ValArr<ORDER, DIM>& vals_);

	void euler_alg();
	void rk2_alg();
	void rk4_alg();
	void leapfrog_alg();

	std::string fix_indices(const std::string& expr);

	double t;
	double dt;
	double timeout_s;

	ExprParser parser;
	ExprTable table;
	Dict table_refs;

	ExprArr<ORDER, DIM> deriv_exprs;
	Expr stop_expr;
	
	Alg curr_alg;
	ValArr<ORDER, DIM> vals_n;
	ValArr<ORDER, DIM> vals_nhalf; // only used in leapfrog
	ValArr<ORDER, DIM> k1, k2, k3, k4;
	
	ValArr<ORDER, DIM> expr_mem, return_mem;

	Vect2D plot_data;
};

/////////////////////////////////////////////////////////////////////////////////
template <const std::size_t ORDER, const std::size_t DIM> 
NDeriv<ORDER,DIM>::NDeriv(const StrArr<ORDER,DIM>& pyderivs, const Dict& consts, double dt, double timeout_s):
	t(0), dt(dt), timeout_s(timeout_s)
{
	// add vals, t and dt to exprtk's table	 
	table.add_vector("vals", expr_mem.data(), expr_mem.size());

	table.add_variable("t", t);
	table.add_variable("dt", dt);

	// register constants into exprtk's variable table
	table_refs = consts;
	std::for_each(std::begin(table_refs), std::end(table_refs), 
		[&](auto& kv_pair) {
			table.add_variable(kv_pair.first, kv_pair.second);
		}
	);
		
	// compile each of the derivative expressions from Python, and register variable table
	std::size_t i, j;
	for (i = 0; i < ORDER; i++){
		for (j = 0; j < DIM; j++) {
			deriv_exprs[i][j].register_symbol_table(table);
			parser.compile(fix_indices(pyderivs[i][j]), deriv_exprs[i][j]);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::init_alg(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c)
{
	// copy initial values
	vals_n = init_vals;

	// compile the stop expression
	stop_expr.register_symbol_table(table);
	parser.compile(fix_indices(stop_c), stop_expr);
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::run_alg()
{	
	Timer timer;
	ei::Array<double, 1, DIM> X;
	plot_data.clear();
	
	X = vals_n.row(0);
	plot_data.emplace_back(X.data(), X.data()+DIM);
	plot_data.back().insert(std::begin(plot_data.back()), t);

	while (true) {
		curr_alg();

		if (stop_expr.value())
			break;

		if (timer.elapsed() > timeout_s)
			throw std::runtime_error("Timeout exceeded");

		t += dt;
		X = vals_n.row(0);
		plot_data.emplace_back(X.data(), X.data()+DIM);
		plot_data.back().insert(std::begin(plot_data.back()), t);
	}
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
const ValArr<ORDER, DIM>& NDeriv<ORDER,DIM>::derivs(const ValArr<ORDER,DIM>& vals_)
{
	// TODO intrmd = vals.unaryExpr 

	// load vals_ into the expr table
	expr_mem = vals_;
	
	std::size_t i, j;
	for (i = 0; i < ORDER; i++){
		for (j = 0; j < DIM; j++) {
			return_mem(i, j) = deriv_exprs[i][j].value();
		}
	}

	return return_mem;
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
	k2 = dt * derivs(vals_n + k1/2);

	vals_n += k2;
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk4_alg()
{
	k1 = dt * derivs(vals_n);
	k2 = dt * derivs(vals_n + k1/2);
	k3 = dt * derivs(vals_n + k2/2);
	k4 = dt * derivs(vals_n + k3);
	
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
void NDeriv<ORDER,DIM>::euler(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c)
{
	init_alg(init_vals, stop_c);
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::euler_alg, this);
	run_alg();
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk2(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c)
{
	init_alg(init_vals, stop_c);
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::rk2_alg, this);
	run_alg();
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::rk4(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c)
{
	init_alg(init_vals, stop_c);
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::rk4_alg, this);
	run_alg();
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
void NDeriv<ORDER,DIM>::leapfrog(NumpyArr<ORDER, DIM> init_vals, const std::string& stop_c)
{
	init_alg(init_vals, stop_c);
	
	// Get first vals_nhalf
	k1 = dt * derivs(vals_n);
	vals_nhalf = vals_n + k1/2;
	
	curr_alg = std::bind(&NDeriv<ORDER,DIM>::leapfrog_alg, this);
	run_alg();
}

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM> 
const Vect2D& NDeriv<ORDER,DIM>::get_plot_data() { return plot_data; }

/////////////////////////////////////////////////////////////////////////////////

template <const std::size_t ORDER, const std::size_t DIM>
std::string NDeriv<ORDER,DIM>::fix_indices(const std::string& expr)
{
	std::string out = "";
	std::regex re(R"((\d+)(?:, ?|\]\[)(\d+))");

	auto re_begin = std::sregex_iterator(std::begin(expr), std::end(expr), re);
	auto re_end   = std::sregex_iterator();

	// No matches, good as is
	if (re_begin == re_end)
		return expr;

	std::size_t r, c, ind;	
	for (auto it = re_begin; it != re_end; it = std::next(it)) {
		std::copy(it->prefix().first, it->prefix().second, std::back_inserter(out));

		r = std::stoi(it->str(1));
		c = std::stoi(it->str(2));

		// Eigen arrays are stored column-major
		ind = c*(ORDER) + r;

		if (c >= DIM || r >= ORDER || ind >= DIM*(ORDER)){
			std::string dims = std::to_string(ORDER) + "x" + std::to_string(DIM);
			throw std::out_of_range("Invalid indices parsed in NDeriv::fix_indices, dim(vals_n) = "+dims);
		}
		
		out += std::to_string(ind);
	}
	
	std::size_t len = std::distance(re_begin, re_end);
	auto re_last = std::next(re_begin, len-1);
	std::copy(re_last->suffix().first, re_last->suffix().second, std::back_inserter(out));

	return out;
}

#endif // CPPDERIV_HPP