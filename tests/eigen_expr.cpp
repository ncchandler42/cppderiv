#include <iostream>
#include <string>

#include <Eigen/Dense>
#include "exprtk.hpp"

namespace ei = Eigen;

int main()
{
	exprtk::symbol_table<double> tbl;
	exprtk::expression<double> expr;
	exprtk::parser<double> psr;

	ei::Array<double, 2, 3> arr;

	arr << 1, 2,
	       3, 4,
	       5, 6;

	std::string expr_s = "arr[1] * (arr[0] + arr[5])";

	// Order matters. Add all symbols before registering table
	tbl.add_vector("arr", arr.data(), arr.size());
	expr.register_symbol_table(tbl);
	psr.compile(expr_s, expr);

	std::cout << expr.value() << "\n";

	arr(0,0) = 10;

	std::cout << expr.value() << "\n";
}