#include <iostream>
#include <string>
#include <regex>
#include <iterator>
#include <algorithm>
#include <stdexcept>

template <const std::size_t ORDER, const std::size_t DIM>
std::string fix_indices(const std::string& expr)
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
		ind = c*(ORDER+1) + r;

		if (c >= DIM || r >= ORDER+1 || ind >= DIM*(ORDER+1)){
			std::string dims = std::to_string(ORDER+1) + "x" + std::to_string(DIM);
			throw std::out_of_range("Invalid indices parsed in fix_indices, dim(vals_n) = "+dims);
		}
		
		out += std::to_string(ind);
	}
	
	std::size_t len = std::distance(re_begin, re_end);
	auto re_last = std::next(re_begin, len-1);
	std::copy(re_last->suffix().first, re_last->suffix().second, std::back_inserter(out));

	return out;
}

int main()
{
	std::string expr = "";

	while (expr != "exit") {
		std::cout << "expr1: ";
		std::getline(std::cin, expr);	
		std::cout << "expr2: " << fix_indices<5,5>(expr) << "\n\n";
	}
}