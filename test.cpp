#include <iostream>
#include <array>

template<const std::size_t LEN>
using moo = std::array<double, LEN>;


template <const std::size_t LEN>
moo<LEN> cow()
{
	return moo<LEN>{1, 2, 3};
}


int main()
{
	std::cout << cow<3>()[0] << "\n";
}