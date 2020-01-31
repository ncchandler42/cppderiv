from setuptools import setup
from cmake_setuptools import CMakeExtension, CMakeBuildExt

setup(
	name="cppderiv",
	description="Compiled assortment of numerical methods to solve ODEs",
	version="0.1.0alpha",
	ext_modules=[CMakeExtension("all")],
	cmdclass={"build_ext": CMakeBuildExt}
)