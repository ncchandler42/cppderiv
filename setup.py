from setuptools import setup
from cmake_setuptools import CMakeExtension, CMakeBuildExt

setup(
	name="cppderiv",
	description="Compiled assortment of numerical methods to solve ODEs",
	version="0.0.0.dev0",
	ext_modules=[CMakeExtension("all")],
	cmdclass={"build_ext": CMakeBuildExt}
)