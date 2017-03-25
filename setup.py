from distutils.core import setup, Extension
import numpy as np

covertreec_module = Extension('covertreec',
		sources = ['src/cover_tree/covertreecmodule.cxx', 'src/cover_tree/cover_tree.cpp'],
		include_dirs=[np.get_include(), 'lib/'],
		extra_compile_args=['-march=corei7', '-pthread', '-std=c++14'],
		extra_link_args=['-march=corei7', '-pthread', '-std=c++14']
)

setup ( name = 'covertreec',
	version = '1.0',
	description = 'Cover Tree Data Structure.',
	ext_modules = [ covertreec_module ],
	packages = ['covertree']
)
