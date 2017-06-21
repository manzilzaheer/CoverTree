# Copyright (c) 2017 Manzil Zaheer All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
