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


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys

PACKAGE_NAME = 'covertree'

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

covertreec_module = Extension('covertreec',
        sources = ['src/cover_tree/covertreecmodule.cxx',  'src/cover_tree/cover_tree.cpp'],
        include_dirs=['lib/'],
        extra_compile_args=['-march=corei7-avx', '-pthread', '-std=c++14'],
        extra_link_args=['-march=corei7-avx', '-pthread', '-std=c++14'] #'-DPRINTVER']
)
setup ( name = 'covertree',
    version = '1.0',
    description = 'Cover Tree -- fast search',
    cmdclass={'build_ext':build_ext},
    install_requires=['numpy>=1.13.1', 'scipy>=0.17', 'scikit-learn>=0.18.1'],
    ext_modules = [ covertreec_module, ],
    packages = ['covertree']
)


