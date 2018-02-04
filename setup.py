#!/usr/bin/env python
from setuptools import setup, find_packages
import os

module_name = 'qrnn'

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(ROOT_PATH, 'src', module_name, 'version.py')).read())

# If there is a "pip.txt" file
def requirements_from_pip():
    return filter(lambda l: not l.startswith('#'), open('pip.txt').readlines())

setup(name=module_name,
      url="https://github.com/lamfo-unb/QRNN",
      author="LAMFO",
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version=__version__, # or open(os.path.join('src', 'mylib', 'resources', 'VERSION')).read().strip()
      install_requires=requirements_from_pip(), # or ["pytest", "pytest-cov", "pytest-xdist", "common-core-python==0.19.3", "common-io-python>=0.23.0", ...]
      include_package_data=True,
      zip_safe=False)