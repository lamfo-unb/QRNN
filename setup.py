#!/usr/bin/env python
from setuptools import setup, find_packages
from os.path import join

MODULE_NAME = 'qrnn'

# If there is a "pip.txt" file
def requirements_from_pip():
    return filter(lambda l: not l.startswith('#'), open('pip.txt').readlines())

setup(name=MODULE_NAME,
      url="https://github.com/lamfo-unb/QRNN",
      author="LAMFO",
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version=(open(join('src', MODULE_NAME, 'resources', 'VERSION'))
               .read().strip()),
      install_requires=requirements_from_pip(),
      include_package_data=True,
      zip_safe=False)