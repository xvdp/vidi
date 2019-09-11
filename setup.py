'''setup file for Vidi'''
import sys
import os
from setuptools import setup, find_packages


if sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version >= 3.4 required.")

def _readme():
    with open('README.rst') as fo_:
        return fo_.read()

def _readversion():
    with open(os.path.join('vidi', 'version.py')) as fo_:
        return fo_.read().split(' = ')[1]

def setup_package():
    ''' setup '''

    metadata = dict(
        name='vidi',
        version=_readversion(),
        description='modules to access video',
        url='http://github.com/xvdp/vidi',
        author='xvdp',
        author_email='xvdp@gmail.com',
        license='tbd',
        dependency_links=['http://pytorch.org'],
        packages=find_packages(),
        long_description=_readme(),
        zip_safe=False)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
