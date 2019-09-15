'''setup file for Vidi'''
import sys
import os
from setuptools import setup, find_packages


if sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version >= 3.4 required.")

def _readme():
    with open('README.md') as fo_:
        return fo_.read()

def set_version(version):
    with open('vidi/version.py', 'w') as _fi:
        _fi.write("version='"+version+"'")
    return version

def setup_package():
    ''' setup '''


    metadata = dict(
        name='vidi',
        version=set_version(version='0.0.5'),
        description='modules to access video',
        url='http://github.com/xvdp/vidi',
        author='xvdp',
        author_email='xvdp@gmail.com',
        license='tbd',
        packages=find_packages(),
        long_description=_readme(),
        zip_safe=False)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
