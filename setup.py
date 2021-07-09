'''setup file for Vidi'''
import sys
from setuptools import setup, find_packages


def _readme():
    with open('README.md') as fo_:
        return fo_.read()

def set_version(version):
    with open('vidi/version.py', 'w') as _fi:
        _fi.write("version='"+version+"'")
    return version

def install_requires():
    with open('requirements.txt') as _fo:
        return _fo.read().split()

def setup_package():
    ''' setup '''
    metadata = dict(
        name='vidi',
        version=set_version(version='0.1'),
        description='modules to access video',
        url='http://github.com/xvdp/vidi',
        author='xvdp',
        author_email='xvdp@gmail.com',
        packages=find_packages(),
        install_requires=['kotools@git+https://github.com/xvdp/kotools@main', 'numpy>=1.15'],
        long_description=_readme(),
        zip_safe=False)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
