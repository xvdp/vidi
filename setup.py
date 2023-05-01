"""@xvdp"""
import os.path as osp
from setuptools import setup, find_packages


def _readme():
    with open('README.md', encoding='utf8') as fo_:
        return fo_.read()

def _set_version(version):
    with open('vidi/version.py', 'w', encoding='utf8') as _fi:
        _fi.write("version='"+version+"'")
        return version

def _required(filename='requirements.txt'):
    _pwd = osp.dirname(osp.realpath(__file__))
    filename = osp.join(_pwd, filename)
    with open(filename, 'r', encoding='utf8') as _fi:
        return _fi.read().split()

def setup_package():
    ''' setup '''
    metadata = dict(
        name='vidi',
        version=_set_version(version='0.2.2'),
        description='modules to access video',
        url='http://github.com/xvdp/vidi',
        author='xvdp',
        author_email='xvdpahlen@gmail.com',
        packages=find_packages(),
        long_description=_readme(),
        install_requires=_required(),
        zip_safe=False)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
