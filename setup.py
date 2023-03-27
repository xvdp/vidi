'''setup file for Vidi'''

from setuptools import setup, find_packages


def _readme():
    with open('README.md', encoding='utf8') as fo_:
        return fo_.read()

def _set_version(version):
    with open('vidi/version.py', 'w', encoding='utf8') as _fi:
        _fi.write("version='"+version+"'")
        return version

def setup_package():
    ''' setup '''
    metadata = dict(
        name='vidi',
        version=_set_version(version='0.15'),
        description='modules to access video',
        url='http://github.com/xvdp/vidi',
        author='xvdp',
        author_email='xvdpahlen@gmail.com',
        packages=find_packages(),
        install_requires=['numpy>=1.15'],
        long_description=_readme(),
        zip_safe=False)

    setup(**metadata)

if __name__ == '__main__':
    setup_package()
