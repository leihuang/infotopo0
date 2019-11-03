from setuptools import setup
from setuptools import find_packages

from version import __version__

setup(
        name='infotopo0',
        packages=find_packages(),
        url='',
        license='',
        author='Lei Huang',
        author_email='lh389@cornell.edu',
        description='kinetic modeling package',
        install_requires=['numpy',
                          'SloppyCell',
                          'pandas',
                          'scipy',
                          'networkx',
                          'dot2tex',
                          'pygraphviz',
                          'sympy',                          
                          ],  
        version=__version__,
        classifiers=['Topic :: Scientific/Engineering :: Bio-Informatics',
                     'Programming Language :: Python :: 2.7',
        ],
)
