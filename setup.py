# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='flicsapp',
    version='0.1.0',
    description='Run FLICS analysis from the browser',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/PBLab/flics-app',  # Optional
    author='PBLab',  # Optional
    author_email='',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=['matplotlib > 3',
                      'numpy > 1.17',
                      'scipy > 1.2',
                      'tifffile',
                      'ipython > 7',
                      'pandas > 0.25',
                      'sympy',
                      'bokeh > 1',
                      'symfit',
                      'xarray > 0.12'],  # Optional
)
