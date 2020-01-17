from setuptools import setup


setup(
    name='mlgw',
    version='0.1.dev0',
    author='S. Schmidt',
    author_email='stefanoschmidt1995@gmail.com',
    packages=['mlgw'],
    package_dir = {'mlgw':'./mlgw'},
    url='none',#'http://pypi.python.org/pypi/??/',
    license='LICENSE.txt',
    description='ML model for generating gravitational waveforms for a BBH coalescence',
    long_description=open('README.txt').read(),
    include_package_data = True,
    package_data={'mlgw': ['TD_model/*', '*.txt']},
    install_requires=[
        "numpy >= 1.16.4",
		"scipy >= 1.3.1",
    ],
)

