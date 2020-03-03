from setuptools import setup


setup(
    name='mlgw',
    version='1.0',
    author='S. Schmidt',
    author_email='stefanoschmidt1995@gmail.com',
    packages=['mlgw'],
    package_dir = {'mlgw':'./mlgw'},
    url="https://github.com/stefanoschmidt1995/MLGW/",
    license='LICENSE.txt',
    description='ML model for generating gravitational waveforms for a BBH coalescence',
    long_description=open('README.txt').read(),
    include_package_data = True,
    package_data={'mlgw': ['TD_model/*']},
    install_requires=[
        "numpy >= 1.16.4",
		"scipy >= 1.3.1",
		"lalsuite >= 6.62"
    ],
)

