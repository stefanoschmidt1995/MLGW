from setuptools import setup
import os

def readme():
    with open('README.rst') as f:
        return f.read()

def get_package_data():
    models = os.listdir("mlgw/TD_models")
    data_list = []
    for model in models:
        folder = './mlgw/TD_models/'+model+'/'
        files = os.listdir(folder)
        for f in files:
            data_list.append(folder+f)
    return data_list

def set_manifest():
    with open('MANIFEST.in', "w") as f:
        f.write("include README.rst\n")
        data_files = get_package_data()
        for data_file in data_files:
            f.write("include "+data_file+"\n")
        f.close()

set_manifest()

setup(
    name='mlgw',
    version='1.2.3',
    author='Stefano Schmidt',
    author_email='stefanoschmidt1995@gmail.com',
    packages=['mlgw'],
    package_dir = {'mlgw':'./mlgw'},
    url="https://github.com/stefanoschmidt1995/MLGW/",
    license='CC by 4.0',
    description='Machine learning modelling of the gravitational waves generated by black-hole binaries',
    long_description=readme(),
    include_package_data = True,
    package_data={'mlgw': get_package_data()},
    install_requires=[
        "numpy >= 1.16.4",
		"scipy >= 1.3.1",
		#"lalsuite >= 6.62"
    ],
	long_description_content_type = 'text/x-rst'
)

