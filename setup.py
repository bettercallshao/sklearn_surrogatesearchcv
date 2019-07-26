from setuptools import find_packages, setup

version = '0.1'

install_requires = [
    'sklearn',
    'pySOT'
]

with open('README.md') as f:
    long_description = f.read()

setup(
    name='sklearn_surrogatesearchcv',
    version=version,
    description="Surrogate adaptive randomized search for hyper parameters"
                "in sklearn.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[],
    keywords='',
    author='Shaoqing Tan',
    author_email='tansq7@gmail.com',
    url='https://github.com/timlyrics/sklearn_surrogatesearchcv',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
)
