from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.1'

install_requires = [
    'sklearn',
    'pySOT'
]


setup(name='sklearn_surrogatesearchcv',
    version=version,
    description="Surrogate adaptive randomized search for hyper parameters fin sklearn.",
    long_description=README + '\n\n' + NEWS,
    classifiers=[
    ],
    keywords='',
    author='Shaoqing Tan',
    author_email='tansq7@gmail.com',
    url='',
    license='MIT',
    packages=find_packages('src'),
    package_dir = {'': 'src'},include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
)
