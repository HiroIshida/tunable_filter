from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "numpy",
    "opencv-python",
    "dataclasses",  # for 3.6
    "PyYAML",
    "types-PyYAML"
]

setup(
    name='tunable_filter',
    version='0.0.2',
    description='Tunable image filter',
    author='Hirokazu Ishida',
    author_email='h-ishida@jsk.imi.i.u-tokyo.ac.jp',
    license='MIT',
    install_requires=install_requires,
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={'tunable_filter': ['py.typed']}
)
