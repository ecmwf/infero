

from setuptools import setup, find_packages


setup(
    name="pyinfero",
    version='0.1.0',
    author='ECMWF',
    author_email='software.support@ecmwf.int',
    license='Apache 2.0',
    description="ECMWF infero interface",
    packages=find_packages(exclude=["test_*", "*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={"": ["*.h"]},
    install_requires=[
    ],
    tests_require=[
        "pytest",
    ],
)

