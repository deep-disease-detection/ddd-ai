from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()

requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="ddd",
    version="0.0.6",
    description="Deep Disease Detection Model",
    license="MIT",
    author="Deep Disease Detection",
    author_email="youselouard@gmail.com",
    url="https://github.com/deep-disease-detection/ddd-ai",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
