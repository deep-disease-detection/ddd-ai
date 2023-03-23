from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()

requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="ddd",
    version="1.0.0",
    description="Virus classification from TEM microscope images using deep learning CNN architectures",
    license="MIT",
    author="Walid Abdaoui, Sabrina Dacelo, Laura Desire, Youssef El Ouard",
    author_email="abdaoui.wa@gmail.com, laura.desire98@gmail.com, youselouard@gmail.com, sabrina.dacelo@gmail.com",
    url="https://github.com/deep-disease-detection/ddd-ai",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
