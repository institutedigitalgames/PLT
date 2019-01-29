import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="pyplt",
    version="0.1.0",
    description="A toolbox for preference learning implemented in Python.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/institutedigitalgames/PLT",
    author="Institute of Digital Games, University of Malta",
    author_email="plt.digitalgames@um.edu.mt",
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux"
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=['ttkthemes',
                      'numpy',
                      'matplotlib',
                      'pandas',
                      'tensorflow',
                      'scikit_learn',
                      'scipy'],

)
