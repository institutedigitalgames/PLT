import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="pyplt",
    version="0.2.0",
    description="A toolbox for preference learning implemented in Python.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/institutedigitalgames/PLT",
    author="Institute of Digital Games, University of Malta",
    author_email="plt.digitalgames@um.edu.mt",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux"
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    # python_requires='3.6',
    install_requires=['ttkthemes==2.1.0',
                      'numpy==1.14.2',
                      'matplotlib==2.2.2',
                      'pandas==0.22.0',
                      'tensorflow==1.7.0',
                      'scikit_learn==0.19.1',
                      'scipy==1.0.1'],

)
