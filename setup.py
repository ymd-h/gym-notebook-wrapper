import os
from setuptools import setup, find_packages
import warnings

description = "Wrapper for running/rendering OpenAI Gym on Jupyter Notebook"
README = os.path.join(os.path.abspath(os.path.dirname(__file__)),'README.md')
if os.path.exists(README):
    with open(README,encoding='utf-8') as f:
        long_description = f.read()
    long_description_content_type='text/markdown'
else:
    warnings.warn("No README.md")
    long_description =  description
    long_description_content_type='text/plain'


setup(name="gym-notebook-wrapper",
      author="Yamada Hiroyuki",
      version="1.3.3",
      install_requires=["gym","matplotlib","pyvirtualdisplay","ipython","moviepy"],
      extras_require={"test": ["brax"]},
      packages=find_packages(),
      url="https://github.com/ymd-h/gym-notebook-wrapper",
      classifiers=["Development Status :: 4 - Beta",
                   "Framework :: Jupyter",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: POSIX :: Linux",
                   "Programming Language :: Python :: 3 :: Only",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Topic :: Software Development :: Libraries"],
      description=description,
      long_description=long_description,
      long_description_content_type=long_description_content_type)
