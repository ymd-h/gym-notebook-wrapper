from setuptools import setup, find_packages

setup(name="gym-notebook-wrapper",
      version="0.0.1",
      install_requires=["gym","matplotlib","pyvirtualdisplay","ipython"],
      packages=["gnwrapper"],
      classifiers=["Development Status :: 3 - Alpha",
                   "Framework :: Jupyter",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: POSIX :: Linux",
                   "Programming Language :: Python :: 3 :: Only",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Topic :: Software Development :: Libraries"])
