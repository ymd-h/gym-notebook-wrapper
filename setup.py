from setuptools import setup, find_packages

setup(name="gym-notebook-wrapper",
      author="Yamada Hiroyuki",
      version="0.0.1",
      install_requires=["gym","matplotlib","pyvirtualdisplay","ipython"],
      packages=["gnwrapper"],
      url="https://gitlab.com/ymd_h/gym-notebook-wrapper",
      classifiers=["Development Status :: 3 - Alpha",
                   "Framework :: Jupyter",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: POSIX :: Linux",
                   "Programming Language :: Python :: 3 :: Only",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Topic :: Software Development :: Libraries"])
