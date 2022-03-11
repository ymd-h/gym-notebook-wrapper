FROM python:3.7

RUN apt update \
    && apt install -y --no-install-recommends \
    ffmpeg \
    python3-opengl \
    xvfb \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip setuptools \
    && pip3 install -U \
    brax \
    coverage \
    gym["classic_control"] \
    ipython \
    matplotlib \
    pyglet \
    pyvirtualdisplay \
    twine \
    unittest-xml-reporting \
    wheel

CMD ["/bin/bash"]
