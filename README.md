# Gym-Notebook-Wrapper

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gym-notebook-wrapper)
![PyPI](https://img.shields.io/pypi/v/gym-notebook-wrapper)
![PyPI - Status](https://img.shields.io/pypi/status/gym-notebook-wrapper)
![PyPI - License](https://img.shields.io/pypi/l/gym-notebook-wrapper)



Gym-Notebook-Wrapper provides small wrappers for running and rendering
[OpenAI Gym](https://github.com/openai/gym) on [Jupyter
Notebook](https://jupyter.org/) or similar (e.g. [Google
Corab](https://colab.research.google.com/)).

# Requirement

- Linux
- [Xvfb](https://www.x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml)
  - On Ubuntu, you can install `sudo apt update && sudo apt install xvfb`.
- Open GL (for some environment)
  - On Ubuntu, you can install `sudo apt update && sudo apt install python-opengl`

# Installation

You can install from
[PyPI](https://pypi.org/project/gym-notebook-wrapper/) with `pip install gym-notebook-wrapper`


# Usage

Two classes are implemented in `gnrwapper` module in this
gym-notebook-wrapper package.

## Simple One Shot Animation

Wrap `gym.Env` class with `gnwrapper.Animation`. That's all! The
`render()` method shows the environment on its output. An example code
is following;

### Code

``` python
import gnwrapper
import gym

env = gnwrapper.Animation(gym.make('CartPole-v1'))

obs = env.reset()

for _ in range(1000):
    next_obs, reward, done, info = env.step(env.action_space.sample())
    env.render()

    obs = next_obs
    if done:
        obs = env.reset()
```

### Limitation

- Calling `render()` method delete the other output for the same cell.
- The output image is shown only once.


## Loop Animation

Wrap `gym.Env` class with `gnwrapper.LoopAnimation`. This wrapper
stores display image when `render()` methos is called and shows the
loop animation by `display(dpi=72,interval=50)` methos.

### Code

``` python
import gnwrapper
import gym

env = gnwrapper.LoopAnimation(gym.make('CartPole-v1'))

obs = env.reset()

for _ in range(100):
    next_obs, reward, done, info = env.step(env.action_space.sample())
    env.render()

    obs = next_obs
    if done:
        obs = env.reset()

env.display()
```


### Limitation

- Require a lot of memory to store and display large steps of display
  - Can raise memory error


# Notes

`gnwrapper.Animation` and `gnwrapper.LoopAnimation` inherit from
`gym.Wrapper`, so that it can access any fields or mothods of
`gym.Env` and `gym.Wrapper` (e.g. `action_space`).
