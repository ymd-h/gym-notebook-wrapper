# Gym-Notebook-Wrapper

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gym-notebook-wrapper)
![PyPI](https://img.shields.io/pypi/v/gym-notebook-wrapper)
![PyPI - Status](https://img.shields.io/pypi/status/gym-notebook-wrapper)
![PyPI - License](https://img.shields.io/pypi/l/gym-notebook-wrapper)



Gym-Notebook-Wrapper provides small wrappers for running and rendering
[OpenAI Gym](https://github.com/openai/gym) on [Jupyter
Notebook](https://jupyter.org/) or similar (e.g. [Google
Colab](https://colab.research.google.com/)).

## 1. Requirement

- Linux
- [Xvfb](https://www.x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml)
  - On Ubuntu, you can install `sudo apt update && sudo apt install xvfb`.
- Open GL (for some environment)
  - On Ubuntu, you can install `sudo apt update && sudo apt install python-opengl`

## 2. Installation

You can install from
[PyPI](https://pypi.org/project/gym-notebook-wrapper/) with `pip install gym-notebook-wrapper`


## 3. Usage

Three classes are implemented in `gnwrapper` module in this
gym-notebook-wrapper package.

### 3.1 Simple One Shot Animation

Wrap `gym.Env` class with `gnwrapper.Animation`. That's all! The
`render()` method shows the environment on its output. An example code
is following;

#### 3.1.1 Code

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

#### 3.1.2 Limitation

- Calling `render()` method delete the other output for the same cell.
- The output image is shown only once.


### 3.2 Loop Animation

Wrap `gym.Env` class with `gnwrapper.LoopAnimation`. This wrapper
stores display image when `render()` methos is called and shows the
loop animation by `display(dpi=72,interval=50)` methos.

#### 3.2.1 Code

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


#### 3.2.2 Limitation

- Require a lot of memory to store and display large steps of display
  - Can raise memory error


### 3.3 Movie Animation

Wrap `gum.Env` class with `gnwrapper.Monitor`. This wrapper inherits
`gym.wrappers.Monitor` and implements `display()` method for embedding
mp4 movie into Notebook.

#### 3.3.1 Code

``` python
import gnwrapper
import gym

env = gnwrapper.Monitor(gym.make('CartPole-v1'),directory="./")

o = env.reset()

for _ in range(100):
    o, r, d, i = env.step(env.action_space.sample())
    if d:
        env.reset()

env.display()
```

#### 3.3.2 Limitation

- Require disk space for save movie

## 4. Notes

`gnwrapper.Animation` and `gnwrapper.LoopAnimation` inherit from
`gym.Wrapper`, so that it can access any fields or mothods of
`gym.Env` and `gym.Wrapper` (e.g. `action_space`).


## 5. Links

- [Repository](https://gitlab.com/ymd_h/gym-notebook-wrapper)
- [GitHub Mirror](https://github.com/yamada-github-account/gym-notebook-wrapper)
