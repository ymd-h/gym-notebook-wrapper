# Gym-Notebook-Wrapper

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gym-notebook-wrapper)
![PyPI](https://img.shields.io/pypi/v/gym-notebook-wrapper)
![PyPI - Status](https://img.shields.io/pypi/status/gym-notebook-wrapper)
![PyPI - License](https://img.shields.io/pypi/l/gym-notebook-wrapper)



Gym-Notebook-Wrapper provides small wrappers for running and rendering
[OpenAI Gym](https://github.com/openai/gym) and
[Brax](https://github.com/google/brax) on [Jupyter
Notebook](https://jupyter.org/) or similar (e.g. [Google
Colab](https://colab.research.google.com/)).

## 1. Requirement

- Linux
- [Xvfb](https://www.x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml) (for Gym)
  - On Ubuntu, you can install `sudo apt update && sudo apt install xvfb`.
- Open GL (for some environment)
  - On Ubuntu, you can install `sudo apt update && sudo apt install python3-opengl`

## 2. Installation

You can install from
[PyPI](https://pypi.org/project/gym-notebook-wrapper/) with `pip install gym-notebook-wrapper`


## 3. Rendering Gym
> **Warning**  
> Gym has changed its API.
> For example, until v0.25.2 `env.step(action)` returns 4 values,
> but from v0.26.0 it returns 5 values. (`done` was divided to
> `termination` and `truncation`.)


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

env = gnwrapper.Animation(gym.make('CartPole-v1', render_mode="rgb_array"))

obs = env.reset()

for _ in range(1000):
    next_obs, reward, term, trunc, info = env.step(env.action_space.sample())
    env.render()

    obs = next_obs
    if term or trunc:
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

env = gnwrapper.LoopAnimation(gym.make('CartPole-v1', render_mode="rgb_array"))

obs = env.reset()

for _ in range(100):
    next_obs, reward, term, trunc, info = env.step(env.action_space.sample())
    env.render()

    obs = next_obs
    if term or trunc:
        obs = env.reset()

env.display()
```


#### 3.2.2 Limitation

- Require a lot of memory to store and display large steps of display
  - Can raise memory error


### 3.3 Movie Animation

Wrap `gym.Env` class with `gnwrapper.Monitor`. This wrapper inherits
`gym.wrappers.Monitor` (for `gym<=0.19.0`) or `gym.wrapper.RecordVideo`
(for `gym>=0.20.0`), and implements `display()` method for embedding mp4
movie into Notebook.

Note: `gym.wrappers.Monitor` was deprecated at `gym==0.20.0`, which
was released on 14th
September 2021. [See](https://github.com/openai/gym/issues/2297).

If you call `display(reset=True)`, the video list is cleared and the
next `display()` method shows only new videos.

#### 3.3.1 Code

``` python
import gnwrapper
import gym

env = gnwrapper.Monitor(gym.make('CartPole-v1', render_mode="rgb_array"),directory="./")

o = env.reset()

for _ in range(100):
    o, r, term, trunc, i = env.step(env.action_space.sample())
    if term or trunc:
        env.reset()

env.display()
```

#### 3.3.2 Limitation

- Require disk space for save movie

### 3.4 Notes

`gnwrapper.Animation` and `gnwrapper.LoopAnimation` inherit from
`gym.Wrapper`, so that it can access any fields or mothods of
`gym.Env` and `gym.Wrapper` (e.g. `action_space`).


## 4. Rendering Brax
Brax has HTML rendering in `brax.io.html`. We provide small wrapper
classes to record episodes automatically and to display on Jupyter
Notebook easily.

Two classes are implemented in `gnwrapper.brax` module. Since this
module requires `brax` package, the statement `import gnwrapper`
doesn't import `gnwrapper.brax` submodule. You must explicitly import
it by `import gnwrapper.brax` or `from gnwrapper import brax` etc.

### 4.1 HTML Viewer with Brax Native Environment
Wrap `brax.envs.Env` with `gnwrapper.brax.BraxHTML`. `step()` method
automatically stores an episode, and saves it as html file at the
episode end. You can embeds HTML viewer by calling `display()`
method. Of cource, you can open the html file with your local browser
as long as you have internet access. (Data is saved in the html file,
however, the viewer is hosted on CDN.)

Since this wrapper has Python side effect, you cannot wrap `step()` /
`reset()` methods with `jax.jit`. Insted, you can wrap internal
(original) `step()` / `reset()` methods by setting `jit=True` at the
wrapper constructor.

#### 4.1.1 Code
```python
from brax import envs
import brax.jumpy as jp

from gnwrapper.brax import BraxHTML

rng = jp.random_prngkey(seed=42)

ant = BraxHTML(envs.create("ant", auto_reset=False), video_callable = lambda ep: True)

for i in range(2):
    rng, rng_use = jp.random_split(rng)
    state = ant.reset(rng_use)

    while True:
        rng, rng_use = jp.random_split(rng)
        state = ant.step(state, jp.random_uniform(rng_use, (ant.action_size,)))
        if state.done:
        # When `state.done = True`, the episode is written at html file.
            break

# We can get list of recorded episodes.
episodes = ant.recorded_episodes()

# `display()` method show all recorded episodes.
# `display(1)` shows only episode 1, if it is recorded
# `display([1, 2])` shows episode 1 & 2, if they are recorded, etc.
ant.display()
```

#### 4.1.2 Parameters

|Argument|Type|Description|
|---|---|---|
|`env`|`brax.envs.Env`|Environment|
|`directory=None`|`Optional[str]`|Directory to store html. If `None`(default), time stamp (`"%Y%m%d-%H%M%S"`) is used. |
|`heght=480`|`int`|Viewer height in px. (There is a Brax bug ([this issue](https://github.com/google/brax/issues/142)), however, PR was merged.) |
|`video_callable=None`|`Optional[Callable[[int], bool]]`| Function to determine whether each episode is recorded or not. If `None` (default), every 1000 and cubic number less than 1000 are recorded |
|`jit=True`|`bool`|Whether `step`/`reset` methods will be wapped by `jax.jit`|


### 4.2 HTML Viewer with Gym compatible Brax Environment
Wrap `brax.wrappers.GymWrapper` with
`gnwrapper.brax.GymHTML`. `step()` method automatically stores an
episode, and saves it as html file at the episode end. You can embeds
HTML viewer by calling `display()` method. Of cource, you can open the
html file with your local browser as long as you have internet
access. (Data is saved in the html file, however, the viewer is hosted
on CDN.)

Since `brax.wrapper.GymWrapper` already wraps `step()` / `reset()`
methods with `jax.jit`, we don't provide functionality to wrap
`jax.jit` again.

#### 4.2.1 Code

```python
from brax import envs
import brax.jumpy as jp

from gnwrapper.brax import GymHTML

rng = jp.random_prngkey(seed=42)
rng, rng_use = jp.random_split(rng)

ant = GymHTML(envs.create_gym_env("ant", auto_reset=False, seed=0), video_callable = lambda ep: True)

for i in range(2):
    obs = ant.reset()
    while True:
        rng, rng_use = jp.random_split(rng)
        obs, rew, done, _ = ant.step(jp.random_uniform(rng_use, ant.action_space.shape))
        if done:
		    # When `done = True`, the episode is written at html file.
            break

# We can get list of recorded episodes.
episodes = ant.recorded_episodes()

# `display()` method show all recorded episodes.
# `display(1)` shows only episode 1, if it is recorded
# `display([1, 2])` shows episode 1 & 2, if they are recorded, etc.
ant.display()
```

#### 4.2.2 Parameters

|Argument|Type|Description|
|---|---|---|
|`env`|`brax.envs.Env`|Environment|
|`directory=None`|`Optional[str]`|Directory to store html. If `None`(default), time stamp (`"%Y%m%d-%H%M%S"`) is used. |
|`heght=480`|`int`|Viewer height in px. (There is a Brax bug ([this issue](https://github.com/google/brax/issues/142)), however, PR was merged.) |
|`video_callable=None`|`Optional[Callable[[int], bool]]`| Function to determine whether each episode is recorded or not. If `None` (default), every 1000 and cubic number less than 1000 are recorded |


### 4.3 Limitation
Since `done` is always `False`, auto reset
(aka. `brax.envs.wrappers.AutoResetWrapper`) is not supported. You
must call `brax.envs.create()` or `brax.envs.create_gym_env()` with
`auto_reset=False` argument.

Vectorized (batched) environments
(aka. `brax.envs.wrappers.VectorWrapper`,
`brax.envs.wrappers.GymVectorWrapper`) are not supported, too. You
should not specify `batch_size` argument at `brax.envs.create()` or
`brax.envs.create_gym_env()`.

## 5. Links

- [Repository at GitHub](https://github.com/ymd-h/gym-notebook-wrapper)
