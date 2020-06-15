# Gym-Notebook-Wrapper

Gym-Notebook-Wrapper provides small wrapper for running and rendering
[OpenAI gym](https://github.com/openai/gym) on [Jupyter
Notebook](https://jupyter.org/) or similar (e.g. [Google
Corab](https://colab.research.google.com/)).

# Requirement

- Linux
- [Xvfb](https://www.x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml)


On Ubuntu, you can install `sudo apt update && sudo apt install xvfb`.


# Installation



# Usage

Wrap `gym.Env` class with `gnwrapper.Notebook`. That's all! The
`render()` method shows the environment on its output. An example code
is following;

``` python
import gnwrapper
import gym

env = gnwrapper.Notebook(gym.make('CartPole-v1'))

obs = env.reset()

for _ in range(1000):
    next_obs, reward, done, info = env.step(env.action_space.sample())
	env.render()

	obs = next_obs
	if done:
	    obs = env.reset()
```

# Limitation

- Calling `render()` method delete the other output for the same cell.
- The output image is shown only once.
