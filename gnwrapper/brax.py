import datetime
import glob
import os
from typing import Optional, Callable, Union, List

from IPython.display import HTML as dHTML, display as ddisplay
import numpy as np

from gym.wrappers.monitor import capped_cubic_video_schedule as default_schedule

from brax.io import html
from brax.io.file import File
from brax.envs import env as benv
import brax.jumpy as jp
import jax

class HTML(benv.Wrapper):
    """
    HTML Wrapper to store Brax trajectory as HTML
    """
    def __init__(self, env: benv.Env, directory: Optional[str]=None, height: int=480,
                 video_callable: Optional[Callable[[int], bool]]=None,
                 jit: bool=True):
        r"""
        Initialize HTML class

        Parameters
        ----------
        env : Brax.envs.Env
            Environment to be wrapped
        directory : str, optional
            Directory to store output html files.
            If ``None`` (default), "%Y%m%d-%H%M%S" is used.
        height : int, optional
            Height in px. The default is ``480``.
        video_callable: (int) -> bool, optional
            Function to determine whether each episode is recorded or not.
        jit : bool
            Whether wrap step/reset function with jax.jit
        """
        super().__init__(env)

        if directory is None:
            directory = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(directory, exist_ok=True)
        self._directory = directory

        self._height = height

        self._episode = 0
        self._callable = video_callable or default_schedule
        self._qps = []

        def step(state, action):
            return self.env.step(state, action)

        def reset(rng):
            return self.env.reset(rng)

        if jit:
            step = jax.jit(step)
            reset = jax.jit(reset)

        self._step = step
        self._reset = reset


    def step(self, state: benv.State, action: jp.ndarray) -> benv.State:
        state = self._step(state, action)

        if self._video_enabled():
            self._qps.append(state.qp)
            if state.done:
                self._save()

        return state

    def reset(self, rng: jp.ndarray) -> benv.State:
        self._episode += 1
        self._qps = []
        return self._reset(rng)

    def _video_enabled(self):
        return self._callable(self._episode)

    def _save(self):
        # Call ``render()`` directly, since ``save_html()`` doesn't take ``height``
        path = os.path.join(self._directory, f"episode-{self._episode}.html")
        with File(path, 'w') as fout:
            fout.write(html.render(self.env.sys, self._qps, self._height))

    def recorded_episode(self):
        htmls = glob.glob(os.path.join(self._directory, "*.html"))
        return sorted([int(h.rsplit("-", maxsplit=1)[-1][:-5]) for h in htmls])

    def display(self, episodes: Optional[Union[int, List[int]]]=None):
        """
        Display saved htmls

        Parameters
        ----------
        episodes: int or list of ints or None
            Episode number(s) to be displayed.
            If ``None`` (default), all the episode will be displayed.
        """
        if episodes is None:
            htmls = glob.glob(os.path.join(self._directory, "*.html"))
        else:
            episodes = np.array(episodes, copy=False, ndmin=1)
            htmls = [os.path.join(self._directory, f"episode-{i}.html")
                     for i in episodes]

        for h in htmls:
            with open(h) as hstr:
                ddisplay(dHTML(hstr.read()))
