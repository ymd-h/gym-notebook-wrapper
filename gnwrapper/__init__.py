import base64
import datetime
import io
import os
from typing import Optional, Callable
import subprocess
from unittest.mock import patch

from gym import Wrapper

try:
    # gym >= 0.20.0
    from gym.wrappers import RecordVideo as _monitor
    _video_callable_key = "episode_trigger"
except ImportError:
    # gym <= 0.19.0
    from gym.wrappers import Monitor as _monitor
    _video_callable_key = "video_callable"

from IPython import display
import matplotlib.pyplot as plt
from matplotlib import animation
from pyvirtualdisplay import Display


class _VirtualDisplaySingleton(object):
    def __new__(cls,*args,**kwargs):
        if not hasattr(cls,"_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,size=(1024, 768)):
        self.size = size

        if not hasattr(self,"_display"):
            self._display = Display(visible=0,size=self.size)

            original = subprocess.Popen
            def Popen(cmd,pass_fds,stdout,stderr,shell):
                return original(cmd,pass_fds=pass_fds,
                                stdout=stdout,stderr=stderr,
                                shell=shell,preexec_fn=os.setpgrp)

            with patch("subprocess.Popen",Popen):
                self._display.start()

    def _restart_display(self):
        self._display.stop()
        self._display.start()


class VirtualDisplay(Wrapper):
    """
    Wrapper for running Xvfb
    """
    def __init__(self,env,size=(1024, 768)):
        """
        Wrapping environment and start Xvfb
        """
        super().__init__(env)
        self.size = size
        self._display = _VirtualDisplaySingleton(size)

    def render(self,mode=None,**kwargs):
        """
        Render environment
        """
        return self.env.render(mode='rgb_array',**kwargs)


class Animation(VirtualDisplay):
    """
    Wrapper for running/rendering OpenAI Gym environment on Notebook
    """
    def __init__(self,env,size=(1024, 768)):
        """
        Wrapping environment for Notebook

        Parameters
        ----------
        env : gym.Env
            Environment to be wrapped
        size : array-like, optional
            Virtual display size, whose default is (1024,768)
        """
        super().__init__(env,size)

        self._img = None

    def render(self,mode=None,**kwargs):
        """
        Render the environment on Notebook

        Parameters
        ----------
        mode : str
            If "rgb_array", return display image

        Returns
        -------
        img : numpy.ndarray or None
            Rendering image when mode == "rgb_array"
        """
        display.clear_output(wait=True)
        _img = self.env.render(mode='rgb_array',**kwargs)
        if self._img is None:
            self._img = plt.imshow(_img)
        else:
            self._img.set_data(_img)

        plt.axis('off')
        display.display(plt.gcf())

        if mode == 'rgb_array':
            return _img

class LoopAnimation(VirtualDisplay):
    """
    Wrapper for OpenAI Gym to display loop animation on Notebook
    """
    def __init__(self,env,size=(1024, 768)):
        """
        Wrap environment for Notebook

        Parameters
        ----------
        env : gym.Env
            Environment to be wrapperd
        size : array-like, optional
            Virtual display size, whose default is (1024, 768)
        """
        super().__init__(env,size)

        self._img = []

    def render(self,mode=None,**kwargs):
        """
        Store rendered image into internal buffer

        Parameters
        ----------
        mode : str
            If "rgb_array", return display image

        Returns
        -------
        img : numpy.ndarray or None
            Rendering image when mode == "rgb_array"
        """
        self._img.append(self.env.render(mode='rgb_array',**kwargs))

        if mode == 'rgb_array':
            return self._img[-1]

    def display(self,*,dpi=72,interval=50):
        """
        Display saved images as loop animation
        """
        plt.figure(figsize=(self._img[0].shape[1]/dpi,
                            self._img[0].shape[0]/dpi),
                   dpi=dpi)
        patch = plt.imshow(self._img[0])
        plt.axis=('off')
        animate = lambda i: patch.set_data(self._img[i])
        ani = animation.FuncAnimation(plt.gcf(),animate,
                                      frames=len(self._img),interval=interval)
        display.display(display.HTML(ani.to_jshtml()))

class Monitor(_monitor):
    """
    Monitor wrapper to store images as videos.

    This class is a shin wrapper for ``gym.wrappers.Monitor`` (gym <= 0.19.0)
    or ``gym.wrappers.RecordVideo`` (gym >= 0.20.0). This class also
    have a method `display`, which shows recorded movies on Notebook.

    See Also
    --------
    gym.wrappers.Monitor : https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py
    gym.wrappers.RecordVideo : https://github.com/openai/gym/blob/master/gym/wrappers/record_video.py
    """
    def __init__(self, env, directory: Optional[str] = None, size = (1024, 768),
                 video_callable: Callable[[int], bool] = None,
                 *args,**kwargs):
        """
        Initialize Monitor class

        Parameters
        ----------
        env : gym.Env
            Environment to be recorded
        directory : str, optional
            Directory to store output movies. When the value is `None`,
            which is default, "%Y%m%d-%H%M%S" is used for directory.
        video_callable : (int) -> bool, optional
            Function to determine whether each episode is recorded or not.
            If ``None`` (default), every 1000 episodes and cubic numbers
            less than 1000 are recorded.
        *args, **kwargs
            Additional arguments and keyword arguments to be passed to
            base class.
        """
        if directory is None:
            directory = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self._display = _VirtualDisplaySingleton(size)

        kwargs[_video_callable_key] = video_callable
        super().__init__(env, directory, *args, **kwargs)
        if not hasattr(self, "videos"):
            # gym >= 0.20.0
            self.videos = []

    def _close_running_video(self):
        if self.video_recorder:
            if hasattr(self, "_close_video_recorder"):
                # gym <= 0.19.0
                self._close_video_recorder()
                self._flush(force=True)
            else:
                # gym >= 0.20.0
                self.close_video_recorder()
                if self.video_recorder.functional:
                    self.videos.append((self.video_recorder.path,
                                        self.video_recorder.metadata_path))
            self.video_recorder = None

    def step(self,action):
        """
        Step Environment
        """
        try:
            return super().step(action)
        except KeyboardInterrupt as k:
            self._close_running_video()
            raise

    def reset(self,**kwargs):
        """
        Reset Environment
        """
        try:
            if hasattr(self, "stats_recorder"):
                # gym <= 0.19.0
                if self.stats_recorder and not self.stats_recorder.done:
                    # StatsRecorder requires `done=True` before `reset()`
                    self.stats_recorder.done = True
                    self.stats_recorder.save_complete()
            else:
                # gym >= 0.20.0
                self._close_running_video()

            return super().reset(**kwargs)
        except KeyboardInterrupt:
            self._close_running_video()
            raise

    def display(self,reset: bool=False):
        """
        Display saved all movies

        If video is running, stop and flush the current video then display all.

        Parameters
        ----------
        reset : bool, optional
            When `True`, clear current video list. This does not delete movie files.
            The default value is `False`, which keeps video list.
        """

        # Close current video.
        self._close_running_video()

        for f in self.videos:
            if not os.path.exists(f[0]):
                continue

            video = io.open(f[0], "r+b").read()
            encoded = base64.b64encode(video)

            display.display(os.path.basename(f[0]))
            display.display(display.HTML(data="""
            <video alt="test" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>
            """.format(encoded.decode('ascii'))))

        if reset:
            self.videos = []
