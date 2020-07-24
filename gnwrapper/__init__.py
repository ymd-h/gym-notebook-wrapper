import base64
import io
import os

from gym import Wrapper
from gym.wrappers import Monitor as _monitor
from IPython import display
import matplotlib.pyplot as plt
from matplotlib import animation
from pyvirtualdisplay import Display


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
        self._display = None
        self._ensure_display()

    def _ensure_display(self):
        """
        Ensure to start virtual display
        """
        # To avoid starting multiple virtual display
        if not os.getenv("DISPLAY",None):
            self._display = self._display or Display(visible=0, size=self.size)
            self._display.start()

    def render(self,mode=None,**kwargs):
        """
        Render environment
        """
        self._ensure_display()
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
    def __init__(self,env,directory,size=(1024, 768),*args,**kwargs):
        VirtualDisplay(env,size)
        super().__init__(env,directory,*args,**kwargs)

    def display(self):
        """
        Display saved all movies
        """
        for f in self.videos:
            video = io.open(f[0], "r+b").read()
            encoded = base64.b64encode(video)

            display.display(os.path.basename(f[0]))
            display.display(display.HTML(data="""
            <video alt="test" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>
            """.format(encoded.decode('ascii'))))
