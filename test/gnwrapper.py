import unittest
import gnwrapper
import gym

class TestNotebookAnimation(unittest.TestCase):
    def test_render(self):
        env = gnwrapper.Notebook(gym.make("CartPole-v1"))

        env.reset()

        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())
            env.render()

            if d:
                env.reset()
