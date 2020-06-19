import os
import unittest

import gnwrapper
import gym


class TestVirtualDisplay(unittest.TestCase):
    def test_init(self):
        env = gnwrapper.VirtualDisplay(gym.make("CartPole-v1"))
        self.assertIsNotNone(os.getenv("DISPLAY"))

    def test_render_return(self):
        env = gnwrapper.VirtualDisplay(gym.make("CartPole-v1"))
        env.reset()
        self.assertIsNotNone(env.render())


class TestAnimation(unittest.TestCase):
    def test_render(self):
        env = gnwrapper.Animation(gym.make("CartPole-v1"))

        env.reset()

        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())
            env.render()

            if d:
                env.reset()

class TestLoopAnimation(unittest.TestCase):
    def test_render(self):
        env = gnwrapper.LoopAnimation(gym.make("CartPole-v1"))

        env.reset()

        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())
            env.render()

            if d:
                env.reset()

        env.display()

class TestMonitor(unittest.TestCase):
    def test_display(self):
        env = gnwrapper.Monitor(gym.make('CartPole-v1'),directory="./")

        env.reset()

        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())

            if d:
                env.reset()

        env.display()

if __name__ == "__main__":
    unittest.main()
