import os
import unittest
import re

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

    def test_reset_videos(self):
        env = gnwrapper.Monitor(gym.make('CartPole-v1'),
                                directory="./test_reset_videos/")

        env.reset()
        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())

            if d:
                env.reset()
                self.assertNotEqual(len(env.videos),0)

        env.display(reset=True)
        self.assertEqual(len(env.videos),0)

    def test_default_directory(self):
        env = gnwrapper.Monitor(gym.make('CartPole-v1'))

        env.reset()

        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())

            if d:
                env.reset()

        for f in env.videos:
            with self.subTest(file=f[0]):
                self.assertNotNone(re.search(r"[0-9]{8}-[0-9]{6}",f[0]))
        env.display()

if __name__ == "__main__":
    unittest.main()
