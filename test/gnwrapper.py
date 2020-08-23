import os
import unittest
from unittest.mock import MagicMock, patch
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
                self.assertIsNotNone(re.search(r"[0-9]{8}-[0-9]{6}",f[0]))
        env.display()

    def test_last_video(self):
        """
        Check the last video is flushed

        Ref: https://gitlab.com/ymd_h/gym-notebook-wrapper/-/issues/2
        """
        env = gnwrapper.Monitor(gym.make('CartPole-v1'),
                                directory="./test_last_videos/",
                                video_callable=lambda ep: True)
        env.reset()

        n_video = 1
        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())

            if d:
                env.reset()
                n_video += 1

        env.display()
        self.assertEqual(len(env.videos),n_video)
        for f in env.videos:
            with self.subTest(file=f[0]):
                self.assertTrue(os.path.exists(f[0]))

        # Can run normally after
        env.reset()
        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())

            if d:
                env.reset()

        env.display()

    def test_KeyboardInterrupt(self):
        """
        After KeyboardInterrupt, notebook kernel dies.

        Ref: https://gitlab.com/ymd_h/gym-notebook-wrapper/-/issues/4
        """
        CartPole = "gym.envs.classic_control.cartpole.CartPoleEnv"
        VideoRecorder = "gym.wrappers.monitoring.video_recorder.VideoRecorder"

        env = gnwrapper.Monitor(gym.make('CartPole-v1'),
                                directory="./test_keyboard_interrupt/",
                                video_callable=lambda ep: True)

        for func in [f"{CartPole}.step",
                     f"{VideoRecorder}.capture_frame"]:
            env.reset()
            with self.subTest(function=func):
                with patch(func,
                           MagicMock(side_effect=KeyboardInterrupt)):
                    with self.assertRaises(KeyboardInterrupt):
                        env.step(env.action_space.sample())

                env.reset()
                env.step(env.action_space.sample())
                env.display()
                env.render(mode='rgb_array')

            env.reset()
            with patch("io.BytesIO",MagicMock()) as F:
                F.write = MagicMock(side_effect=KeyboardInterrupt)
                with self.assertRaises(KeyboardInterrupt):
                    env.step(env.action_space.sample())

            env.reset()
            env.step(env.action_space.sample())
            env.display()
            env.render(mode='rgb_array')


        for func in [f"{CartPole}.reset",
                     "os.waitpid"]:
            env.reset()
            with self.subTest(function=func):
                with patch(func,
                           MagicMock(side_effect=KeyboardInterrupt)):
                    with self.assertRaises(KeyboardInterrupt):
                        env.reset()

                env.reset()
                env.step(env.action_space.sample())
                env.display()
                env.render(mode='rgb_array')

    def test_display_after_close(self):
        """
        Display after close
        """
        env = gnwrapper.Monitor(gym.make('CartPole-v1'),
                                directory="./test_display_after_close/",
                                video_callable=lambda ep: True)
        env.reset()

        for _ in range(100):
            o, r, d, i = env.step(env.action_space.sample())

            if d:
                env.reset()
        env.close()
        env.display()


if __name__ == "__main__":
    unittest.main()
