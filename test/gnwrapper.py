import unittest
import gnwrapper
import gym


class VirtualDisplay(self):
    def test_init(self):
        env.gnwrapper.VirtualDisplay(gym.make("CartPole-v1"))
        self.assertNotEqual(os.getenv("DISPLAY"),None)


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

if __name__ == "__main__":
    unittest.main()
