import os
import unittest

from brax import envs
import brax.jumpy as jp

from gnwrapper.brax import BraxHTML, GymHTML, _HTML, RaiseWhenAutoReset


class TestBrax(unittest.TestCase):
    def test_raise(self):
        RaiseWhenAutoReset(envs.create("ant", auto_reset=False))
        with self.AssertRaises(ValueError):
            RaiseWhenAutoReset(envs.create("ant", auto_reset=True))

    def test_HTML(self):
        ant = envs.create("ant", auto_reset=False)
        html = _HTML(ant.sys, "test_HTML", 180, lambda ep: True)

        self.assertTrue(os.path.exists("test_HTML"))
        self.assertTrue(html._video_enabled())
        self.assertEqual(html.recorded_episode(), [])
        self.assertEqual(html._episode, 0)

        rng = jp.random_prngkey(42)
        rng, rng_use = jp.random_split(rng_use)
        state = ant.reset(rng_use)
        html.reset()
        self.assertEqual(html._episode, 1)
        self.assertEqual(html._qps, [])
        html.record(state)
        self.assertEqual(len(html._qps), 1)

        html._save()
        self.assertTrue(os.path.exists(os.path.join("test_HTML", "episode-1.html")))

        html.display(0)
        html.display(1)
        html.display([0, 1])
        html.display()

    def test_disabledHTML(self):
        ant = envs.create("ant", auto_reset=False)
        html = _HTML(ant.sys, None, 100, lambda ep: False)

        self.assertFalse(html._video_enabled())
        self.assertEqual(html.recorded_episode(), [])
        self.assertEqual(html._episode, 0)

        rng = jp.random_prngkey(42)
        rng, rng_use = jp.random_split(rng_use)
        state = ant.reset(rng_use)
        html.reset()
        self.assertEqual(html._episode, 1)
        self.assertEqual(html._qps, [])
        html.record(state)
        self.assertEqual(len(html._qps), 0)

    def test_brax(self):
        ant = BraxHTML(envs.create("ant", auto_reset=False),
                       video_callable=lambda ep: True)

        rng = jp.random_prngkey(0)
        rng, rng_use = jp.random_split(rng)
        state = ant.reset(rng_use)

        while True:
            rng, rng_use = jp.random_split(rng)
            state = ant.step(state, jp.random_uniform(rng_use,(ant.action_size,)))
            if state.done:
                break

        self.assertEqual(ant.recorded_episode(), [1])
        ant.display()

    def test_brax_without_jit(self):
        ant = BraxHTML(envs.create("ant", auto_reset=False),
                       video_callable=lambda ep: True,
                       jit=False)

        rng = jp.random_prngkey(0)
        rng, rng_use = jp.random_split(rng)
        state = ant.reset(rng_use)

        while True:
            rng, rng_use = jp.random_split(rng)
            state = ant.step(state, jp.random_uniform(rng_use,(ant.action_size,)))
            if state.done:
                break

        self.assertEqual(ant.recorded_episode(), [1])
        ant.display()

    def test_gym(self):
        ant = GymHTML(envs.create_gym_env("ant", auto_reset=False, seed=42),
                      video_callable=lambda ep: True)

        rng = jp.random_prngkey(0)
        rng, rng_use = jp.random_split(rng)
        obs = ant.reset()

        done = False
        while not done:
            rng, rng_use = jp.random_split(rng)
            obs, rew, done, _ = ant.step(jp.random_uniform(rng_use,
                                                           ant.action_space.shape))

        self.assertEqual(ant.recorded_episode(), [1])
        ant.display()



if __name__ == "__main__":
    unittest.main()
