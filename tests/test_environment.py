import unittest
import numpy as np

import config
from env.smart_grid_env import SmartGridEnv


class TestEnvironmentBasics(unittest.TestCase):
    def test_reset_and_step_shapes(self):
        env = SmartGridEnv()
        obs, info = env.reset(seed=123)
        self.assertEqual(obs.shape[0], 7)
        self.assertIn("soc", info)

        obs2, reward, done, truncated, info2 = env.step(0)
        self.assertEqual(obs2.shape[0], 7)
        self.assertIsInstance(float(reward), float)
        self.assertIn("grid_import", info2)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)

    def test_reward_reasonable_range(self):
        env = SmartGridEnv()
        env.reset(seed=7)
        rewards = []
        for _ in range(24):
            _, reward, done, truncated, _ = env.step(env.action_space.sample())
            rewards.append(float(reward))
            if done or truncated:
                break
        self.assertTrue(all(np.isfinite(r) for r in rewards))
        self.assertGreaterEqual(min(rewards), -1000.0)
        self.assertLessEqual(max(rewards), 50.0)


if __name__ == "__main__":
    unittest.main()
