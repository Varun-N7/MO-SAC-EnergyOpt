import unittest

import config
from env.smart_grid_env import SmartGridEnv


class TestBatteryDegradation(unittest.TestCase):
    def test_battery_health_degrades_with_cycles(self):
        env = SmartGridEnv()
        env.reset(seed=42)

        initial_health = env.battery_health
        initial_cycles = env.cycle_count

        # Push repeated charge/discharge actions to induce throughput/cycles
        for _ in range(200):
            env.pv = env.base_load + 10.0  # ensure charge action can move SOC
            env.step(2)  # charge
            env.step(1)  # discharge

        self.assertGreater(env.cycle_count, initial_cycles)
        self.assertLessEqual(env.battery_health, initial_health)
        self.assertGreaterEqual(env.battery_health, config.BATTERY_MIN_HEALTH)


if __name__ == "__main__":
    unittest.main()
