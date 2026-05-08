"""Test real data integration"""
import sys
sys.path.insert(0, '.')

from env.smart_grid_env import SmartGridEnv

print("Testing Real Data Integration...")
print("=" * 60)

# Test with real data
env = SmartGridEnv(use_real_data=True)
obs, info = env.reset()

print("[OK] Real data environment initialized!")
print(f"  Hour: {info['hour']}")
print(f"  PV: {info['pv']:.2f} kW")
print(f"  Load: {info['load']:.2f} kW")
print(f"  Price: ${info['price']:.3f}/kWh")
print(f"  SOC: {info['soc']:.2f}")
print(f"  Battery Health: {info['battery_health']:.4f}")
print(f"  Cycle Count: {info['cycle_count']:.4f}")
print(f"  Comfort: {info['comfort']:.2f}")

# Test a few steps
print("\nTesting steps with real data...")
for i in range(3):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step {i+1}: Action={env.action_names[action]}, Reward={reward:.2f}, Hour={info['hour']}")

print("\n[OK] Real data integration works correctly!")
