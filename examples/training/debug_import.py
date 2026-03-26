import sys
print(f"Python Executable: {sys.executable}")

print("1. Attempting to import gym_pusht...")
import gym_pusht
print(f"   Success. File: {gym_pusht.__file__}")

print("2. Attempting to import PushTEnv directly...")
# 这行代码之前被 try 块包裹了，现在裸奔运行
from gym_pusht.envs.pusht_env import PushTEnv
print("   Success. Class found.")