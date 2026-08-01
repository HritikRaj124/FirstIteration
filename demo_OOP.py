import time
import random

def fake_control_tick(target):
  start = time.perf_counter()

  distance_to_danger = random.uniform(0,1 )
  slowed_down = distance_to_danger < 0.3

  elapsed_ms = (time.perf_counter() - start) * 1000

  return {
    "slowed_down": slowed_down,
    "distance_to_danger": distance_to_danger,
    "compute_time_ms": elapsed_ms,
  }


log = {"t": [], "slowed_down": [], "distance_to_danger": [], "compute_time_ms": []}

for tick in range(5):
  t = tick * 0.02
  info = fake_control_tick(target=1.0)

  log["t"].append(t)
  log["slowed_down"].append(info["slowed_down"])
  log["distance_to_danger"].append(info["distance_to_danger"])
  log["compute_time_ms"].append(info["compute_time_ms"])

print(log)