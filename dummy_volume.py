# %%

import numpy as np
import matplotlib.pyplot as plt

# %%

data = np.random.exponential(scale=1, size=7_000)

spoofed = 2 + 1.5 * np.random.rand(1200)

spoofed = spoofed[spoofed > 0]

# %%

plt.hist(data, bins=100, density=True)
xs = np.linspace(0, 10, 100)
ys = np.exp(-xs)
plt.plot(xs, ys, color="red")

plt.show()

# %%
concatenated = np.concatenate([data, spoofed])

# %%

plt.hist(concatenated, bins=100, density=True)
xs = np.linspace(0, 10, 100)
ys = np.exp(-xs)
plt.plot(xs, ys, color="red")
plt.xlim(0, 6)

# %%

np.save("data/dummy_volume.npy", concatenated)

# %%
spoofed2 = 2 + 1.5 * np.random.rand(1600)

concatenated2 = np.concatenate([data, spoofed2])

# %%

plt.hist(concatenated2, bins=100, density=True)
xs = np.linspace(0, 10, 100)
ys = np.exp(-xs)
plt.plot(xs, ys, color="red")
plt.xlim(0, 6)
# %%

rates = np.random.normal(loc=5, scale=1, size=2_000)

rates = rates[rates > 0]

spoofed_rates = np.random.normal(loc=8, scale=0.6, size=500)

all_rates = np.concatenate([rates, spoofed_rates])

# %%

plt.hist(all_rates, bins=100, density=True)
xs = np.linspace(0, 15, 100)
var = 1
loc = 5

ys = 1 / np.sqrt(2 * np.pi * var) * np.exp(-((xs - loc) ** 2) / (2 * var))

plt.plot(xs, ys, color="red")

# %%

np.save("data/dummy_rates.npy", all_rates)

# %%
