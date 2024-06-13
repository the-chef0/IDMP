import pickle
import numpy as np
from matplotlib import pyplot as plt

file = open("l_infs_tsp.pickle",'rb')
l_infs = pickle.load(file)

file = open("l_infs_relative_tsp.pickle",'rb')
l_infs_relative = pickle.load(file)

file = open("mse_vals_tsp.pickle",'rb')
MSE_vals = pickle.load(file)

l_infs = np.array(l_infs) / 100
MSE_vals = np.array(MSE_vals) / 100

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))
    
ax1.hist(MSE_vals, bins=30)
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('Frequency')
ax1.set_title("MSE")

ax2.hist(l_infs, bins=30)
ax2.set_xlabel('Magnitude')
ax2.set_ylabel('Frequency')
ax2.set_title("L-infinity norm")

ax3.hist(l_infs_relative, bins=30)
ax3.set_xlabel("% of CaVE gradient range")
ax3.set_ylabel('Frequency')
ax3.set_title("Relative L-inf norm")

fig.suptitle("Goodness of cubic fit on CaVE gradient", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("errors_tsp.pdf")