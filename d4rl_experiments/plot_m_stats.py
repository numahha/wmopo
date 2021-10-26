import numpy as np
import matplotlib.pyplot as plt
import sys


data = np.loadtxt("progress_alpha0.2_iter1.csv")
plt.plot(data[:,0],data[:,1],color="r", label=r"Real return ($\alpha$=0.2)")
plt.plot(data[:,0],data[:,3],color="r", linestyle = "dashed",label=r"Sim return ($\alpha$=0.2)")

data = np.loadtxt("progress_alpha0.0_iter1.csv")
plt.plot(data[:,0],data[:,1],color="b",label=r"Real return ($\alpha$=0)")
plt.plot(data[:,0],data[:,3],color="b", linestyle = "dashed",label=r"Sim return ($\alpha$=0)")

#plt.title(args[1])
plt.rcParams["legend.framealpha"] = 1
plt.xlabel('Training steps in M-step', fontsize=15)
plt.ylabel('Undiscounted return', fontsize=15)
plt.legend(loc='upper left')
plt.savefig("fig_2_abc.pdf")
plt.show()
