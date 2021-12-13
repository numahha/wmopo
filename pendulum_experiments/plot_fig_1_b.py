import numpy as np
import matplotlib.pyplot as plt


data = []
for i in range(10):
    data.append(np.loadtxt("alpha1_"+str(i+1)+"/temp_mle_e_step_score_alpha1.0_skipgradFalse.csv",delimiter=','))
data = np.array(data)[:,:11]
plt.errorbar(range(data.shape[1]),-data.mean(axis=0),yerr=data.std(axis=0),label="Alg.1 (full version)")


data = []
for i in range(10):
    data.append(np.loadtxt("skipalpha1_"+str(i+1)+"/temp_mle_e_step_score_alpha1.0_skipgradTrue.csv",delimiter=','))
data = np.array(data)[:,:11]
plt.errorbar(range(data.shape[1]),-data.mean(axis=0),yerr=data.std(axis=0),label="Alg.2 (simplified version)")


plt.xlabel('Iteration',fontsize=15)
plt.ylabel('Weighted loss function',fontsize=15)
#plt.xlim([-0.5, 11.5])
plt.legend(fontsize=15)
plt.savefig("fig_1_b.pdf")
plt.show()


