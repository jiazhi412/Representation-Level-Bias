import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')
dir = './result/colored_model/reproduce'
name = 'colored_model_color_{}_category.npy'

std_list = [float(i)/10 for i in range(0,10,1)]
align_length = 40000
for i in range(len(std_list)):
    path = os.path.join(dir, name.format(std_list[i]))
    bias = np.load(path)
    bias[0] = 0
    plt.plot(range(bias.shape[0]), bias, label = str(std_list[i]))    
    # plt.show()

plt.legend()
plt.xlabel("Iteration", fontsize=15)
plt.ylabel("Mutual Information Estimation", fontsize=15)
plt.savefig("colored_mnist_convergence.png")

