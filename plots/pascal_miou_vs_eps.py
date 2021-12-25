import numpy as np
import matplotlib.pyplot as plt

# eps_vec = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, 32])
# miou_vec = 0.01 * np.array([77.68, 68.34, 57.88, 43.33, 32.67, 27.34, 25.57, 23.72, 13.45])

eps_vec = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32])
miou_vec = 0.01 * np.array([68.34, 57.88, 43.33, 32.67, 27.34, 25.57, 23.72, 13.45])
plt.semilogx(eps_vec, miou_vec)
plt.axhline(y=0.7768, color='r', linestyle='--')
plt.axhline(y=0.772, color='g', linestyle='--')
plt.xticks([0.25, 0.5, 1, 2, 4, 8, 16, 32], [0.25, 0.5, 1, 2, 4, 8, 16, 32])
plt.xlabel('1/255 * eps')
plt.ylabel('mIOU')
plt.title('PASCAL VOC mIOU vs FGSM noise power')
plt.show()
