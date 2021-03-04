import matplotlib.pyplot as plt
import numpy as np

plt.title('Result Analysis')
plt.plot(list(range(100)), np.random.rand(100), color='green', label='training accuracy')
plt.plot(list(range(100)), np.random.rand(100), color='red', label='testing accuracy')
plt.legend()  # 显示图例

plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()
