import numpy as np
import random
import matplotlib.pyplot as plt

# 采样20个点
left_point = np.pi / 2
right_point = np.pi * 3 / 2
x = np.linspace(left_point, right_point, 20)
origin_data = np.cos(x)

plt.plot(origin_data)
plt.show()
result = 0.0
count = 0


def GoldSplit(left_point, right_point):
    # 迭代求解局部最小值
    global count
    count +=1
    left = left_point
    right = right_point
    tao = 0.618
    a_l = left + (1 - tao) * (right - left)
    a_r = left + tao * (right - left)
    if np.abs(a_r-a_l) < 0.01:
        return a_r
    if np.cos(a_l) < np.cos(a_r):
        right = a_r
        return GoldSplit(left, right)
    elif np.cos(a_l) > np.cos(a_r):
        left = a_l
        return GoldSplit(left, right)

result = GoldSplit(left_point,right_point)
print(result)

print("一共进行了：",count, '轮')

