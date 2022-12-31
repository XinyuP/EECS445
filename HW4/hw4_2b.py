import math
import numpy as np
theta = [0.5, 0.3, 0.2, 0.2, 0.3, 0.5]
k = ['A', 'B']
z = []
sunny = [4, 7, 1, 8, 3, 3]
cloudy = [2, 1, 2, 1, 3, 2]
rainy = [4, 2, 7, 1, 4, 5]

for i in range(6):
    denominator = pow(theta[0], sunny[i]) * pow(theta[1], cloudy[i]) * pow(theta[2], rainy[i]) + \
        pow(theta[3], sunny[i]) * pow(theta[4],
                                      cloudy[i]) * pow(theta[5], rainy[i])

    p_a = pow(theta[0], sunny[i]) * pow(theta[1], cloudy[i]) * \
        pow(theta[2], rainy[i]) / denominator
    p_b = pow(theta[3], sunny[i]) * pow(theta[4], cloudy[i]) * \
        pow(theta[5], rainy[i]) / denominator

    z.append(np.argmax([p_a, p_b]))

    print("i = ", i)
    print("p_a: ", p_a)
    print("p_b: ", p_b)
    print("city with highest probability: ", k[np.argmax([p_a, p_b])])
