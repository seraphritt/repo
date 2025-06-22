import matplotlib.pyplot as plt
from scipy import stats

x = [1, 4, 5, 7, 8, 10, 12, 124]
y = [1, 8, 10, 14, 16, 20, 124, 248]

slope, intercept, relations, n, desvio_padrao = stats.linregress(x, y)

def myFunc(x):
    return slope * x + intercept

res = list(map(myFunc, x))

plt.scatter(x, y)
plt.plot(x, res)
plt.show()

print(relations)