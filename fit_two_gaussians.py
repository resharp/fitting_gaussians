# do not use pylab!
# https://www.tutorialspoint.com/matplotlib/matplotlib_pylab_module.htm
from numpy.core.multiarray import concatenate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import pandas as pd


df_data = pd.read_csv("data/test data.csv", names=["weerstand"])

df_data = df_data.sort_values(by="weerstand")

plt.hist(df_data.weerstand, bins=400)
plt.show()

min_value = df_data.weerstand.min()
max_value = df_data.weerstand.max()
min_int = int(min_value)
max_int = int(max_value)

bins = np.arange(min_int, max_int, 1)

pdf = np.histogram(df_data.weerstand, bins=bins)[0]

bins = np.delete(bins, 0) # remove first
plt.plot(bins, pdf)
plt.show()

# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
# data = concatenate((np.random.normal(loc=1, scale=.2, size=5000),
#                     np.random.normal(loc=1.8, scale=.2, size=500)))
#
# y, x, _ = plt.hist( data, 100, alpha=.3, label='data')
#
# x = (x[1:] + x[:-1])/2  # for len(x)==len(y)

y = pdf
x = bins

plt.plot( x, y, alpha=.3, label='data')

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


expected = (260, 12, 320, 300, 20, 25)

params, cov = curve_fit(bimodal, x, y, expected)
sigma = np.sqrt(np.diag(cov))

plt.plot(x, bimodal(x, *params), color='red', lw=3, label='model')
plt.legend()
plt.title("mean 2nd distribution found: {m}".format(m=params[3].round(3)))
plt.show()

print("params:")
print(params)
print("sigma (uncertainty in the parameter estimation):")
print(sigma)

mean_2nd = params[3]
sd_2nd = params[4]

print("mean: " + str(mean_2nd))
print("sd: " + str(sd_2nd))

threshold = 350

# survival function:
chance = stat.norm.sf(threshold, mean_2nd, sd_2nd)

print("Chance of exceeding threshold {threshold}: {chance}".format(
    threshold=threshold, chance=chance))

