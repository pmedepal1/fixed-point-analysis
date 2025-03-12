import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
from gaussian_sig_test import test_normality

# Let's make our quantization bins ... -2q, -q, 0, q, 2q, ...,
# We don't restrict how many bins we have. 
q = 0.5
NUM_EXPERIMENTS = 10


# Set the mean and standard deviation, and number of samples drawn
mean = q/4
std_dev = q/2
N = 10**6
quantized_errors = np.array([])

#for i in range(NUM_EXPERIMENTS):
# Generate N values drawn from a normal distribution
values = np.random.normal(mean, std_dev, N)
quantized_values = q * np.round(values / q)
quantized_errors = quantized_values - values

avg_error = np.mean(quantized_errors)
std_dev_error = np.std(quantized_errors)

plt.hist(quantized_errors.flatten(), bins=100, alpha=0.5) #, label=f'Experimental noise')
plt.xticks(list(plt.xticks()[0]) + [quantized_errors.min(), quantized_errors.max()])
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of Quantized Error, N={N:.1e}, q={q}\nmean={mean}, std_dev={std_dev}')
plt.savefig('quantized_error.png')

print('Average error: ', avg_error, 'Standard Dev of error: ', std_dev_error)

# Confidence interval for average error
confidence_level = 0.95
degrees_freedom = N - 1
sample_mean = avg_error
sample_std_error = std_dev/np.sqrt(N)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, loc=sample_mean, scale=sample_std_error)

print(f"Confidence Interval for Average Error: {confidence_interval}")

# Confidence interval for standard deviation of error
alpha = 1 - confidence_level
lower_percentile = alpha / 2
upper_percentile = 1 - alpha / 2

lower_bound = np.sqrt(((N - 1) * std_dev_error**2) / stats.chi2.ppf(1 - lower_percentile, N - 1))
upper_bound = np.sqrt(((N - 1) * std_dev_error**2) / stats.chi2.ppf(lower_percentile, N - 1))

print(f"Confidence Interval for Standard Deviation of Error: ({lower_bound}, {upper_bound})")

# Run normality tests on the input distribution - sanity check since the input
# is drawn from a normal distribution.
normality_results = test_normality(values)
for test, result in normality_results.items():
    print(f"{test} Test:")
    print(f"  Statistic: {result['statistic']:.4f}")
    print(f"  p-value: {result['p-value']:.4f}")
    print(f"  Data follows normal distribution: {result['normal']}")
    print("-")
