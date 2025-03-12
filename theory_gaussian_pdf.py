import numpy as np 
import mpmath as mp # https://mpmath.org/doc/current/functions/elliptic.html - used for jacobian elliptic function evaluation
import matplotlib.pyplot as plt 

# Code to plot PDF of quantization noise for a gaussian distribution. We can 
# Settings control for the quantization grid size, mean, and std dev.
q = 0.5
mu = q/4 #q/4
sigma = q/2

# Calculate the parameters for the jacobian elliptic function
psi = 2*np.pi/q
k_prime = psi*mu 
m = sigma**2 * psi**2 / 2
term_exponent = np.exp(-m)

# Create a range of x values
num_points = 500
x = np.linspace(-2*q, 2*q, num_points)

# Calculate the PDF of the quantization noise
y = np.zeros(num_points)
for i in range(num_points):
  if x[i] >= -q/2 and x[i] <= q/2:
    y[i] = 1/q * mp.jtheta(3, (k_prime + psi*x[i])/2, term_exponent)

# Plot the graph
plt.plot(x, y)

# Add labels and title
plt.xlabel('e (error)')
plt.ylabel('f_e PDF')
plt.title(f'Plot of quantization noise pdf, q={q}')

# Show the plot
plt.savefig('plot_shift.png')


# Calculate expectation value and variance of the quantization noise

expectation = (-q/np.pi) * term_exponent * np.sin(2*np.pi*mu/q)
variance= (q**2/12) - (q**2/np.pi**2) * term_exponent * np.sin(2*np.pi*mu/q)
std_dev = variance**0.5
print('Expectation: ', expectation, 'Std Dev: ', std_dev)

