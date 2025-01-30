import matplotlib.pyplot as plt
import numpy as np

B_up = [34, 1476, 2700, 3950, 4980, 6230, 7220, 8070, 8740]
I_up = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

I_down = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
B_down = [8200, 6610, 6150, 5350, 4150, 3010, 1580, 350]

up_a, up_b = np.polyfit(I_up, B_up, 1)
down_a, down_b = np.polyfit(I_down, B_down, 1)

plt.plot(I_down, B_down, 'ro-', label='Down')
plt.plot(I_up, B_up, 'bo-', label='Up')
#plt.plot(I_up, up_a*np.array(I_up) + up_b, 'b')
#plt.plot(I_down, down_a*np.array(I_down) + down_b, 'r')
plt.ylabel('Magnetic Field (G)')
plt.xlabel('Current (A)')
plt.title('Hysteresis Loop')
plt.grid()
plt.legend()
plt.savefig('hysteresis_loop.png')
plt.show()

# Perform linear fits with covariance matrix
params_up, cov_up = np.polyfit(I_up, B_up, 1, cov=True)
params_down, cov_down = np.polyfit(I_down, B_down, 1, cov=True)

# Extract coefficients and errors
up_a, up_b = params_up
down_a, down_b = params_down

# Calculate standard errors for slope (a) and intercept (b)
up_a_error = np.sqrt(cov_up[0, 0])
up_b_error = np.sqrt(cov_up[1, 1])
down_a_error = np.sqrt(cov_down[0, 0])
down_b_error = np.sqrt(cov_down[1, 1])

print("Up fit: B = (%.1f ± %.1f)I + (%.1f ± %.1f)" % (up_a, up_a_error, up_b, up_b_error))
print("Down fit: B = (%.1f ± %.1f)I + (%.1f ± %.1f)" % (down_a, down_a_error, down_b, down_b_error))




