from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time


times = np.linspace(0.0, 10.0, 200)

psi0 = tensor(fock(10, 2), fock(10, 5))

a = tensor(qeye(10), destroy(10))

sm = tensor(destroy(10), qeye(10))

H = 2 * np.pi * a.dag() * a + 2 * np.pi * sm.dag() * sm + 2 * \
    np.pi * 0.25 * (sm * a.dag() + sm.dag() * a)

data = mcsolve(H, psi0, times, [np.sqrt(0.1) * a],
               [a.dag() * a, sm.dag() * sm])

print("done")
start_time = time.time()
data1 = mesolve(H, psi0, times, [np.sqrt(0.1) * a],
                [a.dag() * a, sm.dag() * sm])

end_time = time.time()
print("Time taken for mesolve: ", end_time - start_time)

# plt.plot(times, data.expect[0], times, data.expect[1])
# plt.plot(times, data1.expect[0], times, data1.expect[1])

plt.plot(times, data.expect[0], 'r', label='data.expect[0]')
plt.plot(times, data.expect[1], 'g', label='data.expect[1]')
plt.plot(times, data1.expect[0], 'b', label='data1.expect[0]')
plt.plot(times, data1.expect[1], 'm', label='data1.expect[1]')
plt.show()
