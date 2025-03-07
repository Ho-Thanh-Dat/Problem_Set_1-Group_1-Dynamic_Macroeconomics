import numpy as np
import matplotlib.pyplot as plt

def rouwenhorst(n, p, q, sigma, mu):
    """
    Generate Rouwenhorst's method transition matrix and state space for AR(1)
    """
    pi = np.array([[p, 1 - p], [1 - q, q]])
    
for i in range(2, n):
        pi_temp = np.zeros((i + 1, i + 1))
        pi_temp[:i, :i] += p * pi
        pi_temp[:i, 1:] += (1 - p) * pi
        pi_temp[1:, :i] += (1 - q) * pi
        pi_temp[1:, 1:] += q * pi
        pi = pi_temp / pi_temp.sum(axis=1, keepdims=True)
    
z = np.linspace(-sigma * np.sqrt(n - 1), sigma * np.sqrt(n - 1), n) + mu
    return z, pi

def simulate_markov_chain(P, states, periods, seed=2025):
    """Simulate a Markov Chain given a transition matrix and state space."""
    np.random.seed(seed)
    n = len(states)
    state_idx = np.random.choice(n)  # Start with a random state
    simulation = []
    
for _ in range(periods):
        simulation.append(states[state_idx])
        state_idx = np.random.choice(n, p=P[state_idx])
    
return np.array(simulation)

# Parameters
n_states = 7
sig_eps = 1
mu = 0.5
gamma_values = [0.75, 0.85, 0.95, 0.99]
periods = 50

time = np.arange(periods)
plt.figure(figsize=(10, 6))

for gamma in gamma_values:
    sigma = sig_eps / np.sqrt(1 - gamma**2)
    states, P = rouwenhorst(n_states, (1 + gamma) / 2, (1 + gamma) / 2, sigma, mu)
    simulation = simulate_markov_chain(P, states, periods)
    plt.plot(time, simulation, label=f'γ={gamma}')

plt.xlabel("Time Periods")
plt.ylabel("State Values")
plt.title("Simulated Markov Chains for Different γ Values")
plt.legend()
plt.show()
