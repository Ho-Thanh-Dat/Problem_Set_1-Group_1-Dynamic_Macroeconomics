import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=4)


def rouwenhorst(rho, sigma_u, n):
    """
    Discretizes an AR(1) process using Rouwenhorst's method.

    Args:
        rho: Persistence parameter (AR(1) coefficient).
        sigma_u: Standard deviation of the innovation term.
        n: Number of states in the Markov chain.

    Returns:
        A tuple containing:
            - transition_matrix: The n x n transition matrix.
            - state_vector: The n-dimensional state vector.
    """

    # 1. Construct the two-state transition matrix for the base case (n=2).
    p = (1 + rho) / 2
    Pi_2 = np.array([[p, 1 - p],
                     [1 - p, p]])

    # 2. Calculate the standard deviation of the unconditional distribution.
    sigma_y = np.sqrt(sigma_u**2 / (1 - rho**2))

    # 3. Construct the state vector for n=2.  The spacing is crucial.
    state_vector_2 = np.array([-sigma_y * np.sqrt(n - 1), sigma_y * np.sqrt(n - 1)])

    # 4. Iterate to construct the transition matrix and state vector for higher n.
    if n > 2:
        for i in range(3, n + 1):
            # Pad the previous transition matrix with zeros.
            Pi_prev = np.zeros((i, i))
            #Fix: Ensure Pi_2 is the correct size to add to Pi_prev (i-1,i-1)
            Pi_prev[:i-1, :i-1] += p * Pi_2
            Pi_prev[:i-1, 1:] += (1-p) * Pi_2
            Pi_prev[1:, :i-1] += (1-p) * Pi_2
            Pi_prev[1:, 1:] += p * Pi_2

            Pi_2 = Pi_prev / (1 + rho) #Correct to /2 before.  Incorrectly scaled probabilities.
            Pi_2 = Pi_2 / np.sum(Pi_2, axis=1, keepdims=True) #renormalize due to numerical error

            # State vector is evenly spaced, maintaining the unconditional std. dev.
            state_vector_i = np.linspace(-sigma_y * np.sqrt(n - 1), sigma_y * np.sqrt(n - 1), i)

    #State vector
    state_vector = np.linspace(-sigma_y * np.sqrt(n-1), sigma_y*np.sqrt(n-1),n)

    return Pi_2, state_vector




def simulate_markov_chain(transition_matrix, state_vector, n_periods, initial_state_index=None):
    """
    Simulates a Markov chain given a transition matrix, state vector, and number of periods.

    Args:
        transition_matrix: The transition matrix of the Markov chain.
        state_vector: The state vector.
        n_periods: The number of periods to simulate.
        initial_state_index: (Optional) The index of the initial state.  If None, a
                            state is drawn from a uniform distribution.

    Returns:
        A numpy array representing the simulated state indices over time.
    """

    n_states = transition_matrix.shape[0]
    states = np.zeros(n_periods, dtype=int)

    # Determine initial state
    if initial_state_index is None:
        # Draw from a uniform distribution
        states[0] = np.random.choice(n_states)
    else:
        states[0] = initial_state_index

    # Simulate the chain
    for t in range(1, n_periods):
        previous_state = states[t - 1]
        probabilities = transition_matrix[previous_state, :]
        states[t] = np.random.choice(n_states, p=probabilities)

    return states

    # (b) Discretize the AR(1) process with gamma1 = 0.85 and n = 7.

gamma1 = 0.85
sigma_epsilon = 1  # Standard deviation of the innovation term
n_states = 7

transition_matrix, state_vector = rouwenhorst(gamma1, sigma_epsilon, n_states)

print("Transition Matrix:")
print(transition_matrix)
print("\nState Vector:")
print(state_vector)

# (c) Simulate the Markov Chain for 50 periods. (Moved from separate section)
np.random.seed(2025)
n_periods = 50
initial_state_index = None #draw from uniform

simulated_states = simulate_markov_chain(transition_matrix, state_vector, n_periods, initial_state_index)
simulated_values = state_vector[simulated_states]
print(simulated_values)

plt.figure(figsize=(10, 6))
plt.plot(simulated_values)
plt.title(f"Markov Chain Simulation (γ1 = {gamma1})")
plt.xlabel("Period")
plt.ylabel("State Value")
plt.show()


# (d) Repeat for different gamma1 values and plot together.(Moved from separate section)

gamma1_values = [0.75, 0.85, 0.95, 0.99]
plt.figure(figsize=(12, 8))
for gamma1 in gamma1_values:
  transition_matrix, state_vector = rouwenhorst(gamma1, sigma_epsilon, n_states)
  simulated_states = simulate_markov_chain(transition_matrix, state_vector, n_periods, initial_state_index)
  simulated_values = state_vector[simulated_states]
  plt.plot(simulated_values, label=f'γ1 = {gamma1}')

plt.title("Markov Chain Simulations for Different γ1 Values")
plt.xlabel("Period")
plt.ylabel("State Value")
plt.legend()
plt.show()