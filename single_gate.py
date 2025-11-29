import numpy as np
import matplotlib.pyplot as plt

def normalize(state):
    return state / np.linalg.norm(state)

# 1. Setup: Define the "Multi-Truth" Space (3 Qubits = 8 parallel realities)
num_qubits = 3
num_states = 2**num_qubits

# Create a superposition of ALL possible inputs (000 to 111)
# This represents asking "Every possible question" simultaneously.
# Initially, all amplitudes are equal (1/sqrt(N))
state_vector = np.ones(num_states) / np.sqrt(num_states)

print("--- Initial Superposition (The 'Multi-Truth' Input) ---")
print(np.round(state_vector, 3))

# 2. Define a "Logic Oracle" (The Unitary Gate)
# Let's say the "Truth" we are looking for is the state |101> (Index 5)
# In classical logic, you'd have to check row-by-row.
# Here, we build a diagonal matrix that flips the phase of ONLY the target.
# This happens in ONE operation.

target_index = 5  # |101>
Oracle = np.eye(num_states) # Identity matrix
Oracle[target_index, target_index] = -1 # Phase Flip (The "Mark")

# 3. Apply the Gate (The "One Shot" Update)
# This matrix multiplication updates the entire truth table instantly.
processed_state = np.dot(Oracle, state_vector)

print("\n--- After Logic Gate (Hidden Phase Information) ---")
print("Amplitudes:", np.round(processed_state, 3))
print("Note: The magnitude is the same, but index 5 has a negative phase.")

# 4. The ARP Connection: Adaptive Interference Steering
# Classically, if we measure now, we still just get a random result.
# We need to "steer" the wavefunction using an interference operator (Diffusion).

# The Diffusion Operator (Inversion about the mean)
# This converts "Phase differences" into "Amplitude differences"
Diffusion = 2 * np.outer(state_vector, state_vector) - np.eye(num_states)

# Apply Diffusion
final_state = np.dot(Diffusion, processed_state)

print("\n--- After Interference Steering (The 'Selection') ---")
print("Amplitudes:", np.round(final_state, 3))
print("Probabilities:", np.round(np.abs(final_state)**2, 3))

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(range(num_states), np.abs(final_state)**2, color='cyan', edgecolor='blue')
plt.title(f"Quantum Logic Selection: Target |{target_index:03b}>")
plt.xlabel("Truth Table Row (State Index)")
plt.ylabel("Probability (Truth Magnitude)")
plt.axhline(y=0, color='k')
plt.show()
