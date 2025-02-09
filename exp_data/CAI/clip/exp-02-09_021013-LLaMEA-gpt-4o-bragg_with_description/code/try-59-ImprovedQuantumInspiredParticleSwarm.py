import numpy as np

class ImprovedQuantumInspiredParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        num_particles = 30
        evaluations = 0
        
        # PSO Hyperparameters
        omega = 0.5  # Inertia weight
        phi_p = 0.5  # Personal influence
        phi_g = 0.5  # Global influence
        quantum_prob_init = 0.25  # Initial quantum probability modified
        quantum_prob_final = 0.1  # Final quantum probability modified

        # Initialize particle positions and velocities
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        
        # Initialize personal best and global best
        personal_best_positions = particles.copy()
        personal_best_values = np.full(num_particles, float('-inf'))
        
        global_best_position = None
        global_best_value = float('-inf')
        
        # Initialize elite memory
        elite_memory_size = 5 + int(0.1 * self.dim)  # Dynamic elite memory size
        elite_memory_positions = np.zeros((elite_memory_size, self.dim))
        elite_memory_values = np.full(elite_memory_size, float('-inf'))

        while evaluations < self.budget:
            for i in range(num_particles):
                # Evaluate current particle
                value = func(particles[i])
                evaluations += 1

                # Update personal best
                if value > personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i]

                # Update global best
                if value > global_best_value:
                    global_best_value = value
                    global_best_position = particles[i]

                # Update elite memory
                if value > np.min(elite_memory_values):
                    min_elite_index = np.argmin(elite_memory_values)
                    elite_memory_values[min_elite_index] = value
                    elite_memory_positions[min_elite_index] = particles[i]

            # Calculate adaptive quantum probability
            quantum_prob = quantum_prob_init + (quantum_prob_final - quantum_prob_init) * (evaluations / self.budget) * 1.2  # Adjusted adaptive quantum influence

            # Update particle velocities and positions
            for i in range(num_particles):
                # Quantum-inspired perturbation with adaptive probability
                if np.random.rand() < quantum_prob:
                    elite_indices = np.random.choice(range(elite_memory_size), size=4, replace=False)  # Enhanced sampling
                    random_elite_member = elite_memory_positions[np.random.choice(elite_indices)]  # Random elite selection
                    velocities[i] = np.random.uniform(-1.5, 1.5, self.dim)  # Diversified quantum perturbation
                else:
                    r_p = np.random.rand(self.dim)
                    r_g = np.random.rand(self.dim)
                    omega = 0.5 - 0.3 * (evaluations / self.budget)  # Inertia weight decay
                    velocities[i] = (omega * velocities[i] +
                                     phi_p * r_p * (personal_best_positions[i] - particles[i]) +
                                     phi_g * r_g * (global_best_position - particles[i]))

                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], func.bounds.lb, func.bounds.ub)

            # Ensure not exceeding budget
            if evaluations >= self.budget:
                break

        return global_best_position, global_best_value