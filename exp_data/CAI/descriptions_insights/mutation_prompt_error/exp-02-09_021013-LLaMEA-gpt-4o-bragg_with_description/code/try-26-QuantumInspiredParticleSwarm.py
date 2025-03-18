import numpy as np

class QuantumInspiredParticleSwarm:
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
        quantum_prob = 0.2  # Adjusted for better exploration

        # Initialize particle positions and velocities
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        
        # Initialize personal best and global best
        personal_best_positions = particles.copy()
        personal_best_values = np.full(num_particles, float('-inf'))
        
        global_best_position = None
        global_best_value = float('-inf')

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

            # Update particle velocities and positions
            phi_p = 0.5 + evaluations / (2 * self.budget)  # Dynamic update of phi_p
            for i in range(num_particles):
                # Quantum-inspired perturbation
                if np.random.rand() < quantum_prob:
                    velocities[i] = (np.random.uniform(-1, 1, self.dim) 
                                     * (particles[i] - global_best_position))
                else:
                    r_p = np.random.rand(self.dim)
                    r_g = np.random.rand(self.dim)
                    velocities[i] = (omega * velocities[i] +
                                     phi_p * r_p * (personal_best_positions[i] - particles[i]) +
                                     phi_g * r_g * (global_best_position - particles[i]))

                # Differential Evolution component
                if np.random.rand() < 0.1:  # Probability for applying DE
                    a, b, c = np.random.choice(num_particles, 3, replace=False)
                    mutant_vector = particles[a] + 0.8 * (particles[b] - particles[c])
                    mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)
                    if func(mutant_vector) > func(particles[i]):
                        particles[i] = mutant_vector

                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], func.bounds.lb, func.bounds.ub)

            # Ensure not exceeding budget
            if evaluations >= self.budget:
                break

        return global_best_position, global_best_value