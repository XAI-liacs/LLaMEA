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
        omega_start = 0.9  # Adaptive inertia weight start
        omega_end = 0.4  # Adaptive inertia weight end
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
            # Calculate adaptive inertia weight
            omega = omega_end + (omega_start - omega_end) * ((self.budget - evaluations) / self.budget)

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

                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], func.bounds.lb, func.bounds.ub)

            # Ensure not exceeding budget
            if evaluations >= self.budget:
                break

        return global_best_position, global_best_value