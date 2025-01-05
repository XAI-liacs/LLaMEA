import numpy as np

class QuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.5
        self.c1 = 2.0
        self.c2 = 2.0
        self.quantum_alpha = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        lb, ub = np.array(lb), np.array(ub)

        particles = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum-inspired velocity update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))

                # Quantum-inspired position update
                delta = np.random.uniform(0, 1, self.dim)
                quantum_flip = np.random.rand(self.dim) < self.quantum_alpha
                quantum_positions = np.where(quantum_flip, 
                                             lb + (ub - lb) * np.sin(np.pi * delta), 
                                             particles[i] + velocities[i])

                # Particle position update and boundary handling
                particles[i] = np.clip(quantum_positions, lb, ub)

                # Evaluate fitness
                particle_score = func(particles[i])
                evaluations += 1

                # Update personal best
                if particle_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = particle_score

                # Update global best
                if particle_score < global_best_score:
                    global_best_position = particles[i]
                    global_best_score = particle_score

        return global_best_position