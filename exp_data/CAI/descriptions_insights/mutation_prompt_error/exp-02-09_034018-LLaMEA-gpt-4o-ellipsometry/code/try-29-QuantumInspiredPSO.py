import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.inertia_weight = 0.7
        self.cognitive_const = 1.4
        self.social_const = 1.4
        self.quantum_gate = 0.5

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        
        evaluations = self.pop_size
        iteration = 0
        max_iterations = self.budget // self.pop_size

        while evaluations < self.budget:
            inertia_weight_dynamic = 0.9 - (iteration / max_iterations) * 0.5
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight_dynamic * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best[i] - population[i]) +
                                 self.social_const * r2 * (global_best - population[i]))
                
                # Quantum-inspired update
                theta = np.arctan2(velocities[i], population[i])
                quantum_velocity = self.quantum_gate * np.tan(theta)
                population[i] += quantum_velocity
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                # Evaluate
                particle_score = func(population[i])
                evaluations += 1
                if particle_score < personal_best_scores[i]:
                    personal_best[i] = population[i]
                    personal_best_scores[i] = particle_score
                    if particle_score < func(global_best):
                        global_best = population[i]

            iteration += 1
            self.quantum_gate = 0.5 * np.cos(iteration * np.pi / max_iterations)

        return global_best