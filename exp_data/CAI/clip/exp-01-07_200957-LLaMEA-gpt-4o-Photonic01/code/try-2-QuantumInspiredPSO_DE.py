import numpy as np

class QuantumInspiredPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        eval_count = 0
        generation = 0
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(np.clip(self.particles[i], lb, ub))
                eval_count += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                    
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
            
            if eval_count >= self.budget:
                break

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.cognitive_weight * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.social_weight * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component)
                
                # Quantum-inspired position update
                phi = np.random.uniform(0, 2 * np.pi, self.dim)
                distance = np.abs(self.particles[i] - self.global_best_position)
                self.particles[i] += np.sin(phi) * distance / np.log(1 + generation + 1)
                self.particles[i] = np.clip(self.particles[i], lb, ub)
                
                if np.random.rand() < self.crossover_rate:
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.particles[i])
                    trial_score = func(trial)
                    eval_count += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial
                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial
            
            # Adaptive parameter adjustment
            self.inertia_weight = 0.5 + 0.3 * np.cos(np.pi * eval_count / self.budget)
            self.cognitive_weight = 1.2 + 0.8 * (eval_count / self.budget)
            self.social_weight = 1.2 + 0.8 * (1 - eval_count / self.budget)
            self.mutation_factor = 0.4 + 0.1 * np.cos(np.pi * generation / 10)
            generation += 1

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_score