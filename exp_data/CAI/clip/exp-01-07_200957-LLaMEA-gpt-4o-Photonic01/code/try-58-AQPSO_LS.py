import numpy as np

class AQPSO_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.particles = np.random.rand(self.population_size, self.dim)
        self.best_positions = np.copy(self.particles)
        self.best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.inertia_weight = 0.9
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.quantum_radius = 0.1
        self.local_search_prob = 0.1

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        eval_count = 0
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(np.clip(self.particles[i], lb, ub))
                eval_count += 1
                
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.particles[i]
                    
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            if eval_count >= self.budget:
                break

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.cognitive_weight * r1 * (self.best_positions[i] - self.particles[i])
                social_component = self.social_weight * r2 * (self.global_best_position - self.particles[i])
                self.particles[i] += self.inertia_weight * (cognitive_component + social_component)

                if np.random.rand() < self.local_search_prob:
                    local_search_step = np.random.normal(0, self.quantum_radius, self.dim)
                    local_candidate = np.clip(self.particles[i] + local_search_step, lb, ub)
                    local_score = func(local_candidate)
                    eval_count += 1
                    if local_score < self.best_scores[i]:
                        self.best_scores[i] = local_score
                        self.best_positions[i] = local_candidate
                        if local_score < self.global_best_score:
                            self.global_best_score = local_score
                            self.global_best_position = local_candidate

            self.inertia_weight = 0.5 + 0.4 * (1 - eval_count / self.budget)
            self.cognitive_weight = 1.2 + 0.8 * (1 - eval_count / self.budget)
            self.social_weight = 1.2 + 0.8 * (eval_count / self.budget)
            self.quantum_radius *= (0.95 if eval_count / self.budget > 0.5 else 1.05)
            self.local_search_prob = 0.1 + 0.2 * (1 - eval_count / self.budget)

        return self.global_best_position, self.global_best_score