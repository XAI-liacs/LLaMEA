import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.population_size = self.initial_population_size
        self.min_population_size = 5
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.995
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                progress = evaluations / self.budget
                self.inertia_weight = (self.inertia_weight_initial - self.inertia_weight_final) * (1 - progress ** 2) + self.inertia_weight_final
                self.cognitive_weight = 1.5 + (0.75 * progress)
                self.social_weight = 1.5 - (0.5 * progress)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.cognitive_weight * np.random.rand() * (personal_best_positions[i] - positions[i]) +
                               self.social_weight * np.random.rand() * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocity[i], lb, ub)
                
                current_score = func(positions[i])
                evaluations += 1
                
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]
                    
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i]
            
            if evaluations >= self.budget / 2 and self.population_size > self.min_population_size:
                self.population_size -= 1  # Adaptive population size reduction
            
            best_candidate = global_best_position + np.random.normal(0, 1, self.dim)
            best_candidate = np.clip(best_candidate, lb, ub)
            best_candidate_score = func(best_candidate)
            evaluations += 1
            
            if best_candidate_score < global_best_score or np.random.rand() < np.exp((global_best_score - best_candidate_score) / self.temperature):
                global_best_position = best_candidate
                global_best_score = best_candidate_score
            
            self.temperature *= self.cooling_rate * (1 - evaluations / (2 * self.budget))
        
        return global_best_position