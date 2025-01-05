import numpy as np

class HybridQuantumAnnealingWithAdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.positions = None
        self.best_position = None
        self.best_score = float('inf')
        self.annealing_rate = 0.99
        self.simplex_size = 5
    
    def initialize_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.simplex_size, self.dim))
    
    def quantum_annealing_step(self, individual, temperature):
        quantum_flip = np.random.normal(0, temperature, self.dim)
        return individual + quantum_flip
    
    def adaptive_nelder_mead(self, func, evaluations):
        alpha, gamma, rho, sigma = 1, 2, 0.5, 0.5
        lb, ub = func.bounds.lb, func.bounds.ub
        
        scores = np.array([func(pos) for pos in self.positions])
        evaluations += self.simplex_size
        
        while evaluations < self.budget:
            # Order according to scores
            idx = np.argsort(scores)
            self.positions = self.positions[idx]
            scores = scores[idx]
            
            centroid = np.mean(self.positions[:-1], axis=0)
            # Reflection
            reflection = centroid + alpha * (centroid - self.positions[-1])
            reflection = np.clip(reflection, lb, ub)
            reflection_score = func(reflection)
            evaluations += 1
            
            if reflection_score < scores[0]:
                # Expansion
                expansion = centroid + gamma * (reflection - centroid)
                expansion = np.clip(expansion, lb, ub)
                expansion_score = func(expansion)
                evaluations += 1
                
                if expansion_score < reflection_score:
                    self.positions[-1] = expansion
                    scores[-1] = expansion_score
                else:
                    self.positions[-1] = reflection
                    scores[-1] = reflection_score
            elif reflection_score < scores[-2]:
                self.positions[-1] = reflection
                scores[-1] = reflection_score
            else:
                # Contraction
                contraction = centroid + rho * (self.positions[-1] - centroid)
                contraction = np.clip(contraction, lb, ub)
                contraction_score = func(contraction)
                evaluations += 1
                
                if contraction_score < scores[-1]:
                    self.positions[-1] = contraction
                    scores[-1] = contraction_score
                else:
                    # Shrink
                    for j in range(1, self.simplex_size):
                        self.positions[j] = self.positions[0] + sigma * (self.positions[j] - self.positions[0])
                        self.positions[j] = np.clip(self.positions[j], lb, ub)
                        scores[j] = func(self.positions[j])
                    evaluations += self.simplex_size - 1
            
            if scores[0] < self.best_score:
                self.best_score = scores[0]
                self.best_position = self.positions[0]
            
            # Break if meeting the budget
            if evaluations >= self.budget:
                break
        
        return evaluations
    
    def __call__(self, func):
        self.initialize_positions(func.bounds)
        evaluations = 0
        temperature = 1.0
        
        while evaluations < self.budget:
            for i in range(self.simplex_size):
                if evaluations >= self.budget:
                    break
                
                new_position = self.quantum_annealing_step(self.positions[i], temperature)
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                new_score = func(new_position)
                evaluations += 1
                
                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_position = new_position
                
                if new_score < func(self.positions[i]):
                    self.positions[i] = new_position
            
            evaluations = self.adaptive_nelder_mead(func, evaluations)
            temperature *= self.annealing_rate