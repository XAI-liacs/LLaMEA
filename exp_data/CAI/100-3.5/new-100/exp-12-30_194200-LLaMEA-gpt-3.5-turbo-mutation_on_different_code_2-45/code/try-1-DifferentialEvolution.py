class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.7, CR=0.9, alpha=0.1):  # Added alpha for weighted average
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.alpha = alpha
        self.f_opt = np.Inf
        self.x_opt = None
    
    def mutate(self, population, target_idx):
        idxs = list(range(len(population)))
        idxs.remove(target_idx)
        mutant_vector = np.mean(population, axis=0) + self.alpha * (population[target_idx] - np.mean(population, axis=0))  # Weighted average
        return mutant_vector