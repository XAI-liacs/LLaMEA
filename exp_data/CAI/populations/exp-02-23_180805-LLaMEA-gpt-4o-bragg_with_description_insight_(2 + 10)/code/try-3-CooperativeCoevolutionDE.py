import numpy as np
from scipy.optimize import minimize

class CooperativeCoevolutionDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Split problem into subcomponents for coevolution
        num_subcomponents = 5
        sub_dim = self.dim // num_subcomponents
        sub_bounds = [(lb, ub)] * sub_dim
        
        subpopulations = [lb + (ub - lb) * np.random.rand(population_size, sub_dim) for _ in range(num_subcomponents)]
        best_subsolutions = [subpop[np.argmin([func(self._assemble_solution(subpop, idx)) for subpop in subpopulations])] for idx, subpop in enumerate(subpopulations)]
        eval_count = population_size * num_subcomponents

        def subcomponent_func(subsolution, index):
            solution = self._assemble_solution([subsolution if i == index else best_subsolutions[i] for i in range(num_subcomponents)], index)
            return func(np.clip(solution, lb, ub))

        while eval_count < self.budget:
            for idx in range(num_subcomponents):
                for i in range(population_size):
                    if eval_count >= self.budget:
                        break

                    # Mutation
                    indices = np.random.choice(range(population_size), 3, replace=False)
                    a, b, c = subpopulations[idx][indices]
                    mutant = np.clip(a + F * (b - c), lb, ub)
                    
                    # Crossover
                    cross_points = np.random.rand(sub_dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, sub_dim)] = True
                    trial = np.where(cross_points, mutant, subpopulations[idx][i])
                    
                    # Calculate fitness
                    f_trial = subcomponent_func(trial, idx)
                    eval_count += 1

                    # Selection
                    if f_trial < subcomponent_func(subpopulations[idx][i], idx):
                        subpopulations[idx][i] = trial
                        if f_trial < subcomponent_func(best_subsolutions[idx], idx):
                            best_subsolutions[idx] = trial

                # Local refinement using periodic embedding
                if eval_count + sub_dim <= self.budget:
                    res = minimize(lambda x: subcomponent_func(np.clip(x, lb, ub), idx), best_subsolutions[idx], method='L-BFGS-B', bounds=sub_bounds)
                    eval_count += res.nfev
                    if res.fun < subcomponent_func(best_subsolutions[idx], idx):
                        best_subsolutions[idx] = res.x

        return self._assemble_solution(best_subsolutions)

    def _assemble_solution(self, subcomponents, focus_idx=None):
        # Reconstruct the full-dimensional solution from subcomponents
        solution = np.zeros(self.dim)
        for idx, subcomponent in enumerate(subcomponents):
            start = idx * len(subcomponent)
            end = start + len(subcomponent)
            solution[start:end] = subcomponent
        return solution

# Example usage:
# func = YourBlackBoxFunction()
# optimizer = CooperativeCoevolutionDE(budget=1000, dim=10)
# best_solution = optimizer(func)