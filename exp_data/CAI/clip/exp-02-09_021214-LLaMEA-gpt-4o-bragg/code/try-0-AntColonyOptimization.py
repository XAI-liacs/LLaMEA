import numpy as np

class AntColonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pheromones = None
        self.alpha = 1.0  # pheromone importance
        self.beta = 2.0   # heuristic importance
        self.evaporation_rate = 0.5
        self.num_ants = 10
        self.best_solution = None
        self.best_obj = float('inf')
    
    def initialize_pheromones(self, bounds):
        self.pheromones = np.ones((self.num_ants, self.dim))
        self.lb, self.ub = bounds.lb, bounds.ub
    
    def update_pheromones(self, ants_positions, ants_obj):
        for i in range(self.num_ants):
            contribution = 1.0 / (1.0 + ants_obj[i])  # better solutions contribute more pheromone
            self.pheromones[i] = (1 - self.evaporation_rate) * self.pheromones[i] + contribution

    def construct_solution(self):
        solution = []
        for d in range(self.dim):
            probabilities = (self.pheromones[:, d] ** self.alpha) * ((1.0 / (1.0 + np.abs(np.random.uniform(-1, 1, self.num_ants)))) ** self.beta)
            probabilities /= np.sum(probabilities)
            chosen_ant = np.random.choice(range(self.num_ants), p=probabilities)
            step = np.random.uniform(self.lb[d], self.ub[d]) * self.pheromones[chosen_ant, d]
            solution.append(np.clip(step, self.lb[d], self.ub[d]))
        return np.array(solution)
    
    def local_search(self, solution):
        # Simple local perturbation: random small change
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        new_solution = solution + perturbation
        return np.clip(new_solution, self.lb, self.ub)
    
    def __call__(self, func):
        self.initialize_pheromones(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            ants_positions = [self.construct_solution() for _ in range(self.num_ants)]
            ants_positions = [self.local_search(sol) for sol in ants_positions]
            ants_obj = [func(pos) for pos in ants_positions]
            evaluations += self.num_ants
            
            self.update_pheromones(ants_positions, ants_obj)
            
            min_idx = np.argmin(ants_obj)
            if ants_obj[min_idx] < self.best_obj:
                self.best_obj = ants_obj[min_idx]
                self.best_solution = ants_positions[min_idx]
        
        return self.best_solution