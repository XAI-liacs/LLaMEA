import numpy as np

class HEA_ALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.mutation_rate = 0.1
        self.learning_rate = 0.05
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.pop_size):
            solution = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'solution': solution, 'value': float('inf')})
        return population

    def mutate(self, solution, lb, ub):
        mutation = self.mutation_rate * np.random.randn(self.dim)
        new_solution = solution + mutation
        new_solution = np.clip(new_solution, lb, ub)
        return new_solution

    def adapt_learning_rate(self, success_rate):
        if success_rate > 0.2:
            self.learning_rate *= 1.2
        else:
            self.learning_rate *= 0.9
        self.learning_rate = np.clip(self.learning_rate, 0.01, 0.1)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)
        
        success_count = 0
        iteration_count = 0
        
        while evaluations < self.budget:
            new_population = []
            for individual in population:
                candidate_solution = individual['solution'] + self.learning_rate * np.random.randn(self.dim)
                candidate_solution = np.clip(candidate_solution, lb, ub)
                
                new_value = func(candidate_solution)
                evaluations += 1
                
                if new_value < individual['value']:
                    new_population.append({'solution': candidate_solution, 'value': new_value})
                    success_count += 1
                else:
                    new_population.append(individual)
                
                if new_value < self.best_value:
                    self.best_value = new_value
                    self.best_solution = candidate_solution.copy()

                if evaluations >= self.budget:
                    break
            
            iteration_count += 1
            if iteration_count % 5 == 0:
                self.adapt_learning_rate(success_count / self.pop_size)
                success_count = 0

            population = new_population

        return self.best_solution, self.best_value