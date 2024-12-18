class AcceleratedDynamicBatAlgorithm(AdaptiveDynamicBatAlgorithm):
    def __call__(self, func):
        improvement_rates = []
        step_size = 0.1
        for _ in range(self.budget):
            self.pulse_rate = 0.9 / (1 + np.exp(-1 * ((_ - self.budget) / self.budget * 10 - 5)))
            frequencies = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand(self.population_size)
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    self.velocities[i] += (self.population[i] - self.best_solution) * frequencies[i] * self.adaptive_mutation_factor
                self.population[i] += self.velocities[i] * step_size
                fitness = func(self.population[i])
                improvement_rate = (self.best_fitness - fitness) / self.best_fitness
                improvement_rates.append(improvement_rate)
                if fitness < self.best_fitness:
                    self.best_solution = self.population[i]
                    self.best_fitness = fitness
            if len(improvement_rates) > 10:
                current_improvement_rate = np.mean(improvement_rates[-10:])
                if current_improvement_rate > 0.6:
                    self.adaptive_mutation_factor *= 1.2
                    step_size = min(step_size * 1.1, 0.5)  # Reactive step size adaptation
                elif current_improvement_rate < 0.4:
                    self.adaptive_mutation_factor *= 0.8
                    step_size = max(step_size * 0.9, 0.05)  # Reactive step size adaptation
        return self.best_solution