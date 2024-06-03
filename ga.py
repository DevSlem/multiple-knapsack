import random
import time
import argparse
from util import load_knapsack_problem, save_results

class GeneticAlgorithm:
    def __init__(self, num_items, num_knapsacks, capacities, values, weights, population_size=200, crossover_rate=0.7, mutation_rate=0.05, selection_pressure=0.5, elite_size=5):
        self.num_items = num_items
        self.num_knapsacks = num_knapsacks
        self.capacities = capacities
        self.values = values
        self.weights = weights
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.elite_size = elite_size
        self.population = [self.random_solution() for _ in range(population_size)]

    def random_solution(self):
        return [random.randint(-1, self.num_knapsacks - 1) for _ in range(self.num_items)]

    def calculate_fitness(self, solution):
        total_value = 0
        knapsack_weights = [0] * self.num_knapsacks
        for item_index, knapsack_index in enumerate(solution):
            if knapsack_index == -1:
                continue  # Item is not in any knapsack
            if knapsack_weights[knapsack_index] + self.weights[item_index] <= self.capacities[knapsack_index]:
                knapsack_weights[knapsack_index] += self.weights[item_index]
                total_value += self.values[item_index]
        return total_value

    def selection(self):
        sorted_population = sorted(self.population, key=self.calculate_fitness, reverse=True)
        self.population = sorted_population[:int(self.selection_pressure * self.population_size)]

    def multi_point_crossover(self, parent1, parent2, num_points=2):
        points = sorted(random.sample(range(1, self.num_items), num_points))
        child1, child2 = parent1[:], parent2[:]
        for i in range(len(points)):
            if i % 2 == 0:
                start = points[i]
                end = points[i + 1] if i + 1 < len(points) else self.num_items
                child1[start:end], child2[start:end] = child2[start:end], child1[start:end]
        return child1, child2

    def mutate(self, solution):
        for i in range(self.num_items):
            if random.random() < self.mutation_rate:
                solution[i] = random.randint(-1, self.num_knapsacks - 1)
        return solution

    def generate_new_population(self):
        new_population = self.population[:self.elite_size]  # Keep elite solutions
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(self.population, 2)
            child1, child2 = self.multi_point_crossover(parent1, parent2)
            new_population.extend([self.mutate(child1), self.mutate(child2)])
        self.population = new_population

    def run(self, generations=5000):
        for generation in range(generations):
            self.selection()
            self.generate_new_population()
            best_solution = max(self.population, key=self.calculate_fitness)
            
            if generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {self.calculate_fitness(best_solution)}")
        return self.calculate_fitness(best_solution), best_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", type=str)
    parser.add_argument("--generations", type=int, default=5000)
    args = parser.parse_args()

    problem_name = args.problem_name
    knapsack_df, item_df = load_knapsack_problem(problem_name)
    capacities = knapsack_df['capacity'].values
    values = item_df['value'].values
    weights = item_df['weight'].values

    start = time.time()
    ga = GeneticAlgorithm(len(values), len(capacities), capacities, values, weights)
    fit, gen = ga.run(generations=args.generations)
    train_time = time.time() - start

    # Save results using save_results function
    result_df = save_results(
        problem_name=problem_name,
        method="Genetic Algorithm",
        total_value=int(fit),
        train_time=train_time,
        inference_time=train_time,
    )

    print(result_df)