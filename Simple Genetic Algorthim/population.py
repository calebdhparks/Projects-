from individual import Individual
import random


class Population:
    """
        A class that describes a population of virtual individuals
    """

    def __init__(self, target, size, mutation_rate):
        self.population = []
        self.generations = 0
        self.target = target
        self.mutation_rate = mutation_rate
        self.best_ind = None
        self.finished = False
        self.perfect_score = 1.0
        self.max_fitness = 0.0
        self.average_fitness = 0.0
        self.mating_pool = []

        for i in range(size):
            ind = Individual(len(target))
            # print(ind.genes)
            ind.calc_fitness(target)

            if ind.fitness > self.max_fitness:
                self.max_fitness = ind.fitness

            self.average_fitness += ind.fitness
            self.population.append(ind)
        # sorted_pop=sorted(self.population,key=lambda ind: ind.fitness)
        self.average_fitness /= size

    def print_population_status(self):
        print("\nPopulation " + str(self.generations))
        print("Average fitness: " + str(self.average_fitness))
        print("Best individual: " + str(self.best_ind))
        # print(self.population[0].crossover(self.population[1]))
        # self.population[0].mutate(self.mutation_rate)

    # Generate a mating pool according to the probability of each individual
    def natural_selection(self):
        # Implementation suggestion based on Lab 3:
        # Based on fitness, each member will get added to the mating pool a certain number of times
        # a higher fitness = more entries to mating pool = more likely to be picked as a parent
        # a lower fitness = fewer entries to mating pool = less likely to be picked as a parent

        probs=[]
        for x in range(len(self.population)):
            probs.append(self.population[x].fitness)
        self.mating_pool=random.choices(self.population,probs,k=len(self.population)*2)
        # if self.generations>2000:
        #     print(self.mating_pool)

    # Generate the new population based on the natural selection function
    def generate_new_population(self):
        index=0
        for x in range(0,len(self.mating_pool),2):
            # new_child=self.mating_pool[x].crossover(self.mating_pool[x+1])
            new_child=self.mating_pool[x].crossover2(self.mating_pool[x+1],self.target)
            # new_child.mutate(self.mutation_rate)
            new_child.mutate2(self.mutation_rate,self.target)
            self.population[index]=new_child
            index+=1
            # print(x,self.mating_pool[x].genes,self.mating_pool[x+1].genes)
    # Compute/Identify the current "most fit" individual within the population
    def evaluate(self):
        self.average_fitness=0
        for x in range(len(self.population)):
            self.population[x].calc_fitness(self.target)
            this_fitness=float(self.population[x].fitness)
            self.average_fitness+=this_fitness
            if self.population[x].fitness > self.max_fitness:
                self.max_fitness = self.population[x].fitness
                self.best_ind=self.population[x]
            if(this_fitness==self.perfect_score):
                self.finished=True
        self.average_fitness/=len(self.population)
        self.generations+=1

    def getGen(self):
        return [self.generations,self.average_fitness]

