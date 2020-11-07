import random
import string
import sys

class Individual:
    """
        Individual in the population
    """

    def __init__(self, size):
        self.fitness = 0
        self.genes = self.generate_random_genes(size)

    # Fitness function: returns a floating points of "correct" characters
    def calc_fitness(self, target):
        score = 0
        for x in range(len(self.genes)):
            if(ord(self.genes[x])==ord(target[x])):
                score+=1
        # insert your code to calculate the individual fitness here

        self.fitness = score/len(target)
    def mini_fitness(self,input,target):
        score = 0
        for x in range(len(target)):
            if (ord(input[x]) == ord(target[x])):
                score += 1
        return score
    def __repr__(self):
        return ''.join(self.genes) + " -> fitness: " + str(self.fitness)

    @staticmethod
    def generate_random_genes(size):
        genes = []

        for i in range(size):
            genes.append(random.choice(string.ascii_letters+" "+"."))

        return genes

    # The crossover function selects pairs of individuals to be mated, generating a third individual (child)
    def crossover(self, partner):
        # Crossover suggestion: child with half genes from one parent and half from the other parent
        ind_len = len(self.genes)
        first_half=ind_len//2
        child = Individual(ind_len)
        choice=random.randint(0,1)
        first=""
        second=""
        if choice:
            first=self.genes[0:first_half]
            second=partner.genes[first_half:]
        else:
            first=partner.genes[0:first_half]
            second=self.genes[first_half:]
        new_child=first+second
        # print(new_child,type(new_child))
        child.genes=new_child

        return child
    def crossover2(self,partner,target):
        ind_len = len(self.genes)
        first_half = ind_len // 2
        child = Individual(ind_len)
        first=''
        second=''
        fh_self=self.genes[0:first_half]
        fh_partner=partner.genes[0:first_half]
        sh_self=self.genes[first_half:]
        sh_partner=partner.genes[first_half:]
        fh_target=target[0:first_half]
        sh_target=target[first_half:]
        self_partner=self.mini_fitness(fh_self,fh_target)+self.mini_fitness(sh_partner,sh_target)
        partner_self=self.mini_fitness(fh_partner,fh_target)+self.mini_fitness(sh_self,sh_target)
        if self_partner>partner_self:
            first = self.genes[0:first_half]
            second = partner.genes[first_half:]
        elif partner_self>self_partner:
            first = partner.genes[0:first_half]
            second = self.genes[first_half:]
        else:
            choice = random.randint(0, 1)
            if choice:
                first = self.genes[0:first_half]
                second = partner.genes[first_half:]
            else:
                first = partner.genes[0:first_half]
                second = self.genes[first_half:]
        new_child = first + second
        # print(new_child,type(new_child))
        child.genes = new_child
        return child

    # Mutation: based on a mutation probability, the function picks a new random character and replace a gene with it
    def mutate(self, mutation_rate):
        # code to mutate the individual here
        choices=[1,0]
        rates=[mutation_rate,1-mutation_rate]
        choice=random.choices(choices,rates,k=1)[0]
        # print(self.genes)
        if choice:
            pos=int(random.uniform(0,len(self.genes)))
            self.genes[pos]=random.choice(string.ascii_letters+" "+".")
            # print(self.genes)

    def mutate2(self,mutation_rate,target):
        choices = [1, 0]
        rates = [mutation_rate, 1 - mutation_rate]
        choice = random.choices(choices, rates, k=1)[0]
        # print(self.genes)
        if choice:
            idx=[]
            for x in range(len(target)):
                if (ord(self.genes[x]) != ord(target[x])):
                   idx.append(x)
            mut_idx=random.choice(idx)
            self.genes[mut_idx] = random.choice(string.ascii_letters + " "+".")





