import random
import math
from collections import deque
from bisect import insort
import os
import time
import random

'''
PROBLEMA:
MINIMIZAR O CUSTO
RESTRIÇÃO: TODAS AS LINHAS PRECISAM ESTAR COBERTAS
'''

class Solution:
    def __init__(self, columns, cost) -> None:
        self.columns = columns
        self.cost = cost
        self.fitness = 0

class SCP:
    """
    Classe criada para representar o problema.
    """
    
    def __init__(self, quant_linhas, quant_colunas, colunas, linhas_cobertas, custos, colunas_que_cobrem_linha) -> None:
        self.quant_colunas = quant_colunas
        self.quant_linhas = quant_linhas
        self.colunas = colunas
        self.linhas_cobertas = linhas_cobertas
        self.custos = custos
        self.colunas_que_cobrem_linha = colunas_que_cobrem_linha

    def busca_vizinhanca_first(self, S, Z_S, P1, P2):
        # Passo 0
        linhas_cobertas = [len(self.colunas_que_cobrem_linha[k]) for k in range(1, self.quant_linhas+1)]
        S_prime = set(self.colunas) - S
        N_S = len(S)
        Q_S = max(self.custos[i-1] for i in S)
   
        wi = [0 for _ in range(self.quant_linhas+1)]
        i = 0

        for linhas in self.colunas_que_cobrem_linha:
            for colunas in linhas:
                if colunas in S:
                    wi[i] += 1
            i += 1
        
        d = 0
        D = math.ceil(P1 * N_S)
        E = math.ceil(P2 * Q_S)

        while S:
            # Passo 1
            k = random.choice(list(S))
            # Passo 2
            S.remove(k)
            S_prime = list(S_prime)
            insort(S_prime, k)
            S_prime = set(S_prime)
            
            Z_S -= round(self.custos[k-1], 2)
          
            cont = [0 for _ in range(len(linhas_cobertas)+1)]
            
            for i in range(len(linhas_cobertas)+1):
                if k in self.colunas_que_cobrem_linha[i]:
                    cont[i] += 1

            wi = [wi[i] - cont[i] for i in range(len(linhas_cobertas)+1)]
 
            d += 1
            if d == D:
                break
        
        while True:
            # Passo 3
            U = {i for i in range(self.quant_linhas+1) if wi[i] == 0}
            U = U - {0}

            if not U:
                break

            # Passo 4

            S_prime_E = {j for j in S_prime if self.custos[j-1] <= E}

            #wi significa QUANTAS COLUNAS COBRE A LINHA i
            #self.linhas_cobertas[i] significa quais linhas sao cobertas pela coluna i
            #S_prime_E === melhores colunas para serem colocadas na solução
            
            alpha = [[1 if wi[i] == 0 and i in self.linhas_cobertas[j-1] and j in S_prime_E else 0 for j in range(self.quant_colunas+1)] for i in range(self.quant_linhas+1)]

            v_j = [sum(alpha[i][j] for i in range(self.quant_linhas+1)) if j in S_prime_E else 0 for j in range(self.quant_colunas+1)]

            for j in S_prime_E:
                if v_j[j] != 0:
                    k = j
                    break

            # Passo 5
            S_prime.remove(k)
            S.add(k)

            Z_S += round(self.custos[k-1], 2)

            cont = [0 for _ in range(len(linhas_cobertas)+1)]

            for i in range(len(linhas_cobertas)+1):
                if k in self.colunas_que_cobrem_linha[i]:
                    cont[i] += 1
            wi = [wi[i] + cont[i] for i in range(len(linhas_cobertas)+1)]

        # Passo 6
        for k in reversed(list(S)):
            cont = [0 for _ in range(len(linhas_cobertas)+1)]
            for i in range(len(linhas_cobertas)+1):
                if k in self.colunas_que_cobrem_linha[i]:
                    cont[i] += 1


            flag = True
            for i in self.linhas_cobertas[k-1]:
                if wi[i] - cont[i] < 1:
                    flag = False

            if flag:
                S.remove(k)
                S_prime.add(k)

                Z_S -= round(self.custos[k-1], 2)

                cont = [0 for _ in range(len(linhas_cobertas)+1)]
                for i in range(len(linhas_cobertas)+1):
                    if k in self.colunas_que_cobrem_linha[i]:
                        cont[i] += 1
                wi = [wi[i] - cont[i] for i in range(len(linhas_cobertas)+1)]

        return S.copy(), round(Z_S, 2)
        

    def busca_vizinhanca(self, S, Z_S, P1, P2):
        # Passo 0
        len_lines = self.quant_linhas+1
        len_columns = self.quant_colunas+1

        S_prime = set(self.colunas) - S
        N_S = len(S)
        Q_S = max(self.custos[i-1] for i in S)
   
        wi = [sum(1 for col in self.colunas_que_cobrem_linha[i] if col in S) for i in range(len_lines)]
        
        d = 0
        D = math.ceil(P1 * N_S)
        E = math.ceil(P2 * Q_S)

        while S:
            # Passo 1
            k = random.choice(list(S))
            # Passo 2
            S.remove(k)
            S_prime = list(S_prime)
            insort(S_prime, k)
            S_prime = set(S_prime)

            Z_S -= round(self.custos[k-1], 2)
          
            cont = [0 for _ in range(len_lines)]
            
            for i in range(len_lines):
                if k in self.colunas_que_cobrem_linha[i]:
                    cont[i] += 1

            wi = [wi[i] - cont[i] for i in range(len_lines)]
 
            d += 1
            if d == D:
                break
        
        while True:
            # Passo 3
            U = {i for i in range(len_lines) if wi[i] == 0}
            U = U - {0}

            if not U:
                break

            # Passo 4

            S_prime_E = {j for j in S_prime if self.custos[j-1] <= E}

            #wi significa QUANTAS COLUNAS COBRE A LINHA i
            #self.linhas_cobertas[i] significa quais linhas sao cobertas pela coluna i
            #S_prime_E === melhores colunas para serem colocadas na solução
            
            alpha = [[1 if wi[i] == 0 and i in self.linhas_cobertas[j-1] and j in S_prime_E else 0 for j in range(len_columns)] for i in range(len_lines)]

            alpha_transpose = list(map(list, zip(*alpha)))  # Transpose alpha matrix

            v_j = [sum(alpha_transpose[j]) if j in S_prime_E else 0 for j in range(len_columns)]

            beta_j = [self.custos[j-1] / v_j[j] if v_j[j] != 0 and j in S_prime_E else float('inf') for j in S_prime_E]

            beta_min = min(beta_j)

            K = [list(S_prime_E)[j] for j in range(len(S_prime_E)) if beta_j[j] == beta_min]

            # Passo 5
            k = random.choice(list(K))
            S_prime.remove(k)
            S.add(k)

            Z_S += round(self.custos[k-1], 2)

            cont = [0 for _ in range(len_lines)]

            for i in range(len_lines):
                if k in self.colunas_que_cobrem_linha[i]:
                    cont[i] += 1
            wi = [wi[i] + cont[i] for i in range(len_lines)]

        # Passo 6
        for k in reversed(list(S)):
            cont = [0 for _ in range(len_lines)]
            for i in range(len_lines):
                if k in self.colunas_que_cobrem_linha[i]:
                    cont[i] += 1

            flag = True
            for i in self.linhas_cobertas[k-1]:
                if wi[i] - cont[i] < 1:
                    flag = False

            if flag:
                S.remove(k)
                S_prime.add(k)

                Z_S -= round(self.custos[k-1], 2)

                cont = [0 for _ in range(len_lines)]
                for i in range(len_lines):
                    if k in self.colunas_que_cobrem_linha[i]:
                        cont[i] += 1
                wi = [wi[i] - cont[i] for i in range(len_lines)]

        solucao = Solution(S.copy(), round(Z_S, 2))

        return solucao
    
    def genetic_algorithm(self):
        # Initialize population
        population = []
        population_size = 20
        for _ in range(population_size):
            individual = self.vasko_wilson()
            population.append(individual)

        population.sort(key=lambda individual: individual.cost)
        # Evolutionary loop
        generations = 100
        # Halting criterea: number of generations
        for _ in range(generations):
            # Calculate the fitness sum of the population
            fitness_sum = sum(individual.cost for individual in population)

            # Evaluate fitness of each individual
            self.evaluate_fitness(population, fitness_sum)

            # Select parents for reproduction
            parents = self.selection_roulette(population)

            # Apply crossover and mutation to create new offspring
            offspring = self.crossover(parents)
            
            population = self.mutation(population, mutation_rate=0.1, population_size=population_size)

            # Replace the worst individual from the old population with the offspring
            population.sort(key=lambda individual: individual.cost)  # Sort the population by cost
            population.pop()
            population.append(offspring)

        # Select the best individual as the solution
        best_individual = population[0]

        return best_individual

    def genetic_algorithm_local_search(self, population_size, generations, mutation_rate, selection_method):
        # Initialize population
        population = []

        for _ in range(population_size):
            individual = self.vasko_wilson()
            population.append(individual)

        population.sort(key=lambda individual: individual.cost)
        # Evolutionary loop

        # Halting criterea: number of generations
        for _ in range(generations):
            print("População: ")
            for individual in population:
                print(individual.columns, "custo: ", individual.cost)
            # Calculate the fitness sum of the population
            fitness_sum = sum(individual.cost for individual in population)

            # Evaluate fitness of each individual
            self.evaluate_fitness(population, fitness_sum)

            # Select parents for reproduction
            if selection_method == "roulette":
                parents = self.selection_roulette(population)
            else:
                parents = self.selection_tournament(population, population_size//2)

            # Apply crossover and mutation to create new offspring
            offspring = self.crossover(parents)
            
            population = self.mutation(population, mutation_rate=mutation_rate, population_size=population_size)

            offspring = self.busca_vizinhanca(offspring.columns.copy(), offspring.cost, 0.8, 1)

            # Replace the worst individual from the old population with the offspring
            population.sort(key=lambda individual: individual.cost)  # Sort the population by cost
            population.pop()
            population.append(offspring)

            for individual in population:
                individual = self.busca_vizinhanca(individual.columns.copy(), individual.cost, 0.8, 1)

            population.sort(key=lambda individual: individual.cost)
        # Select the best individual as the solution
        best_individual = population[0]

        return best_individual

    def evaluate_fitness(self, population, fitness_sum):
        # Calculate the fitness score for the given individual based on its cost
        for individual in population:
            individual.fitness = (individual.cost/fitness_sum)*100


    def selection_tournament(self, population, tournament_size):
        # Select two different parents
        parents = []
        for _ in range(2):
            # Randomly select a number of individuals from the population
            tournament = random.sample(population, tournament_size)
            
            # Sort the tournament by fitness
            tournament.sort(key=lambda individual: individual.fitness)
            
            # Select the individual with the highest fitness
            parents.append(tournament[-1])

        # Return the selected parents
        #print("Pais selecionados: ", parents[0].columns, parents[1].columns)
        return parents
    
    def selection_roulette(self, population):
        # Select two different parents
        parents = []
        already_used_fitness = 0
        for _ in range(2):
            # Generate a random number between 0 and the total fitness score
            random_number = random.uniform(0, 100 - already_used_fitness)
            
            # Iterate through the population and find the individuals that correspond to the random number
            cumulative_fitness = 0
            for individual in population:
                cumulative_fitness += individual.fitness
                
                if cumulative_fitness >= random_number and individual not in parents:
                    parents.append(individual)
                    already_used_fitness += individual.fitness
                    break

        # Return the selected parents
        print("Pais selecionados: ", parents[0].columns, parents[1].columns)
        return parents


    def crossover(self, parents):
        parent1 = parents[0]
        parent2 = parents[1]
        
        len_lines = self.quant_linhas+1
        # Perform crossover operation to create a new individual
        # For example, you can randomly select a crossover point and swap the genetic material between parents
        
        offspring_cols = parent1.columns.union(parent2.columns)

        wi = [sum(1 for col in self.colunas_que_cobrem_linha[i] if col in offspring_cols) for i in range(len_lines)]

        # Iterate through the columns and remove the unnecessary ones
        for k in reversed(list(offspring_cols)):
            cont = [sum(1 for col in self.colunas_que_cobrem_linha[i] if col in offspring_cols) for i in range(len_lines)]

            flag = True
            for i in self.linhas_cobertas[k-1]:
                if wi[i] - cont[i] < 1:
                    flag = False

            if flag:
                offspring_cols.remove(k)
                wi = [wi[i] - cont[i] for i in range(len_lines)]
        
        valor = 0
        for i in offspring_cols:
            valor += self.custos[i-1]

        offspring = Solution(offspring_cols, round(valor, 2))
        #print("Filho gerado: ", offspring.columns)
        # Return the offspring
        return offspring

    def mutation(self, population, mutation_rate, population_size):
        # Implement your mutation method here
        len_lines = self.quant_linhas+1

        population_to_mutate = [Solution(individual.columns.copy(), individual.cost) for individual in population]
        individuals_to_mutate = random.sample(population_to_mutate, int(mutation_rate*population_size))

        for individual in individuals_to_mutate:
            # Add random amount of columns
            num_columns_to_add = random.randint(1, int(len(self.colunas)*0.1))
            columns_to_add = random.sample(self.colunas, num_columns_to_add)
            # Remove columns that are already in the individual
            available_columns = [column for column in columns_to_add if column not in individual.columns]
            
            individual.columns.update(available_columns)
            
            # Shuffle the solution
            individual.columns = list(individual.columns)

            random.shuffle(individual.columns)

            individual.columns = set(individual.columns)
                    
            # Remove unnecessary columns
            wi = [sum(1 for col in self.colunas_que_cobrem_linha[i] if col in individual.columns) for i in range(len_lines)]

            # Iterate through the columns and remove the unnecessary ones
            for k in reversed(list(individual.columns)):

                cont = [sum(1 for col in self.colunas_que_cobrem_linha[i] if col in individual.columns) for i in range(len_lines)]

                flag = True
                for i in self.linhas_cobertas[k-1]:
                    if wi[i] - cont[i] < 1:
                        flag = False

                if flag:
                    individual.columns.remove(k)
                    
                    wi = [wi[i] - cont[i] for i in range(len_lines)]
            
            valor = 0
            for i in individual.columns:
                valor += self.custos[i-1]
            individual.cost = round(valor, 2)

        population.sort(key=lambda x: x.cost)
        print("-----------------------ORDENAÇÃO DA POPULAÇÃO APÓS MUTAÇÃO:----------------------")
        for individual in population:
            print(individual.columns, "custo: ", individual.cost)
        for _ in range(int(mutation_rate*population_size)):
            population.pop()
        population.extend(individuals_to_mutate)
        population.sort(key=lambda individual: individual.cost)

        return population

    def vasko_wilson(self):
        M = set(range(1, self.quant_linhas+1))
        P = deque(self.linhas_cobertas)
        P.appendleft([])
        k = [len(P[j]) for j in range(len(self.linhas_cobertas)+1)]
        
        R = M.copy()
        S = set()
        t = 1
        j = []

        while R:
            k = {j: len(set(P[j]).intersection(R)) for j in range(self.quant_colunas+1)}
            
            j_a_escolher = random_greedy(self.custos, k)
           
            minimo = float('inf')
            for i in range(len(j_a_escolher)):
                if j_a_escolher[i] < minimo:
                    min_idx = i
                    minimo = j_a_escolher[i]

            j.append(min_idx)

            Pj_t = P[j[t-1]]
            R = list(R)
            
            for i in range(len(Pj_t)):
                if (Pj_t[i] in R):
                    R.remove(Pj_t[i])
            S.add(j[t-1])

            t += 1

        # Passo 2
        sorted_S = set(sorted(S, key=lambda j: self.custos[j-1], reverse=True)).copy()

        for i in sorted_S:
            if is_covering_feasible(self.linhas_cobertas, sorted_S - {i}, M):
                sorted_S = sorted_S.copy() - {i}

        S = sorted_S.copy()

        valor = 0
        for i in S:
            valor += self.custos[i-1]

        solucao = Solution(S.copy(), round(valor, 2))
        
        return solucao   


def random_greedy(c, k):
    j_a_escolher = []

    x = random.choice([1, 2, 3, 4, 5, 6, 7, 9])

    match x:
        case 1:
            for j in range(len(k)):
                if k[j] != 0:
                    j_a_escolher.append(c[j-1])
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher
        case 2:
            for j in range(len(k)):
                if k[j] != 0:
                    j_a_escolher.append(c[j-1] / k[j])
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher
        case 3:
            for j in range(len(k)):
                if k[j] != 0 and k[j] != 1:
                    j_a_escolher.append(c[j-1]/math.log2(k[j]))
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher
        case 4:
            for j in range(len(k)):
                if k[j] != 0 and k[j] != 1:
                    j_a_escolher.append(c[j-1]/ (k[j] * math.log2(k[j])))
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher
        case 5:
            for j in range(len(k)):
                if k[j] != 0 and k[j] != 1:
                    j_a_escolher.append(c[j-1]/(k[j] * math.log(k[j])))
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher
        case 6:
            for j in range(len(k)):
                if k[j] != 0:
                    j_a_escolher.append(c[j-1]/(k[j] * k[j]))
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher
        case 7:
            for j in range(len(k)):
                if k[j] != 0:
                    j_a_escolher.append(math.sqrt(c[j-1])/(k[j] * k[j]))
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher
        case 9:
            for j in range(len(k)):
                if k[j] != 0:
                    j_a_escolher.append(k[j] / c[j-1])
                else:
                    j_a_escolher.append(float('inf'))
            return j_a_escolher


def parse_arquivo(nome_arquivo):
    indice_coluna = []
    custo = []
    indice_linha = []
    
    with open(nome_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        
        for i, linha in enumerate(linhas):
            if linha.startswith("LINHAS") or linha.startswith ("Linhas"):
                partes = linha.split()
                quant_linhas = int(partes[1])

            if linha.startswith("COLUNAS") or linha.startswith("COLUNA") or linha.startswith("Colunas"):
                partes = linha.split()
                quant_colunas = int(partes[1])

            if linha.startswith("DADOS") or linha.startswith("Densidade"):
                colunas_que_cobrem_linha = [[] for _ in range(int(quant_linhas)+1)]
                for j in range(i+1, len(linhas)):
                    partes = linhas[j].split()
                    if partes: 
                        coluna_atual = int(partes[0])
                        indice_coluna.append(coluna_atual)
                        custo.append(float(partes[1]))
                        novas_linhas = (list(map(int, partes[2:])))
                        for i in novas_linhas:
                            colunas_que_cobrem_linha[i].append(coluna_atual)
                        indice_linha.append(novas_linhas)
                break
    scp = SCP(quant_colunas=quant_colunas, quant_linhas=quant_linhas, colunas=indice_coluna, linhas_cobertas=indice_linha, custos=custo, colunas_que_cobrem_linha=colunas_que_cobrem_linha)
    return scp


def is_covering_feasible(columns, selected_columns, M):
        covered_rows = set()
        for col_idx in selected_columns:
            covered_rows.update(columns[col_idx-1])

        covered_rows = sorted(covered_rows)

        return set(M) == set(covered_rows)


"""
Daqui pra baixo ficam os testes!!!
"""

def constructor_test(file_path, constructor_func=lambda x: x.vasko_wilson(), iterations=101):
    file_path = os.path.realpath(file_path)
    scp = parse_arquivo(file_path)
    melhorz = float('inf')
    melhors = set()
    for _ in range(iterations):
        S, Z_S = constructor_func(scp)
        if Z_S < melhorz:
            melhorz = Z_S
            melhors = S.copy()
    print("O melhor resultado para o construtor foi:", melhors)
    print("Custo:", melhorz)

def teste_genetico_local_search(file_path):
    file_path = os.path.realpath(file_path)
    scp = parse_arquivo(file_path)
    melhor_solucao = scp.genetic_algorithm_local_search(20, 300, 0.2, "roulette")
    print("Melhor solução encontrada: ", melhor_solucao.columns)
    print("Custo: ", melhor_solucao.cost)

def teste_genetico(file_path):
    file_path = os.path.realpath(file_path)
    scp = parse_arquivo(file_path)
    melhor_solucao = scp.genetic_algorithm()
    print("Melhor solução encontrada: ", melhor_solucao.columns)
    print("Custo: ", melhor_solucao.cost)

if __name__ == "__main__":
    t1 = time.time()
    """
    Testes do algoritmo genético com busca local
    """
    teste_genetico_local_search("Teste_03.dat")

    """
    Testes do algoritmo genético sem busca local
    """
    #teste_genetico("Teste_01.dat")

    t2 = time.time()
    print("O tempo total em segundos da execução foi", round(t2 - t1, 2))