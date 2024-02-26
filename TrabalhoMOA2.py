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

            beta_j = [self.custos[j-1] / v_j[j] if v_j[j] != 0 and j in S_prime_E else float('inf') for j in S_prime_E]

            beta_min = min(beta_j)

            K = [list(S_prime_E)[j] for j in range(len(S_prime_E)) if beta_j[j] == beta_min]

            # Passo 5
            k = random.choice(list(K))
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

        solucao = Solution(S.copy(), round(Z_S, 2))

        return solucao

    def genetic_algorithm(self):
        # Initialize population
        population = []
        population_size = 10
        for _ in range(population_size):
            individual = self.vasko_wilson()
            population.append(individual)

        # Evolutionary loop
        generations = 100
        # Halting criterea: number of generations
        for _ in range(generations):
            # Calculate the fitness sum of the population
            fitness_sum = sum(individual.cost for individual in population)

            # Evaluate fitness of each individual
            self.evaluate_fitness(population, fitness_sum)

            # Select parents for reproduction
            parents = self.selection_roulette(population, fitness_sum)

            # Apply crossover and mutation to create new offspring
            offspring = self.crossover(parents)
            offspring = self.mutation(offspring)

            # Replace the old population with the new offspring
            population = offspring

        # Select the best individual as the solution
        best_individual = max(individual.fitness, key=lambda x: x[1])[0] # NÃO FUNCIONA
        best_fitness = self.evaluate_fitness(best_individual)

        return best_individual, best_fitness

    def evaluate_fitness(self, population, fitness_sum):
        # Calculate the fitness score for the given individual based on its cost
        for individual in population:
            individual.fitness = (individual.cost/fitness_sum)*100
    
    def selection_roulette(self, population, fitness_sum):
        # Implement roulette wheel selection method here
        # Calculate the total fitness score of the population

        # Generate a random number between 0 and the total fitness score
        random_number = random.uniform(0, fitness_sum)

        # Iterate through the fitness scores and find the individual that corresponds to the random number
        cumulative_fitness = 0
        for individual in population:
            cumulative_fitness += individual.fitness
            if cumulative_fitness >= random_number:
                selected_individual = individual
                break
        # Return the selected individual as the parent
        return selected_individual


    def crossover(self, parents):
        # Implement your crossover method here
        # Perform crossover operation on the selected parents to create offspring
        # Return the offspring
        return

    def mutation(self, offspring):
        # Implement your mutation method here
        # Perform mutation operation on the offspring
        # Return the mutated offspring
        return

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
            
            j_a_escolher = random_greedy(self.custos, k, 1)
           
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
    

    def luca(self):
        melhor_solucao = None
        melhor_custo = float('inf')

        for _ in range(5):
            M = set(range(1, self.quant_linhas + 1))
            P = deque(self.linhas_cobertas)
            P.appendleft([])

            k = [len(P[j]) for j in range(len(self.linhas_cobertas) + 1)]

            R = M.copy()
            S = set()
            t = 1
            j = []

            while R:
                k = {j: len(set(P[j]).intersection(R)) for j in range(self.quant_colunas + 1)}

                j_a_escolher = random_greedy(self.custos, k, 0)
                minimo = float('inf')
                for i in range(len(j_a_escolher)):
                    if j_a_escolher[i] < minimo:
                        min_idx = i
                        minimo = j_a_escolher[i]

                j.append(min_idx)

                Pj_t = P[j[t - 1]]
                R = list(R)

                for i in range(len(Pj_t)):
                    if (Pj_t[i] in R):
                        R.remove(Pj_t[i])
                S.add(j[t - 1])

                t += 1

            sorted_S = set(sorted(S, key=lambda j: self.custos[j-1], reverse=True))

            for i in sorted_S:
                if is_covering_feasible(self.linhas_cobertas, sorted_S - {i}, M):
                    sorted_S = sorted_S - {i}

            S = sorted_S

            valor = 0
            for i in S:
                valor += self.custos[i-1]

            if valor < melhor_custo:
                melhor_custo = round(valor, 2)
                melhor_solucao = S.copy()

            print("Reinicializando...")

        print("Melhor solução encontrada: ", melhor_solucao)
        print("Custo: ", melhor_custo)

        solucao = Solution(melhor_solucao.copy(), round(melhor_custo, 2))

        return solucao   


def random_greedy(c, k, choice):
    j_a_escolher = []
    if choice:
        x = random.choice([1, 2, 3, 4, 5, 6, 7])
    else:
        x = random.choice([9, 2])

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

def run_test(file_path, constructor_func=lambda x: x.vasko_wilson(), improving_func=lambda x, S, Z_S, P1, P2: x.busca_vizinhanca(S, Z_S, P1, P2), imp_iterations=11, cons_iterations=11):
    file_path = os.path.realpath(file_path)
    scp = parse_arquivo(file_path)
    
    melhor = float('inf')
    for _ in range(cons_iterations):
        Sol = constructor_func(scp)
        print("Solucao do CONSTRUTOR: ", Sol.columns)
        print("CUSTO: ", Sol.cost)
        for _ in range(imp_iterations):
            Sol1 = improving_func(scp, S=Sol.columns.copy(), Z_S=Sol.cost, P1=0.8, P2=1)
            print("Solucao melhorada: ", Sol1.columns)
            print("CUSTO: ", Sol1.cost)
            if Sol1.cost < melhor:
                sres = Sol1.columns.copy()
                melhor = Sol1.cost

    for _ in range(26):
        retryS = scp.busca_vizinhanca(S=sres.copy(), Z_S=melhor, P1=0.2, P2=1)
        if retryS.cost <= melhor:
            melhor = retryS.cost
            sres = retryS.columns.copy()

    print("MELHOR MELHORAMENTO: ", sres)
    print("CUSTO:", melhor)

def teste1():
    random.seed(6)
    run_test("Teste_01.dat")

def teste2():
    random.seed(7)
    run_test("Teste_02.dat", constructor_func=lambda x: x.luca())

def teste3():
    random.seed(3)
    run_test("Teste_03.dat")

def teste4():
    random.seed(1)
    run_test("Teste_04.dat")

def wren1():
    random.seed(10)
    run_test("Wren_01.dat", constructor_func=lambda x: x.luca())

def wren2():
    random.seed(20)
    run_test("Wren_02.dat", constructor_func=lambda x: x.luca(), imp_iterations=5)

def wren3():
    random.seed(6)
    run_test("Wren_03.dat")

def wren4():
    random.seed(6)
    run_test("Wren_04.dat")

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

def c_teste1():
    random.seed(1)
    constructor_test("Teste_01.dat")

def c_teste2():
    random.seed(1)
    constructor_test("Teste_02.dat", constructor_func=lambda x: x.luca(), iterations=501)

def c_teste3():
    random.seed(1)
    constructor_test("Teste_03.dat")

def c_teste4():
    random.seed(1)
    constructor_test("Teste_04.dat")

def c_wren1():
    random.seed(2)
    constructor_test("Wren_01.dat", constructor_func=lambda x: x.luca())

def c_wren2():
    random.seed(12)
    constructor_test("Wren_02.dat")

def c_wren3():
    random.seed(17)
    constructor_test("Wren_03.dat")

def c_wren4():
    random.seed(13)
    constructor_test("Wren_04.dat")

if __name__ == "__main__":
    t1 = time.time()
    """
    Testes dos construtores
    """
    #c_teste1()
    #c_teste2()
    #c_teste3()
    #c_teste4()
    #c_wren1()
    #c_wren2()
    #c_wren3()
    #c_wren4()
    """
    Testes dos melhorativos
    """
    teste1()
    #teste2()
    #teste3()
    #teste4()
    #wren1()
    #wren2()
    #wren3()
    #wren4()
    t2 = time.time()
    print("O tempo total em segundos da execução foi", round(t2 - t1, 2))