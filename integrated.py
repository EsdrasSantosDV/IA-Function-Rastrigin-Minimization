import tkinter
from tkinter import *
from tkinter import ttk
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random as rd
from scipy.optimize import dual_annealing
from matplotlib.figure import Figure
import math
import random
import pandas as pd
from copy import deepcopy


def f(x, y):
    return 21.5 + x*math.sin(4*math.pi*x) + y*math.sin(20*math.pi*y)

x_interval = (-3.1, 12.1)
y_interval = (4.1, 5.8)

def f_opt(params):
    x, y = params
    return f(x, y)

x_bounds = (-3.1, 12.1)
y_bounds = (4.1, 5.8)

result = dual_annealing(f_opt, bounds=[x_bounds, y_bounds])

x_opt, y_opt = result.x
f_opt = result.fun

print(f"Optimal point: ({x_opt}, {y_opt})")
print(f"Optimal value: {f(11.87468546,  5.77504386)}")


def create_individual():
    x1 = random.uniform(-3.1, 12.1)
    x2 = random.uniform(4.1, 5.8)
    return np.array([x1, x2])


def create_population(population_size):
    return [create_individual() for _ in range(population_size)]

#O FITNESS
# Retorna o valor de aptidão, que é o inverso do valor da função de RASTRIGIN. Quanto menor o valor da função, maior a aptidão do indivíduo.
def fitness(individual, print_params = False):
    return 1 / (21.5 + individual[0] * np.sin(4*np.pi*individual[0]) + individual[1] * np.sin(20*np.pi*individual[1]))


def select_roulette(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    probs = [f / total_fitness for f in fitnesses]

    parents = []

    for _ in range(num_parents):
        r = random.random()  # Gera um número aleatório entre 0 e 1
        for i, individual in enumerate(population):
            r -= probs[i]
            if r <= 0:
                parents.append(deepcopy(individual))
                break

    return parents




def select_tournament(population, fitnesses, num_parents, tournament_size):
    parents = []
    for _ in range(num_parents):
        contenders = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(contenders, key=lambda x: x[1])[0]
        parents.append(deepcopy(winner))

    return parents


def radcliff_crossover(parent1, parent2, crossover_rate):
    beta = random.uniform(0, 1)
    if beta > float(crossover_rate):
        # Verifica se o valor de beta é maior que a taxa de crossover especificada
        # Se for, retorna os pais originais sem fazer o crossover
        return parent1, parent2
    # Realiza o cálculo do crossover RADCLIFF
    x1A = beta * parent1[0] + (1 - beta) * parent2[0]
    x2A = beta * parent1[1] + (1 - beta) * parent2[1]
    x1B = (1 - beta) * parent1[0] + beta * parent2[0]
    x2B = (1 - beta) * parent1[1] + beta * parent2[1]
    # Retorna os filhos resultantes do crossover RADCLIFF
    return np.array([deepcopy(x1A), deepcopy(x2A)]), np.array([deepcopy(x1B), deepcopy(x2B)])


def wright_crossover(parent1, parent2,crossover_rate,children):
    beta = random.uniform(0, 1)
    if beta > float(crossover_rate):
        return parent1, parent2
    # First child
    childA = 0.5 * (parent1 + parent2)
    # Second child
    childB = 1.5 * parent1 - 0.5 * parent2
    # Third child
    childC = 1.5 * parent2 - 0.5 * parent1

    if children==1:
        return childA
    elif children==2:
        return childB
    else:return childC




def mutate(individual, mutation_rate):
    individual_copy = deepcopy(individual)
    should_mutate = random.random()
    if should_mutate < mutation_rate:
        choose_random = random.randint(0, 1)
        if choose_random == 0:
            individual_copy[0] = random.uniform(-3.1, 12.1)
        else:
            individual_copy[1] = random.uniform(4.1, 5.8)
    return deepcopy(individual_copy)


def replace_population(population, new_individuals):
    population.sort(key=fitness)
    population[:len(new_individuals)] = deepcopy(new_individuals)

    return deepcopy(population)


def genetic_algorithm(population_size,num_generations,mutacao_probabilidade,tamanho_elitismo,probabilidade_de_cruzamento,selection_method,tamanho_torneio=None,randclif_method="randclif"):
    # Criação inicial da população.
    population = create_population(population_size)


    print(randclif_method);


    # Inicializar melhor fitness e a geração em que foi encontrado.
    geracaoencontrada = 0
    best_fitness = 0

    # Loop principal do algoritmo genético, repetido por um número definido de gerações.
    for gen in range(num_generations):

        # Cálculo da aptidão (fitness) para cada indivíduo na população.
        fitnesses = [fitness(individual) for individual in population]

        # Seleção dos pais para a próxima geração
        if selection_method == "tournament":
            parents = select_tournament(population, fitnesses, population_size // 2, tamanho_torneio)
        else:
            parents = select_roulette(population, fitnesses, population_size // 2)

        # Criação de filhos a partir dos pais usando cruzamento.
        children = []
        right_childrens = []

        for i in range(0, len(parents) - 1, 2):

            if randclif_method == "randclif":
                child1, child2 = radcliff_crossover(parents[i], parents[i + 1], probabilidade_de_cruzamento)
            else:

                child1 = wright_crossover(parents[i], parents[i + 1],probabilidade_de_cruzamento,1)
                child2= wright_crossover(parents[i], parents[i + 1],probabilidade_de_cruzamento,2)
                child3 = wright_crossover(parents[i], parents[i + 1], probabilidade_de_cruzamento, 3)

                # Calculate the fitness of each child
                fitnessA = fitness(child1)
                fitnessB = fitness(child2)
                fitnessC = fitness(child3)



            ##child1, child2 = radcliff_crossover(parents[i], parents[i + 1], probabilidade_de_cruzamento)
            children.append(child1)
            children.append(child2)



        # Se o elitismo está sendo usado, os melhores indivíduos são copiados.
        if (tamanho_elitismo > 0):
            population.sort(key=fitness, reverse=True)
            elites = deepcopy(population[:tamanho_elitismo])

        # Mutação é aplicada aos filhos.
        mutated_children = [mutate(child, mutacao_probabilidade) for child in children]
        # Substituição da população atual pelos filhos mutados.
        population = replace_population(population, mutated_children)

        # Se o elitismo está sendo usado, os piores indivíduos são substituídos pelos elites.
        if (tamanho_elitismo > 0):
            population.sort(key=fitness)
            population[:tamanho_elitismo] = deepcopy(elites)

        # Recalcular a aptidão da população.
        statistic_fitness = [fitness(individual) for individual in population]

        if (tamanho_elitismo > 0):
            fitness(elites[0])

        # Se a melhor aptidão encontrada nesta geração é melhor do que a melhor até agora, atualizar best_fitness e geracaoencontrada.
        if max(statistic_fitness) > best_fitness:
            geracaoencontrada = gen
            best_fitness = max(statistic_fitness)

        print('Geração',gen+1,'Fitness: ',max(statistic_fitness),'Melhor Geração',geracaoencontrada+1)


    # Encontrar o melhor indivíduo da última geração.
    best_individual = max(population, key=fitness)
    print(best_individual)
    melhor_geracao.config(text=f"Geração em que foi encontrado a melhor geração: {geracaoencontrada+1}")
    return best_individual




def submit_button_event():
    population_size = int(form_tamanho_da_populacao.get())
    probabilidade_de_cruzamento = float(form_probabilidade_de_cruzamento.get())
    mutacao_probabilidade = float(form_mutacao_probabilidade.get())
    num_generations = int(form_quantidade_geracoes.get())
    if check_var.get():
        tamanho_torneio = int(form_tamanho_torneio.get())
    else:
        tamanho_torneio = None
    tamanho_elitismo = int(form_tamanho_elitismo.get())
    tamanho_elitismo = int(form_tamanho_elitismo.get())
    selection_method = "tournament" if check_var.get() else "roulette"
    randclif_method= "randclif" if check_var2.get() else "wrigth"
    print("Tamanho da população:", population_size)
    print("Probabilidade de cruzamento:", probabilidade_de_cruzamento)
    print("Probabilidade de mutação:", mutacao_probabilidade)
    print("Quantidade de gerações:", num_generations)
    print("Tamanho do torneio:", tamanho_torneio)
    print("Tamanho do elitismo:", tamanho_elitismo)
    print("Selection method:", selection_method)
    schedule = genetic_algorithm(population_size,num_generations,mutacao_probabilidade,tamanho_elitismo,probabilidade_de_cruzamento,selection_method,tamanho_torneio,randclif_method)
    print("",f(schedule[0],schedule[1]))
    porcent = abs(f_opt/f(schedule[0], schedule[1])) * 100
    porcentagem_de_erro.config(text=f"Porcentagem de erro entre o Máximo encontrado e o Maximo Real:{100-porcent}")
    resultadoreal.config(text=f"Valor Maximo da Função, Valor Real:{f_opt}")
    imagem_valores_encontrado.config(text=f"Imagem do Valor dado os Valores Encontrado: {f(schedule[0], schedule[1])}")
    valores_encontrados.config(text=f"Imagem do Valor dado os Valores Encontrado: {schedule}")


def fill_form():
    form_tamanho_da_populacao.delete(0, tkinter.END)
    form_tamanho_da_populacao.insert(0, str(210))
    form_probabilidade_de_cruzamento.delete(0, tkinter.END)
    form_probabilidade_de_cruzamento.insert(0, str(0.85))
    form_mutacao_probabilidade.delete(0, tkinter.END)
    form_mutacao_probabilidade.insert(0, str(0.05))
    form_quantidade_geracoes.delete(0, tkinter.END)
    form_quantidade_geracoes.insert(0, str(300))
    form_tamanho_torneio.delete(0, tkinter.END)
    form_tamanho_torneio.insert(0, str(8))
    form_tamanho_elitismo.delete(0, tkinter.END)
    form_tamanho_elitismo.insert(0, str(3))

def toggle_torneio_visibility():
    if check_var.get():
        label_tamanho_torneio.place(x=100, y=200)
        form_tamanho_torneio.place(x=325, y=200)
    else:
        label_tamanho_torneio.place_forget()
        form_tamanho_torneio.place_forget()


window = Tk()
window.title("IA-TRABALHO-GRUPO-ESDRAS-JOAO-OTAVIO-FELIPE MENDES")
window.geometry('600x600')
window.configure(background="gray")

label_form=tkinter.Label(window,text="Dados das Entradas",background="gray")
label_tamanho_da_populacao = tkinter.Label(window, text="Tamanho da População:",background="gray")
label_probabilidade_de_cruzamento = tkinter.Label(window, text="Probabilidade de Cruzamento:",background="gray")
label_mutacao_probabilidade = tkinter.Label(window, text="Probabilidade de Mutação:",background="gray")
label_quantidade_geracoes = tkinter.Label(window, text="Quantidade de Geracoes:",background="gray")
label_tamanho_torneio =  tkinter.Label(window, text="Tamanho do Torneio:",background="gray")
label_tamanho_elitismo = tkinter.Label(window, text="Tamanho do Elitismo:", background="gray")
result_best_individual = tkinter.Label(window, text="Melhor Individuo Encontrado:")
result_function_value = tkinter.Label(window, text="Aptidão:")
melhor_geracao=tkinter.Label(window,text="Geração em que foi encontrado a melhor geração:")
valores_encontrados=tkinter.Label(window,text="Os valores encontrado da funcao:")
resultadoreal=tkinter.Label(window,text="Valor Maximo da Função:")
porcentagem_de_erro=tkinter.Label(window,text="Porcentagem de erro entre o Máximo encontrado e o Maximo Real:")
resultadoreal=tkinter.Label(window,text="Valor Maximo da Função, Valor Real:")
porcentagem_de_erro=tkinter.Label(window,text="Porcentagem de erro entre o Máximo encontrado e o Maximo Real:")
imagem_valores_encontrado = tkinter.Label(window, text="Imagem do Valor dado os Valores Encontrado:")

form_tamanho_elitismo = tkinter.Entry()
form_tamanho_da_populacao=tkinter.Entry()
form_probabilidade_de_cruzamento=tkinter.Entry()
form_mutacao_probabilidade=tkinter.Entry()
form_quantidade_geracoes=tkinter.Entry()
form_tamanho_torneio=tkinter.Entry()


label_form.place(x=200,y=10)
label_tamanho_da_populacao.place(x=100,y=50)
form_tamanho_da_populacao.place(x=325,y=50)
label_probabilidade_de_cruzamento.place(x=100,y=80)
form_probabilidade_de_cruzamento.place(x=325,y=80)
label_quantidade_geracoes.place(x=100,y=110)
form_quantidade_geracoes.place(x=325,y=110)
label_mutacao_probabilidade.place(x=100,y=140)
form_mutacao_probabilidade.place(x=325,y=140)
label_tamanho_elitismo.place(x=100, y=170)
form_tamanho_elitismo.place(x=325, y=170)
result_best_individual.place(x=100, y=700)
result_function_value.place(x=100, y=750)
melhor_geracao.place(x=100,y=400)
valores_encontrados.place(x=100,y=430)
resultadoreal.place(x=100,y=460)
imagem_valores_encontrado.place(x=100,y=490)
porcentagem_de_erro.place(x=100,y=520)
#CHECKBOX
check_var = tkinter.BooleanVar()
check_torneio = tkinter.Checkbutton(window, text="Torneio", variable=check_var, command=toggle_torneio_visibility, background="gray")
check_torneio.place(x=100, y=230)
label_tamanho_torneio.place_forget()
form_tamanho_torneio.place_forget()

check_var2 = tkinter.BooleanVar()
check_torneio = tkinter.Checkbutton(window, text="Randclif", variable=check_var2, background="gray")
check_torneio.place(x=100, y=260)


#SUBMIT
submit_button = tkinter.Button(window, text="Submit", command=submit_button_event)
submit_button.place(x=350, y=300)

#BOTAO PREENCHER
preencher_button=tkinter.Button(window,text="To Fill",command=fill_form)
preencher_button.place(x=350,y=350)

window.mainloop()