import tkinter
from tkinter import *
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random as rd
from scipy.optimize import dual_annealing
from matplotlib.figure import Figure
import math
from scipy.optimize import minimize_scalar, shgo
def f(x, y):
    return 21.5 + x*math.sin(4*math.pi*x) + y*math.sin(20*math.pi*y)

x_interval = (-3.1, 12.1)
y_interval = (4.1, 5.8)

def neg_f(params):
    x, y = params
    return -f(x, y)

x_bounds = (-3.1, 12.1)
y_bounds = (4.1, 5.8)

result = dual_annealing(neg_f, bounds=[x_bounds, y_bounds])

x_opt, y_opt = result.x
f_opt = -result.fun

print(f"Optimal point: ({x_opt}, {y_opt})")
print(f"Optimal value: {f_opt}")
def create_individual():
    x = random.uniform(x_interval[0], x_interval[1])
    y = random.uniform(y_interval[0], y_interval[1])
    return (x, y)

def mutate_individual(individual, mutation_rate):
    if random.random() < mutation_rate:
        x = individual[0] + random.uniform(-0.1, 0.1)
        y = individual[1] + random.uniform(-0.1, 0.1)
        x = max(min(x, x_interval[1]), x_interval[0])
        y = max(min(y, y_interval[1]), y_interval[0])
        return (x, y)
    else:
        return individual

def crossover(parent1, parent2, crossover_rate, crossover_type):
    if random.random() < crossover_rate:
        if crossover_type == "one_point":
            x = (parent1[0] + parent2[0]) / 2
            y = (parent1[1] + parent2[1]) / 2
        elif crossover_type == "two_points":
            cut_point = random.uniform(0.25, 0.75)
            x = parent1[0] * cut_point + parent2[0] * (1 - cut_point)
            y = parent1[1] * cut_point + parent2[1] * (1 - cut_point)
        return (x, y)
    else:
        return parent1 if random.random() < 0.5 else parent2
def tournament_selection(population, fitnesses, tournament_size):
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_index]

def roulette_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    r = random.random() * total_fitness
    idx = 0
    while r > 0:
        r -= fitnesses[idx]
        idx += 1
    return population[idx - 1]

def genetic_algorithm(pop_size, mutation_rate, crossover_rate, generations, selection_method, tournament_size=None, elitism_size=1,crossover_type ="two_point"):
    # Initialize population
    population = [create_individual() for _ in range(pop_size)]

    populacao_pela_geracao=0
    geracaoencontrada=0
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [f(individual[0], individual[1]) for individual in population]

        # Select parents
        if selection_method == "tournament":
            select = lambda: tournament_selection(population, fitnesses, tournament_size)
        elif selection_method == "roulette":
            select = lambda: roulette_selection(population, fitnesses)

        # Elitism
        elite_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:elitism_size]
        new_population = [population[i] for i in elite_indices]

        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = select()
            parent2 = select()
            offspring = crossover(parent1, parent2, crossover_rate, crossover_type)  # include crossover_type here
            offspring = mutate_individual(offspring, mutation_rate)
            new_population.append(offspring)
        population = new_population
        if populacao_pela_geracao != max(population, key=lambda x: f(x[0], x[1])):
            populacao_pela_geracao = max(population, key=lambda x: f(x[0], x[1]))
            geracaoencontrada=gen+1



        print(f"adjssja:{geracaoencontrada}")
        result_best_individual.config(text=f"Melhor Individuo Encontrado: {populacao_pela_geracao}")
        print(f"Melhor Individuo Encontrado: {populacao_pela_geracao}")
        print(f"Aptidão: {f(populacao_pela_geracao[0], populacao_pela_geracao[1])}")
        result_function_value.config(text=f"Aptidão: {f(populacao_pela_geracao[0], populacao_pela_geracao[1])}")


    best_individual = max(population, key=lambda x: f(x[0], x[1]))
    melhor_geracao.config(text=f"Geração em que foi encontrado a melhor geração: {geracaoencontrada}")
    return best_individual


def submit_button_event():
    tamanho_cromossomo = int(form_tamanho_cromossomo.get())
    tamanho_da_populacao = int(form_tamanho_da_populacao.get())
    probabilidade_de_cruzamento = float(form_probabilidade_de_cruzamento.get())
    mutacao_probabilidade = float(form_mutacao_probabilidade.get())
    quantidade_geracoes = int(form_quantidade_geracoes.get())
    if check_var.get():
        tamanho_torneio = int(form_tamanho_torneio.get())
    else:
        tamanho_torneio = None
    tamanho_elitismo = int(form_tamanho_elitismo.get())
    tamanho_elitismo = int(form_tamanho_elitismo.get())
    selection_method = "tournament" if check_var.get() else "roulette"
    crossover_type = crossover_var.get()
    print("Tamanho do cromossomo:", tamanho_cromossomo)
    print("Tamanho da população:", tamanho_da_populacao)
    print("Probabilidade de cruzamento:", probabilidade_de_cruzamento)
    print("Probabilidade de mutação:", mutacao_probabilidade)
    print("Quantidade de gerações:", quantidade_geracoes)
    print("Tamanho do torneio:", tamanho_torneio)
    print("Tamanho do elitismo:", tamanho_elitismo)
    print("Selection method:", selection_method)
    print("Crossover type:", crossover_type)

    best_individual = genetic_algorithm(
        pop_size=tamanho_da_populacao,
        mutation_rate=mutacao_probabilidade,
        crossover_rate=probabilidade_de_cruzamento,
        generations=quantidade_geracoes,
        selection_method=selection_method,
        tournament_size=tamanho_torneio,
        elitism_size=tamanho_elitismo,
        crossover_type=crossover_type
    )

    # Atualizar os Labels com os resultados
    result_best_individual.config(text=f"Melhor Individuo Encontrado: {best_individual}")
    result_function_value.config(text=f"Aptidão: {f(best_individual[0], best_individual[1])}")
    porcent=abs(f(best_individual[0], best_individual[1])/f_opt)*100
    porcentagem_de_erro.config(text=f"Porcentagem de erro entre o Máximo encontrado e o Maximo Real:{porcent}")
    resultadoreal.config(text=f"Valor Maximo da Função:{f_opt}")

def fill_form():
    form_tamanho_da_populacao.delete(0, tkinter.END)
    form_tamanho_da_populacao.insert(0, str(30))
    form_tamanho_cromossomo.delete(0, tkinter.END)
    form_tamanho_cromossomo.insert(0, str(10))
    form_probabilidade_de_cruzamento.delete(0, tkinter.END)
    form_probabilidade_de_cruzamento.insert(0, str(0.85))
    form_mutacao_probabilidade.delete(0, tkinter.END)
    form_mutacao_probabilidade.insert(0, str(0.1))
    form_quantidade_geracoes.delete(0, tkinter.END)
    form_quantidade_geracoes.insert(0, str(30))
    form_tamanho_torneio.delete(0, tkinter.END)
    form_tamanho_torneio.insert(0, str(15))
    form_tamanho_elitismo.delete(0, tkinter.END)
    form_tamanho_elitismo.insert(0, str(1))


def toggle_torneio_visibility():
    if check_var.get():
        label_tamanho_torneio.place(x=500, y=200)
        form_tamanho_torneio.place(x=650, y=200)
    else:
        label_tamanho_torneio.place_forget()
        form_tamanho_torneio.place_forget()


def on_crossover_selection():
    selected_crossover = crossover_var.get()



window = Tk()
window.title("IA-TRABALHO-GRUPO-ESDRAS-JOAO-OTAVIO-FELIPE MENDES")
window.geometry('1920x1080')
window.configure(background="gray")

label_form=tkinter.Label(window,text="Dados das Entradas",background="gray")
label_tamanho_cromossomo = tkinter.Label(window, text="Tamanho do Cromossomo:",background="gray")
label_tamanho_da_populacao = tkinter.Label(window, text="Tamanho da População:",background="gray")
label_probabilidade_de_cruzamento = tkinter.Label(window, text="Probabilidade de Cruzamento:",background="gray")
label_mutacao_probabilidade = tkinter.Label(window, text="Probabilidade de Mutação:",background="gray")
label_quantidade_geracoes = tkinter.Label(window, text="Quantidade de Geracoes:",background="gray")
label_tamanho_torneio =  tkinter.Label(window, text="Tamanho do Torneio:",background="gray")
label_tamanho_elitismo = tkinter.Label(window, text="Tamanho do Elitismo:", background="gray")
result_best_individual = tkinter.Label(window, text="Melhor Individuo Encontrado:")
result_function_value = tkinter.Label(window, text="Aptidão:")
melhor_geracao=tkinter.Label(window,text="Geração em que foi encontrado a melhor geração:")
resultadoreal=tkinter.Label(window,text="Valor Maximo da Função:")
porcentagem_de_erro=tkinter.Label(window,text="Porcentagem de erro entre o Máximo encontrado e o Maximo Real:")


form_tamanho_elitismo = tkinter.Entry()
#INPUTS
form_tamanho_cromossomo=tkinter.Entry()
form_tamanho_da_populacao=tkinter.Entry()
form_probabilidade_de_cruzamento=tkinter.Entry()
form_mutacao_probabilidade=tkinter.Entry()
form_quantidade_geracoes=tkinter.Entry()
form_tamanho_torneio=tkinter.Entry()

#LOCALIZAÇÃO
label_form.place(x=200,y=10)
label_tamanho_da_populacao.place(x=100,y=50)
form_tamanho_da_populacao.place(x=325,y=50)
label_tamanho_cromossomo.place(x=100,y=80)
form_tamanho_cromossomo.place(x=325,y=80)
label_probabilidade_de_cruzamento.place(x=100,y=110)
form_probabilidade_de_cruzamento.place(x=325,y=110)
label_quantidade_geracoes.place(x=100,y=140)
form_quantidade_geracoes.place(x=325,y=140)
label_mutacao_probabilidade.place(x=100,y=170)
form_mutacao_probabilidade.place(x=325,y=170)
label_tamanho_elitismo.place(x=100, y=200)
form_tamanho_elitismo.place(x=325, y=200)
result_best_individual.place(x=100, y=700)
result_function_value.place(x=100, y=750)
melhor_geracao.place(x=100,y=800)
resultadoreal.place(x=100,y=850)
porcentagem_de_erro.place(x=100,y=900)
#CHECKBOX
check_var = tkinter.BooleanVar()
check_torneio = tkinter.Checkbutton(window, text="Torneio", variable=check_var, command=toggle_torneio_visibility, background="gray")
check_torneio.place(x=100, y=230)
label_tamanho_torneio.place_forget()
form_tamanho_torneio.place_forget()

#RADIO BUTTON
crossover_var = StringVar(value="one_point")
radio_one_point = Radiobutton(window, text="Cruzamento de um ponto", variable=crossover_var, value="one_point", command=on_crossover_selection, background="gray")
radio_two_points = Radiobutton(window, text="Cruzamento de dois pontos", variable=crossover_var, value="two_points", command=on_crossover_selection, background="gray")
radio_one_point.place(x=100, y=260)
radio_two_points.place(x=300, y=260)

#SUBMIT
submit_button = tkinter.Button(window, text="Submit", command=submit_button_event)
submit_button.place(x=350, y=300)

#BOTAO PREENCHER
preencher_button=tkinter.Button(window,text="To Fill",command=fill_form)
preencher_button.place(x=350,y=350)

window.mainloop()