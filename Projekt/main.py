import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from copy import deepcopy
import os
from IPython.display import display

#-------------------------------------------------------Analiza i przygotowanie danych---------------------------------------------------------------------------------------------

#znajdowanie ścieżki do folderu
script_dir = os.path.dirname(__file__)
rel_path = "filmy.txt"
abs_file_path = os.path.join(script_dir, rel_path)

#wczytanie pliku z folderu projektu
data =  pd.read_csv(abs_file_path,delimiter='\t',usecols=["Id","Release_Date","Movie","Budget", "Domestic_Gross", "Worldwide_Gross"])

#usuwamy dwa filmy z listy, które zaburzają cały wykres, Avatar i Endgame to filmy, których sukces był zbudowany
# na zupełnie innych rzeczach niż tylko budżet 
data.drop(data.index[data['Movie'] == "Avengers: Endgame"].tolist(), axis=0, inplace=True)
data.drop(data.index[data['Movie'] == "Avatar"].tolist(), axis=0, inplace=True)

#Wybieramy z naszej bazy interesujące nas kolumny, budżet, zarobek w kraju produkcji, zarobek na świecie, oraz stworzenie nowej kolumny, czyli zarobek poza ojczyzną
data_temp = [ [int(i[3].replace(u'\xa0','').replace('$','').replace(',',''))/1000000, int(i[4].replace(u'\xa0','').replace('$','').replace(',',''))/1000000, int(i[5].replace(u'\xa0','').replace('$','').replace(',',''))/1000000] for i in data.values if i[5] != "$0" and i[4] != "$0" ]  
data_temp = [ i for i in data_temp if i[2] != 0 and i[0] <= 300 ]
data_tab_orginal = deepcopy(data_temp) 

data_counter = [[],[],[]]
temp = []

for i in data_temp:
    if not data_counter[0].__contains__(i[0]):
        data_counter[0].append(i[0])
        data_counter[1].append(i)
        data_counter[2].append(1)
    else:
        index = data_counter[0].index(i[0])
        data_counter[1][index][1] += i[1]
        data_counter[1][index][2] += i[2]
        data_counter[2][index] += 1

for i in range(len(data_counter[0])):
    temp.append(((data_counter[0][i]), data_counter[1][i][1]/data_counter[2][i], data_counter[1][i][2]/data_counter[2][i]))

data_tab_average = temp

# Wyświetlenie tabeli danych  (10 pierwszych elementów)
a = pd.DataFrame(data_tab_average,columns={0 : "Budżet",1 : "Dochód w kraju",2 : "Dochód na świecie"})
fig, axs = plt.subplots(1, 1)
data = np.random.random((10, 3))
columns = ("Budżet", "Dochód w kraju", "Dochód na świecie")
axs.axis('tight')
axs.axis('off')
the_table = axs.table(cellText=a.head(10).values, colLabels=columns, loc='center')
plt.show()


X = [i[0] for i in data_tab_average]                     #  X dla wykresów wykorzystujących wszystkie dane
X_U = [i[0] for i in data_tab_orginal if i[0]>i[2]]      # under, wybiera filmy, które na siebie nie zarobiły
X_A = [i[0] for i in data_tab_orginal if i[0]<=i[2]]     # above, wybiera filmy, które na siebie zarobiły 


Y_D = [i[1] for i in data_tab_average]                   # Y_D są to zarobione pieniądze z kraju, z którego pochodzi film
Y_W = [i[2] - i[1] for i in data_tab_average]            # Y_W są to zarobione pieniądze poza krajem, z którego pochodzi film
Y_W_D = [i[2] for i in data_tab_average]                 # Y_W_D są to zarobione pieniądze na całym świecie

Y_U = [i[2] for i in data_tab_orginal if i[0]>i[2]]      # Y_U są to zarobione pieniądze na całym świecie, dla filmów które na siebie nie zarobiły
Y_A = [i[2] for i in data_tab_orginal if i[0]<=i[2]]     # Y_A są to zarobione pieniądze na całym świecie, dla filmów które na siebie zarobiły

#Zamieniamy wcześniejsze listy na macierze, aby móć wykorzytsać operacje macierzowe wyliczenia najlepszych parametrów do modelu
X = np.array(X)

X_U = np.array(X_U)
X_A = np.array(X_A)

Y_D = np.array(Y_D)
Y_W = np.array(Y_W)
Y_W_D = np.array(Y_W_D)

Y_U = np.array(Y_U)
Y_A = np.array(Y_A)



#-------------------------------------------------------Rozwiązanie problemu------------------------------------------------------------------------------------------------

#Wyliczamy X_test i Y_pred, które bedą tworzyć prostą odpowiadającą najlepszym parametrom, Y_fill, które nie pozawala jeść błędowi ponieżej 0, oraz Q_3, 
#będące miarą błedu popełnianego przez model, parameters, które zwracają te wartości znalezione przez model

def Find_Best_Parameters(X: np.ndarray, Y: np.ndarray, coeff: np.ndarray):
    X_test = np.linspace(start=X.min(), stop=X.max(), num=300)
    Y_pred = np.zeros_like(X_test)
    Y_ = np.zeros_like(Y)                        # posłuży nam do wyliczenia Q_3
    parameters = []                                    

    for i, c in enumerate(coeff.ravel()[::-1]):
        Y_pred += (X_test ** i) * c
        Y_ += (X ** i) * float(c)
        parameters += [round(c, 4)]

    E = Y - Y_
    Q_1 = np.sum(E ** 2)
    Q_3 = np.sqrt(Q_1) / Y.size

    Y_fill = Y_pred - Q_3
    for i in range(Y_fill.size ):
        if Y_fill[i] < 0:
            Y_fill[i] = 0

    return X_test, Y_pred, Y_fill, Q_3, parameters

# Plot_Fig_List służy nam do dodania na wykres zarówno lini danych odnoszacych się do zysków w ojczyźnie, jak i tych poza nią, co da nam porównanie
#                               (X,Y,coeff,line color,fill color,plot label,scatter label)
def Plot_Fig_List(plot_set: list[(np.ndarray, np.ndarray, np.ndarray, str, str, str, str)], title: str, xlabel: str, ylabel: str) -> None:
        plt.figure(figsize=(10,16))
        for i in range(len(plot_set)):

            #Wykorzystujemy analityczny algorytm do estymacji parametrów modelu liniowego
            _X = plot_set[i][0][np.newaxis, :] ** plot_set[i][2]
            _Y = plot_set[i][1][np.newaxis, :]
            _T = np.linalg.inv(_X@_X.transpose())@_X@_Y.transpose()

            X_test, Y_pred, Y_fill, Q_3, parameters = Find_Best_Parameters(plot_set[i][0],plot_set[i][1],_T)

            func_str = ""
            for j in range(len(parameters)):
                func_str +=f"({parameters[j]}) * x^{j} + "

            plt.plot(X_test, Y_pred, color=plot_set[i][3], label=f"{plot_set[i][5]} y_{i}={func_str[:-2]}")    # zamodelowana prosta
            plt.fill_between(X_test, Y_fill, Y_pred + Q_3, color=plot_set[i][4], alpha=0.2)                    # dodanie lini błędu
            plt.scatter(plot_set[i][0], plot_set[i][1], label=plot_set[i][6], alpha=0.5)                       # dodanie na wykres filmów

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title)
        plt.legend()
        plt.show()


plt.figure(figsize=(10,16))
plt.scatter(X_U, Y_U, label='Filmy, które na siebie nie zarobiły', alpha=0.7)
plt.scatter(X_A, Y_A, label='Filmy, które na siebie zarobiły', alpha=0.7)
plt.xlabel('x - Budżet filmu (mln USD)', fontsize=14)
plt.ylabel('y - Dochód z filmu (mln USD)', fontsize=14)
plt.title(f"Filmy, które na siebie zarobiły stanowią {len(X_A)/(len(X_A)+len(X_U))*100}% wszystkich filmów")
plt.legend()
plt.show()

# funkcje bazowe - wielomimany n-tego rzędu, np:
# [[1], [0]] - liniowa
# [[2], [1], [0]] - 2nd order,
# [[3], [2], [1], [0]] - 3rd order

Plot_Fig_List([(X,Y_W_D,[[1], [0]],'green','white','',''),(X,Y_W_D,[[2], [1], [0]],'grey','white','',''), (X,Y_W_D,[[3], [2], [1], [0]],'grey','white','',''), (X,Y_W_D,[[4], [3], [2], [1], [0]],'grey','white','','') ],'Zależność między budżetem filmu, a jego dochodem', 'x - Budżet filmu (mln USD)', 'y - Dochód z filmu (mln USD)')
Plot_Fig_List([ (X,Y_W_D,[[1], [0]],'yellow','red','estymowany trend dochodu ze świata','dochód filmu na świecie')],'Zależność między budżetem filmu, a jego dochodem', 'x - Budżet filmu (mln USD)', 'y - Dochód z filmu')
Plot_Fig_List([(X,Y_D,[[1], [0]],'yellow','blue','estymowany trend dochodu z ojczyzny','dochód filmu w ojczyźnie'),(X,Y_W,[[1], [0]],'grey','blue','estymowany trend dochodu zza granicy','dochód filmu za granicą')],'Zależność między budżetem filmu, a jego dochodem', 'x - Budżet filmu (mln USD)', 'y - Dochód z filmu (mln USD)')


