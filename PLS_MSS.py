import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn import preprocessing

window = 20

csv_file = open("ch1.csv", "r")

h = next(csv.reader(csv_file))

ind = 0

pos = []

for row in csv.reader(csv_file):

    if float(row[0]) > 150 and len(pos) == 0:
        pos.append(ind)

    if float(row[0]) > 210 and len(pos) == 1:
        pos.append(ind)

    ind += 1


##
csv_file = open("ch1.csv", "r")

h = next(csv.reader(csv_file))

ch1 = []

for row in csv.reader(csv_file):

    ch1.append([float(row[i]) for i in range(len(row))])

##
csv_file = open("ch2.csv", "r")

h = next(csv.reader(csv_file))

ch2 = []

for row in csv.reader(csv_file):

    ch2.append([float(row[i]) for i in range(len(row))])

##
csv_file = open("ch3.csv", "r")

h = next(csv.reader(csv_file))

ch3 = []

for row in csv.reader(csv_file):

    ch3.append([float(row[i]) for i in range(len(row))])

##
csv_file = open("ch4.csv", "r")

h = next(csv.reader(csv_file))

ch4 = []

for row in csv.reader(csv_file):

    ch4.append([float(row[i]) for i in range(len(row))])


descriptors = []

for ii in range(30):

    descriptors_each = []

    ##ch1

    for iii in range(int(1200/window) -  1):

        descriptors_each.append(ch1[pos[0] + window * (iii + 1)][ii + 1] - ch1[pos[0]][ii + 1])


    ##ch2
    for iii in range(int(1200/window) -  1):

        descriptors_each.append(ch2[pos[0] + window * (iii + 1)][ii + 1] - ch2[pos[0]][ii + 1])
        

    ##ch3
    for iii in range(int(1200/window) -  1):

        descriptors_each.append(ch3[pos[0] + window * (iii + 1)][ii + 1] - ch3[pos[0]][ii + 1])
    

    ##ch4
    for iii in range(int(1200/window) -  1):
        
        descriptors_each.append(ch4[pos[0] + window * (iii + 1)][ii + 1] - ch4[pos[0]][ii + 1])


    descriptors.append(descriptors_each)


ss = preprocessing.StandardScaler()
descriptors_stan = ss.fit_transform(np.array(descriptors))


#Objs
csv_file = open("concentration.csv", "r")

h = next(csv.reader(csv_file))

concentrations = []

for row in csv.reader(csv_file):

    concentrations.append([float(row[i + 1]) for i in range(3)])


X_train = np.array(descriptors_stan)
y_train = np.array(concentrations)


kf = KFold(n_splits = 10, shuffle = True, random_state = 1403)
kf.get_n_splits(X_train)


true_y = []
pred_y = []

for train_index, test_index in kf.split(X_train):

    pls = PLSRegression()
    alpha_list = list(range(1, 16))
    params = {'n_components': alpha_list}

    pls_GSCV = GridSearchCV(pls, params, cv = 5, scoring='neg_mean_squared_error')
    pls_GSCV.fit(np.array(X_train)[train_index], np.array(y_train)[train_index])

    pls = PLSRegression(n_components=pls_GSCV.best_params_['n_components'])

    pls.fit(np.array(X_train)[train_index], np.array(y_train)[train_index])

    true_y.extend(list(np.array(y_train)[test_index]))
    pred_y.extend(list(pls.predict(np.array(X_train)[test_index])))



for iii in range(3):

    true_y_one = [row[iii] for row in true_y]
    pred_y_one = [row[iii] for row in pred_y]

    plt.rcParams["font.size"] = 15
    fig = plt.figure(figsize=(5,5))

    x_max = 108
    x_min = -8

    margin = 0

    ax = fig.add_subplot(1, 1, 1)
    ax.plot([x_min - margin, x_max + margin], [x_min - margin, x_max + margin], color="black", linestyle="dashed")
    plt.xlim(x_min - margin, x_max + margin)
    plt.ylim(x_min - margin, x_max + margin)

    ax.text(0.96, 0.04, "R2=" + str(round(r2_score(true_y_one,pred_y_one),4)), 
            horizontalalignment='right', transform=ax.transAxes, fontsize = 16)
    ax.scatter(true_y_one, pred_y_one, alpha = 0.7, s= 100)
    plt.ylabel('Prediction [%]', fontsize=14)
    plt.xlabel('Ground truth [%]', fontsize=14)
    plt.savefig('Scatter_PLS' + str(iii) + '.pdf', bbox_inches='tight')
    plt.close()
    
    print("PLS", iii)
    print("MSE =", mean_squared_error(true_y_one,pred_y_one))
    print("R2 =", r2_score(true_y_one,pred_y_one))

