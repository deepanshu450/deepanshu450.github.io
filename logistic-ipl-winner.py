# IMPORTING libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# import seaborn as sns

# sigmoid function: provides hypothesis & prediction
# Params: X - Input data with feature vars. & theta: weight
def sigmoid_func(X, theta):
    z = np.dot(X, theta)
    return 1 / (1 + np.exp(-z))

# gradient descent function: provides optimization in Logistic Regression by minimizing cost function
# Params: X - training data with feature vars. , y: training data with target var & h: val of sigmoid func. 
def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y))/y.shape[0]

# loss func: overall loss b/w actual val & predicted val
def loss(pred_y1, y):
    return (-y * np.log(pred_y1) - (1 - y) * np.log(1 - pred_y1)).mean()

# update theta function: update theta to best suitable theta
# Params: theta: weight, alpha: learning rate & gradient: val of gradient descent func.
def update_theta(theta, alpha, gradient):
    return theta - alpha*gradient

# train the model
def train(itr, l_rate, theta, x_train, y_train):
    optimized_thetas = [] # list of best suitable thetas at diff learning rate

    for rate in l_rate:
        costs = []
        for i in range(itr):
            y1 = sigmoid_func(x_train, theta)
            costs.append(loss(y1, y_train))
            gradient = gradient_descent(x_train, y1, y_train)
            theta = update_theta(theta, rate, gradient)

        optimized_thetas.append(theta)
  
        # ---------- 3D plot----------------
        # fig = plt.figure()
        # ax = plt.axes(projection ='3d')
  
        # z = costs
        # x = range(0,itr)
        # y = range(0,itr)
  
        # # plotting
        # ax.set(xlabel='Iterations', ylabel='Iterations', zlabel='Loss')
        # ax.plot3D(x, y, z, 'green')
        # ax.set_title('Iterations v/s Loss')
        # plt.show()

        # ---------- 2D plot----------------
        # plt.title('Iterations v/s Loss')
        # plt.xlabel('iterations')
        # plt.ylabel('Loss')
        # plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
        # plt.plot(range(0,itr), costs)
        # plt.show()

    return optimized_thetas

# predict the data using trained model
def predict(itr, l_rate, thetas, x_test, y_test):
    index = 0
    print("------------------")
    print("Iterations:", itr)
    print("------------------")
    for theta in thetas:
        pred_y = sigmoid_func(x_test, theta) # predicting 'x_test' data using theta 

        # converting large decimal values to .2 & forming data frame
        pred_y_round_off = pd.DataFrame(np.around(pred_y, decimals=2))

        # classifying predicted vals in 0 & 1 based on threshold value: 0.5
        pred_y_round_off.loc[pred_y_round_off[0] < 0.5, "predicted"] = 0
        pred_y_round_off.loc[pred_y_round_off[0] >= 0.5, "predicted"] = 1
        pred_y_round_off['winner'] = pd.DataFrame(y_test)

        # calculating accuracy by comparing actual & predicted vals
        accuracy = (pred_y_round_off.loc[pred_y_round_off['predicted'] == pred_y_round_off['winner']].shape[0]/pred_y_round_off.shape[0])*100

        print("Learning rate:", l_rate[index])
        index += 1
        print("Accuracy: ", accuracy)
        print("\n")

# loading dataset
dataFrame = pd.read_csv("Dataset/encoded_dataset.csv")  # dataset

# mapping
dataFrame.loc[dataFrame["teampoint_diff"] < 0, "teampoint_diff"] = 0
dataFrame.loc[dataFrame["teampoint_diff"] >= 0, "teampoint_diff"] = 1

df = dataFrame.copy()

feature_vars = ['team1', 'powerplay_runs_team1', 'powerplay_wickets_team1', 'team2', 'powerplay_runs_team2',
                 'powerplay_wickets_team2', 'homeground_advantage', 'team1_toss_win', 'teampoint_diff']

target_var = 'team1_win'

X_features = df[feature_vars]
y_target = df[target_var]

learning_rate = [0.001, 0.01, 0.1] #, 0.5, 1] #0.0001, 0.001, 0.01, 
iterations = 5000

intercept = np.ones((X_features.shape[0], 1))
X_features = np.concatenate((intercept, X_features), axis=1)

# splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_target, test_size=1/3, random_state=0)
Y_train = Y_train.to_numpy() # converting series to array
Y_test = Y_test.to_numpy() # converting series to array

theta = np.zeros(X_train.shape[1]) # init. theta

# training the model
model = train(iterations, learning_rate, theta, X_train, Y_train)
# predicting
predict(iterations, learning_rate, model, X_test, Y_test)
