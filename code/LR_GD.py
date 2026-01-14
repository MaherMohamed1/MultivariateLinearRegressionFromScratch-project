import numpy as np 
import pandas as pd
import seaborn as sns
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocessing():    
    # data = load_diabetes()  # if we want to use general dataset for test
    # X, t = data.data, data.target 
    df = pd.read_csv(r"C:\Users\Maher\Desktop\dataset_200x4_regression.csv")
    data = df.to_numpy()
    X, t = data[: , :3], data[: , -1] #
    X = MinMaxScaler().fit_transform(X)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X, t, df

def normal_equations_solution(X, y):
    # (X.T * X)^(-1) * X.T * y
    XT = X.T
    return inv(XT.dot(X)).dot(XT.dot(y))


def cost_f(X, t,weights):
    N = X.shape[0]
    y = np.dot(X,weights) 
    err = y - t 
    cost = err.T @ err / (2 * N)
    return cost
    
def f_deriv(X, t, weights):
    N = X.shape[0]
    y = np.dot(X,weights)
    err = y - t
    gradient = X.T @ err / N
    return gradient

def gradient_descent_linear_regression(X, t, step_size=0.5, precision=0.00001, max_iter=500):
    # if we need to start from random values
    # features = X.shape[1]
    # cur_weights = np.random.rand(features)
    cur_weights =  np.array([1., 1., 1., 1.]) 
    last_weights = cur_weights + 100 * precision
    
    print(f"Initial random cost: {cost_f(X, t, cur_weights)}")
    
    iter = 0
    cost_history = []
    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()
        gradient = f_deriv(X, t, cur_weights) 
        cur_weights -= gradient * step_size
        
        cost_history.append(cost_f(X, t, cur_weights))
        
        iter += 1
    print(f'Optimal error: {cost_f(X, t, cur_weights)}') 
    
    return cur_weights, cost_history

def LR_using_scikit(X, t):
    x = X[:, 1:]
    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(x, t)
    optimal_w = np.array([model.intercept_, *model.coef_])
    print("Scikit parameters: ", optimal_w)
    pred_t = model.predict(x)
    error = mean_squared_error(t, pred_t) / 2
    print("error:", error)
    
    

def visualize_results(df, cost_history):
    sns.pairplot(df, x_vars=['Feat1', 'Feat2', 'Feat3'], y_vars='Target', height=4, aspect=1, kind='scatter') 
    plt.show()

    correlation_matrix = df.corr().round(2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    
    plt.figure(figsize=(8, 6))
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    plt.grid(True)
    plt.show()
    
if __name__=="__main__":
    
    X, t, df = preprocessing()
    LR_using_scikit(X, t)
    
    optimal_weights, cost_history = gradient_descent_linear_regression(X, t)
    # print("optimal_weights:",optimal_weights)
    print("normal equations solution:",normal_equations_solution(X, t))
    visualize_results(df, cost_history)
    



    