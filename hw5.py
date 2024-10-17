import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot(filename):
    df = pd.read_csv(filename)
    plt.plot(df['year'], df['days'], marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Frozen Days')
    plt.title('Number of Frozen Days vs. Year')
    plt.savefig('data_plot.jpg')

def normalize(df):
    years = df['year'].values
    days = df['days'].values
    m = np.min(years)
    M = np.max(years)
    normalized_years = (years - m) / (M - m)
    X_normalized = np.column_stack((normalized_years, np.ones(len(normalized_years))))
    print("Q3:")
    print(X_normalized)
    return X_normalized, days, m, M

def closed(X_normalized, days):
    Y = days.reshape(-1, 1)
    weights = np.linalg.inv(X_normalized.T @ X_normalized) @ X_normalized.T @ Y
    print("Q4:")
    print(weights)
    return weights

def gradient_descent_with_loss(X_normalized, days, learning_rate, iterations):
    n = len(days)
    weights = np.zeros((2, 1))  
    losses = []  

    print("Q5a:")
    for t in range(iterations):
        if t % 10 == 0:
            print(np.array([weights[0, 0], weights[1, 0]]))

        y_hat = X_normalized @ weights
        gradient = (1 / n) * X_normalized.T @ (y_hat - days.reshape(-1, 1))
        weights = weights - learning_rate * gradient
        loss = (1 / (2 * n)) * np.sum((y_hat - days.reshape(-1, 1)) ** 2)
        losses.append(loss)
    
    plt.clf()   
    plt.plot(range(iterations), losses)  
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss vs Iterations')
    plt.savefig('loss_plot.jpg')

    print("Q5b:", learning_rate)
    print("Q5c:", iterations)

    return weights



def predict_2023(weights, m, M):
    x_new = 2023
    w = weights[0, 0]
    b = weights[1, 0]
    x_new_normalized = (x_new - m) / (M - m)
    y_hat = w * x_new_normalized + b
    print("Q6: " + str(y_hat))

def interpret(weights):
    w = weights[0, 0]
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="
    print("Q7a: " + symbol)
    if symbol == ">":
        interpretation = "w > 0 indicates that the number of frozen days increases as the years increase."
    elif symbol == "<":
        interpretation = "w < 0 indicates that the number of frozen days decreases as the years increase."
    else:
        interpretation = "w = 0 indicates no relationship between the number of frozen days and the years."
    print("Q7b: " + interpretation)

def no_freeze(weights, m, M):
    w = weights[0, 0]
    b = weights[1, 0]
    if w != 0:
        x_star = m + (M - m) * (-b / w)
    else:
        x_star = float('inf')
    print("Q8a: " + str(x_star))
    print("Q8b: This prediction assumes a linear trend will continue indefinitely. However, many external factors like climate change, regional weather patterns, and unexpected environmental shifts could cause the actual year for no freezing days to deviate from this estimate.")

if __name__ == "__main__":
    filename = sys.argv[1]  
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])

    df = pd.read_csv(filename)

    plot(filename)

    X_normalized, days, m, M = normalize(df)
    weights_q4 = closed(X_normalized, days)
    weights_gd = gradient_descent_with_loss(X_normalized, days, learning_rate, iterations)
    predict_2023(weights_q4, m, M)
    interpret(weights_q4)
    no_freeze(weights_q4, m, M)

