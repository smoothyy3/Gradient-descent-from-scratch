from math_functions import *

class gradient_descent:
    def __init__(self, learn_rate, max_iter, tolerance):
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.history = []

    def fit(self, cost_function, params):
        # to make the gradient descent applicable for different functions you have to pass the initial parameters
        if not isinstance(params, list) or not params:
            raise ValueError("You must provide a list of initial parameters.")
        for iteration in range(self.max_iter):  # Gradient Descent Iterationen
            gradients = []

            # calc gradients of ervery parameter
            for paramIndex in range(len(params)):
                grad = partial_derivative(cost_function, params, paramIndex)
                gradients.append(grad)

            # check convergence
            if max(abs(g) for g in gradients) < self.tolerance:
                break
            
            # adjust parameter according to gradient descent logic
            for j in range(len(params)):
                params[j] = params[j] - self.learn_rate * gradients[j]
            # get history of parameter updates
            self.history.append((iteration, params[:], cost_function(params)))

        # safe params for predict method        
        self.optimized_params = params
        return params
    
    def predict(self, cost_function):
        if self.optimized_params is None:
            raise ValueError("You must call fit() before predict().")
        return cost_function(self.optimized_params)
    
# Kostenfunktion (Ziel: x → 3)
def cost_function(x):
    return (x[0] - 3) ** 2  # Quadratische Funktion mit Minimum bei x=3

# Initialisierung des Algorithmus
gd = gradient_descent(learn_rate=0.1, max_iter=100, tolerance=1e-6)

# Starte mit einem zufälligen Wert für x, z. B. x = 10
initial_params = [10]

# Starte das Training
optimized_params = gd.fit(cost_function, initial_params)

# Zeige das Ergebnis
print("Optimized Parameters:", optimized_params)
print("Final Cost:", cost_function(optimized_params))
print("History:", gd.history[-5:])  # Zeigt die letzten 5 Iterationen