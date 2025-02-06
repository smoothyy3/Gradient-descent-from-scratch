def mean(lst) -> float:
    if not lst:
        raise ValueError("List cannot be empty")
    
    result = 0
    for x in lst:
        result += x
    return result / len(lst)

# ERROR METRICS: MEAN SQUARED ERROR (MSE)
def MSE(y_true: list, y_pred: list) -> float:
    if len(y_pred) == 0 or len(y_true) == 0:
        raise ValueError("Lists cannot be empty")
    
    if len(y_pred) != len(y_true):
        raise ValueError("Lists must have the same dimension")
    
    errorSum = 0.0

    for i in range(len(y_true)): 
        errorSum += (y_true[i] - y_pred[i])**2

    return errorSum / len(y_pred)

def MSE_derivative(y_true: list, y_pred: list):
    if not y_pred or not y_true:
        raise ValueError("Lists cannot be empty")
    
    if len(y_pred) != len(y_true):
        raise ValueError("Lists must have the same dimension")
    
    errorSum = 0.0

    for i in range(len(y_true)):
        errorSum += (y_true[i] - y_pred[i])

    return (2 / len(y_true)) * errorSum

# GENERAL DERIVATIVE FUNCTIONS
def derivative(f, x, h=1e-5):
    # using the symmetric difference quotient
    return (f(x+h) - f(x-h)) / (2*h)

# PARTIAL DERIVATIVES FOR MULTI-VARIABLE FUNCTIONS
def partial_derivative(f, paramsX, paramIndex, h = 1e-5):
    # copy original parameters
    paramsX_plus = paramsX[:]
    paramsX_minus = paramsX[:]

    # change parameters at index position
    paramsX_plus[paramIndex] += h
    paramsX_minus[paramIndex] -= h

    return (f(paramsX_plus) - f(paramsX_minus)) / (2 * h)