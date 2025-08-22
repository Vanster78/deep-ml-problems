import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    x = x0.copy()

    g = np.zeros_like(x0)
    gn = np.zeros_like(x0)

    for i in range(num_iterations):
        g_i = grad(x)
        gn_i = g_i ** 2

        g = g * beta1 + g_i * (1 - beta1)
        gn = gn * beta2 + gn_i * (1 - beta2)
        
        m = g / (1 - beta1 ** (i+1))
        v = gn / (1 - beta2 ** (i+1))
        
        x -= learning_rate * m / (v ** 0.5 + epsilon)
    
    return x

if __name__ == "__main__":
    def objective_function(x):
        return x[0]**2 + x[1]**2

    def gradient(x):
        return np.array([2*x[0], 2*x[1]])


    x0 = np.array([1.0, 1.0])
    x_opt = adam_optimizer(objective_function, gradient, x0)

    print("Optimized parameters:", x_opt)
