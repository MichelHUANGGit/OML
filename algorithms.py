import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

def Newton2(w, function_class, max_iter=10, epsilon=1e-10):
    # Initialization 
    iter=0
    f = function_class()
    # Store gradients norm throughout the iterations
    grad_logs = [np.linalg.norm(f.nabla(w),2)]
    # Budget of max_iter or until l2 norm of the gradient is below tolerance epsilon
    while(iter<max_iter and grad_logs[-1] > epsilon):
        iter +=1
        print("Iteration :", iter)
        inv_nabla2_f = np.linalg.inv(f.nabla2(w))
        nabla_f_w = f.nabla(w)
        direction = (inv_nabla2_f @ nabla_f_w)
        w = w - direction
        grad_logs.append(np.linalg.norm(nabla_f_w,2))
    # Return final w, and the gradients norm history
    return w, grad_logs

def GlobalNewton(w, function_class, max_iter, tolerance=1e-10, **Armijo_kwargs):
    # Initialization
    identity = np.identity(w.shape[0])
    f = function_class()
    # Grad_logs stores he history of gradients norms
    grad_logs = [np.linalg.norm(f.nabla(w),2)]
    k=0
    nabla_f_w = f.nabla(w)

    while(k<max_iter and grad_logs[-1]>tolerance):
        k+=1
        # compute eigenvalue
        nabla2_f_w = f.nabla2(w)
        lambda_k = 2*max( -np.min(np.linalg.eigvals(nabla2_f_w)) , 1e-10 )
        # direction
        nabla_f_w = f.nabla(w)
        f_w = f(w)
        direction = -np.linalg.inv( nabla2_f_w + lambda_k*identity ) @ nabla_f_w
        # Line serach
        alpha = ArmijoLineSearch(f, f_w, nabla_f_w, w, direction, **Armijo_kwargs)
        # W update
        w = w + alpha * direction
        grad_logs.append(np.linalg.norm(nabla_f_w, 2))
    return w, grad_logs

def ArmijoLineSearch(f, f_w, nabla_f_w, w, direction, c, theta, initial_alpha=1):
    # To avoid redundant recomputations, we take nabla_f(w), f(w) as a parameter
    alpha = initial_alpha
    while( f(w + alpha*direction) > f_w + c*alpha*(direction.T)@(nabla_f_w) ):
        alpha *= theta
    return alpha

def Newton_LineSearch(w, function_class, max_iter=10, tolerance=1e-10, **Armijokwargs):
    f = function_class()
    grad_logs = [np.linalg.norm(f.nabla(w),2)]
    k=0
    early_exit = False
    while(k<max_iter and grad_logs[-1]>tolerance):
        k +=1
        # Direction
        # If Singular hessian, exit the while loop and the algorithm stops
        try :
            inv_nabla2_f = np.linalg.inv(f.nabla2(w))
        except :
            early_exit = True
            break
        nabla_f_w = f.nabla(w)
        direction = -(inv_nabla2_f @ nabla_f_w)
        # Armijo line search
        f_w = f(w)
        alpha = ArmijoLineSearch(f, f_w, nabla_f_w, w, direction, **Armijokwargs)
        # Update
        w = w + alpha*direction
        grad_logs.append(np.linalg.norm(nabla_f_w, 2))
        
    logs = {
        "iterations" : k,
        "function evaluations" : f.func_eval,
        "gradient evaluations" : f.grad_eval,
        "successful" : grad_logs[-1]<=tolerance,
        "early exit" : early_exit
    }
    return w, grad_logs, logs

def QuasiNewton_BFGS(w, function_class, max_iter=10, tolerance=1e-10, use_line_search=True, **Armijokwargs):

    # Initialization
    d = w.shape[0]
    H = np.identity(d)
    k = 0
    f = function_class()
    nabla_f_w = f.nabla(w)
    grad_logs = [np.linalg.norm(nabla_f_w,2)]

    # While the gradient norm is close to 0 with a certain tolerance and the number of iterations hasn't passed the iterations budget
    while(k<max_iter and grad_logs[-1]>tolerance):

        # Direction
        direction = -(H @ nabla_f_w)

        # Armijo line search
        f_w = f(w)
        alpha = ArmijoLineSearch(f, f_w, nabla_f_w, w, direction, **Armijokwargs) if use_line_search else 1

        # Updating w, we store the old w to compute s
        old_w = w.copy()
        w = w + alpha * direction

        # Computing v, s then H
        s = w - old_w
        old_nabla_f_w = nabla_f_w.copy()
        nabla_f_w = f.nabla(w)
        v = nabla_f_w - old_nabla_f_w
        H = get_H(H, v.reshape(d,1), s)

        # Setting variables for the next iteration
        k += 1
        grad_logs.append(np.linalg.norm(nabla_f_w,2))
    
    logs = {
        "iterations" : k,
        "function evaluations" : f.func_eval,
        "gradient evaluations" : f.grad_eval,
        "successful" : grad_logs[-1]<=tolerance,
        "early exit" : False
    }
    return w, grad_logs, logs

def get_H(old_H, v, s):
    '''
    inputs s,v are expected to be of shape (d, 1)
    '''
    d = s.shape[0]
    sv = s.T @ v # scalar
    if sv <= 0:
        return old_H
    else :
        vs = v @ s.T # (d,d) matrix
        right = np.identity(d) - (vs/sv)
        return right.T @ old_H @ right + (s@(s.T)/sv)


def get_limited_H(m, s : deque, v : deque, d):
    '''
    We implement this algorithm differently from the Algorithm (1). Here we use deque objects
    And we also move our index up, whereas the index in Algorithm (1) moves down
    We compute s and v outside of the function to avoid redundant recomputation of nabla_f(w)
    '''
    # s stores the history of the m lastest s values
    # v stores the history of the m lastest v values
    H = np.identity(d)
    for i in range(0, min(len(s),m), 1): #if len(s) < m, we don't have full history -> only do len(s) iterations
        sv = s[i].T @ v[i]
        if sv > 0 :
            vs = v[i] @ s[i].T
            right = np.identity(d) - (vs/sv)
            H = right.T @ H @ right + (s[i]@(s[i].T)/sv)
    return H

def QuasiNewton_LBFGS(w, function_class, m=5, use_line_search=True, max_iter=10, tolerance=1e-10, **Armijokwargs):

    # Initialization
    d = w.shape[0]
    H = np.identity(d)
    k = 0
    f = function_class()
    nabla_f_w = f.nabla(w)
    # grad_logs stores the values of the gradient norm throughout the iterations
    grad_logs = [np.linalg.norm(nabla_f_w, 2)]
    # Storing the values of v and s
    v_queue = deque(maxlen=m)
    s_queue = deque(maxlen=m)

    # While the gradient norm is close to 0 with a certain tolerance and the number of iterations hasn't passed the iterations budget
    while(k<max_iter and grad_logs[-1]>tolerance):

        # Direction
        direction = -(H @ nabla_f_w)

        # Armijo line search
        f_w = f(w)
        alpha = ArmijoLineSearch(f, f_w, nabla_f_w, w, direction, **Armijokwargs) if use_line_search else 0.01

        # Updating w
        old_w = w.copy()
        w = w + alpha * direction

        # Computing v, s then H
        s_queue.append((w - old_w).reshape(d,1))
        old_nabla_f_w = nabla_f_w.copy()
        nabla_f_w = f.nabla(w)
        v_queue.append((nabla_f_w - old_nabla_f_w).reshape(d,1))
        H = get_limited_H(m, s_queue, v_queue, d)

        # Setting variables for the next iteration
        k += 1
        grad_logs.append(np.linalg.norm(nabla_f_w, 2))
    
    logs = {
        "iterations" : k,
        "function evaluations" : f.func_eval,
        "gradient evaluations" : f.grad_eval,
        "successful" : np.linalg.norm(nabla_f_w, ord=2)<=tolerance,
        "early exit" : False,
    }
    return w, grad_logs, logs

def run_algorithm(
        initial_ws,
        algorithm_fn,
        fn_class,
        **algo_kwargs,
    ):
    '''
    Runs the given algorithm once on every starting point given -> initial_ws
    '''

    summary = {
        "avg_w_reached" : [],
        "avg_grad_norm_reached" : [],
        "iterations" : [],
        "func_evals" : [],
        "grad_evals" : [],
        "success" : [],
        "early_exits" : []
    }

    # For every w, run the algorithm, store a summary and plot the result
    for initial_w in initial_ws:
        w, grad_norm, logs = algorithm_fn(initial_w, fn_class, **algo_kwargs)
        summary["avg_w_reached"].append(w)
        summary["avg_grad_norm_reached"].append(grad_norm[-1])
        summary["iterations"].append(len(grad_norm)-1)
        summary["func_evals"].append(logs["function evaluations"])
        summary["grad_evals"].append(logs["gradient evaluations"])
        summary["success"].append(logs["successful"])
        summary["early_exits"].append(logs["early exit"])
        plt.plot(grad_norm)
    
    plt.yscale('log')
    plt.ylabel("norm nabla f(w)")
    plt.xlabel("iterations")
    plt.title("Convergence plot")
    plt.show()

    # Verbose
    print("Average w reached :\n", np.mean(summary["avg_w_reached"], axis=0))
    print("Average gradient norm reached :", np.mean(summary["avg_grad_norm_reached"]))
    print("Average iterations :", np.mean(summary["iterations"]))
    print("Average function evaluations :", np.mean(summary["func_evals"]))
    print("Average gradient evaluations :", np.mean(summary["grad_evals"]))
    print("Successful convergence :", sum(summary["success"]),"/ %d" % initial_ws.shape[0])
    print("Early exits :", sum(summary["early_exits"]))
    

if __name__ == '__main__' :
    w01 = np.array([[-1.2],
                [1.]])
    w02 = np.array([[0],
                    [1/200 + 1e-12]])
    print(w01, "\n", w02)