import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import norm # Euclidean norm

from tqdm import tqdm
from collections import deque
from algorithms import get_H, get_limited_H

def SubNewton(
        initial_w,
        loss_class,
        grad_batch_size,
        grad2_batch_size, 
        max_epochs,
        ground_truth=None,
        tolerance=1e-5,
        verbose=True,
        seed=1,
        with_remplacement=True,
        use_line_search=False,
        step_size = 1e-2,
        alpha_batch_L = 0,
        **Armijo_kwargs,
    ):  
    '''
    Subsampling second-order Newton algorithm.
    Parameters
    ----------
    loss_class : loss function instance, the class __call__ method has to be defined as the loss function.
                 It also has to have a grad() and grad2() method that compute the gradient and the hessian

    ground_truth : If known, solution w of the problem, used to compute a condition of early stopping when converging before max_epochs,
                   and store the distance of w to solution w* throughout the epochs.
                   Else, uses the norm of the gradient.

    use_line_search : whether or not to use line_search. If set to True, overrides step_size and alpha_batch_L

    alpha_batch_L : If > 0, computes the batch gradient Lipschitz constant at every iteration,
                    then sets the stepsize to alpha_batch_L/batch_L. Overrides step_size if > 0, else uses line search or step_size. 

    step_size : Either a value of a constant stepsize strategy, or a function of the k (e.g. decreasing step size 1/sqrt(k))

    **Armijo_kwargs : keywords parameters of Armijo Line-search function (c=1e-4, theta=0.5, initial_alpha=1)
    '''

    # Initialization
    np.random.seed(seed)
    N = loss_class.N
    d = loss_class.d
    w = initial_w.copy()
    epoch = 0
    iter_per_epoch = round(N/grad_batch_size)

    # loss_logs stores the history of the (full) loss
    loss_logs = [loss_class(initial_w, indexes=np.arange(N))]

    # If known, norm_logs stores the history of the distance between our w and the solution w*
    # Else stores the norm of the gradients (expensive)
    norm_logs = [norm(w-ground_truth,2)] if ground_truth is not None else [norm(loss_class.grad(initial_w, np.arange(N)),2)]

    if verbose:
        print(' | '.join([name.center(8) for name in ["epoch", "loss", "norm"]]))
        msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])
    
    try :
        while (epoch<max_epochs and norm_logs[-1]>tolerance):
            # Beginning of one epoch
            # ----------------------
            iterator = tqdm(range(iter_per_epoch), desc=msg) if verbose else range(iter_per_epoch)
            for iter in iterator:
                # The randomy selected samples 
                grad_idx = np.random.choice(np.arange(N), size=grad_batch_size, replace=with_remplacement)
                grad2_idx = np.random.choice(np.arange(N), size=grad2_batch_size, replace=with_remplacement)

                # Computing the gradient of the selected batch
                batch_grad = loss_class.grad(w, grad_idx)

                # Computing the Hessian matrix of the selected batch
                batch_grad2 = loss_class.grad2(w, grad2_idx)

                # Direction
                direction = -(np.linalg.inv(batch_grad2) @ batch_grad)

                # Computing the batch loss (used in line search to avoid redundant recomputation)
                batch_loss = loss_class(w, grad_idx)

                # Line search
                if use_line_search :
                    alpha = BatchArmijoLineSearch(
                        loss_fn=loss_class.__call__,
                        batch_loss=batch_loss,
                        batch_grad=batch_grad,
                        w=w, 
                        direction=direction,
                        indexes=grad_idx,
                        **Armijo_kwargs
                    ) 
                # computing L off of a batch at each iteration
                elif alpha_batch_L>0 :
                    alpha = alpha_batch_L / loss_class.get_batch_L(grad_idx)
                # decreasing step_size as a function of iter
                elif callable(step_size) :
                    alpha = step_size(iter)
                # constant step size
                else :
                    alpha = step_size

                # updating w
                w = w + alpha * direction
            # End of the epoch
            # ----------------
            if ground_truth is not None :
                norm_logs.append(norm(w-ground_truth, 2))
            else :
                norm_logs.append(norm(loss_class.grad(w, np.arange(N)),2))
            loss_logs.append(loss_class(w, np.arange(N)))
            epoch += 1
            if verbose :
                msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])
                        
    # If the user decides to interrupt, stores the results                    
    except KeyboardInterrupt :
        return w, loss_logs, norm_logs
    
    return w, loss_logs, norm_logs

def BatchArmijoLineSearch(loss_fn, batch_loss, batch_grad, w, direction, indexes, c=1e-4, theta=0.5, initial_alpha=1):
    # Takes batch_loss (loss evaluated of the batch), batch_grad (gradient loss of the batch) as a parameter to avoid recomputations
    # indexes is to compute the loss of (w + alpha*direction)
    alpha = initial_alpha
    while (loss_fn(w + alpha*direction, indexes) >= batch_loss + c*alpha*direction.dot(batch_grad)): 
        alpha *= theta
        # Safety condition
        if alpha<1e-15:
            print(f'alpha too low using {initial_alpha}')
            return initial_alpha
    return alpha

def BatchBFGS(
        initial_w,
        loss_class,
        grad_batch_size,
        max_epochs,
        ground_truth=None,
        tolerance=1e-5,
        verbose=True,
        seed=1,
        with_remplacement=True,
        use_line_search=False,
        step_size = 1e-2,
        alpha_batch_L = 0,
        **Armijo_kwargs,
    ):
    # Initialization
    np.random.seed(seed)
    N = loss_class.N
    d = loss_class.d
    w = initial_w.copy()
    epoch = 0
    iter_per_epoch = round(N/grad_batch_size)
    H = np.identity(d)

    # loss_logs stores the history of the (full) loss
    loss_logs = [loss_class(initial_w, indexes=np.arange(N))]

    # If known, norm_logs stores the history of the distance between our w and the solution w*
    # Else stores the norm of the gradients (expensive)
    norm_logs = [norm(w-ground_truth,2)] if ground_truth is not None else [norm(loss_class.grad(initial_w, np.arange(N)),2)]

    # Initializing the gradient of the selected batch
    grad_idx = np.random.choice(np.arange(N), size=grad_batch_size, replace=with_remplacement)
    batch_grad = loss_class.grad(w, grad_idx)
    
    if verbose:
        print(' | '.join([name.center(8) for name in ["epoch", "loss", "norm"]]))
        msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])
    
    try :
        while (epoch<max_epochs and norm_logs[-1]>tolerance):
            # Beginning of one epoch
            # ----------------------
            iterator = tqdm(range(iter_per_epoch), desc=msg) if verbose else range(iter_per_epoch)
            for iter in iterator:
                # Direction
                direction = -H @ batch_grad

                # Batch loss (used in line search) to avoid recomputation
                if use_line_search :
                    batch_loss = loss_class(w, grad_idx)

                # ------ step size strategy ------
                # Line search
                if use_line_search :
                    alpha = BatchArmijoLineSearch(
                        loss_fn=loss_class.__call__,
                        batch_loss=batch_loss,
                        batch_grad=batch_grad,
                        w=w, 
                        direction=direction,
                        indexes=grad_idx,
                        **Armijo_kwargs
                    )
                # computing L off of a batch at each iteration
                elif alpha_batch_L>0 :
                    alpha = alpha_batch_L / loss_class.get_batch_L(grad_idx)
                # decreasing step_size as a function of iter
                elif callable(step_size) :
                    alpha = step_size(iter)
                # constant step size
                else :
                    alpha = step_size
                # -------------------------------

                # updating w
                old_w = w.copy()
                w = w + alpha * direction
                
                # Computing the gradient of the selected batch
                old_batch_grad = batch_grad.copy()
                grad_idx = np.random.choice(np.arange(N), size=grad_batch_size, replace=with_remplacement)
                batch_grad = loss_class.grad(w, grad_idx)

                # Computing the next H
                s = w-old_w
                v = batch_grad-old_batch_grad
                H = get_H(H, v.reshape(d,1), s.reshape(d,1))
            # ----------------
            # End of the epoch
            if ground_truth is not None :
                norm_logs.append(norm(w-ground_truth, 2))
            else :
                norm_logs.append(norm(loss_class.grad(w, np.arange(N)),2))
            loss_logs.append(loss_class(w, np.arange(N)))
            epoch += 1
            if verbose :
                msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])

    # If the user decides to interrupt, returns the results                    
    except KeyboardInterrupt :
        return w, loss_logs, norm_logs

    return w, loss_logs, norm_logs


def BatchLBFGS(
        initial_w,
        loss_class,
        grad_batch_size,
        m,
        max_epochs,
        ground_truth=None,
        tolerance=1e-5,
        verbose=True,
        seed=1,
        with_remplacement=True,
        use_line_search=False,
        step_size = 1e-2,
        alpha_batch_L = 0,
        **Armijo_kwargs,
    ):
    # Initialization
    np.random.seed(seed)
    N = loss_class.N
    d = loss_class.d
    w = initial_w.copy()
    epoch = 0
    iter_per_epoch = round(N/grad_batch_size)

    # L-BFGS
    H = np.identity(d)
    v = deque(maxlen=m)
    s = deque(maxlen=m)

    # loss_logs stores the history of the (full) loss
    loss_logs = [loss_class(initial_w, indexes=np.arange(N))]

    # If known, norm_logs stores the history of the distance between our w and the solution w*
    # Else stores the norm of the gradients (expensive)
    norm_logs = [norm(w-ground_truth,2)] if ground_truth is not None else [norm(loss_class.grad(initial_w, np.arange(N)),2)]

    # Initializing the gradient of the selected batch
    grad_idx = np.random.choice(np.arange(N), size=grad_batch_size, replace=with_remplacement)
    batch_grad = loss_class.grad(w, grad_idx)
    
    if verbose:
        print(' | '.join([name.center(8) for name in ["epoch", "loss", "norm"]]))
        msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])
    
    # Try statement to store the results in case of KeyboardInterrupt
    try :
        while (epoch<max_epochs and norm_logs[-1]>tolerance):
            # Beginning of one epoch
            # ----------------------
            iterator = tqdm(range(iter_per_epoch), desc=msg) if verbose else range(iter_per_epoch)
            for iter in iterator:
                # Direction
                direction = -H @ batch_grad

                # batch loss (used in line search)
                if use_line_search :
                    batch_loss = loss_class(w, grad_idx)

                # Line search
                if use_line_search :
                    alpha = BatchArmijoLineSearch(
                        loss_fn=loss_class.__call__,
                        batch_loss=batch_loss,
                        batch_grad=batch_grad,
                        w=w, 
                        direction=direction,
                        indexes=grad_idx,
                        **Armijo_kwargs
                    ) 
                # computing L off of a batch at each iteration
                elif alpha_batch_L>0 :
                    alpha = alpha_batch_L / loss_class.get_batch_L(grad_idx)
                # decreasing step_size as a function of iter
                elif callable(step_size) :
                    alpha = step_size(iter)
                # constant step size
                else :
                    alpha = step_size

                # updating w
                old_w = w.copy()
                w = w + alpha * direction
                
                # Computing the gradient of the selected batch
                grad_idx = np.random.choice(np.arange(N), size=grad_batch_size, replace=with_remplacement)
                old_batch_grad = batch_grad.copy()
                batch_grad = loss_class.grad(w, grad_idx)

                # Computing the next H
                s.append((w-old_w).reshape(d,1))
                v.append((batch_grad-old_batch_grad).reshape(d,1))
                H = get_limited_H(m, s, v, d)
            # End of the epoch
            # ----------------
            if ground_truth is not None :
                norm_logs.append(norm(w-ground_truth, 2))
            else :
                norm_logs.append(norm(loss_class.grad(w, np.arange(N)),2))
            loss_logs.append(loss_class(w, np.arange(N)))
            epoch += 1
            if verbose :
                msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])

    # If the user decides to interrupt, stores the results
    except KeyboardInterrupt :
        return w, loss_logs, norm_logs
    
    return w, loss_logs, norm_logs

def SafeSubNewton(
        initial_w,
        loss_class,
        grad_batch_size,
        grad2_batch_size, 
        max_epochs,
        ground_truth=None,
        tolerance=1e-5,
        verbose=True,
        with_remplacement=True,
        use_line_search=False,
        step_size = 1e-2,
        alpha_batch_L = 0,
        **Armijo_kwargs,
    ):  
    '''
    Subsampling second-order Newton algorithm.
    Parameters
    ----------
    loss_class : loss function instance, the class __call__ method has to be defined as the loss function.
                 It also has to have a grad() and grad2() method that compute the gradient and the hessian

    ground_truth : If known, solution w of the problem, used to compute a condition of early stopping when converging before max_epochs,
                   and store the distance of w to solution w* throughout the epochs.
                   Else, uses the norm of the gradient.

    use_line_search : whether or not to use line_search. If set to True, overrides step_size and alpha_batch_L

    alpha_batch_L : If > 0, computes the batch gradient Lipschitz constant at every iteration,
                    then sets the stepsize to alpha_batch_L/batch_L. Overrides step_size if > 0, else uses step_size. 

    step_size : value of a constant stepsize strategy.

    **Armijo_kwargs : keywords parameters of Armijo Line-search function (c=1e-4, theta=0.5, initial_alpha=1)
    '''

    # Initialization
    N = loss_class.N
    d = loss_class.d
    w = initial_w.copy()
    epoch = 0
    iter = 0
    iter_per_epoch = N/grad_batch_size

    # loss_logs stores the history of the (full) loss
    loss_logs = [loss_class(initial_w, indexes=np.arange(N))]

    # If known, norm_logs stores the history of the distance between our w and the solution w*
    # Else stores the norm of the gradients (expensive)
    norm_logs = [norm(w-ground_truth,2)] if ground_truth is not None else [norm(loss_class.grad(initial_w, np.arange(N)),2)]
    
    if verbose:
        print(' | '.join([name.center(8) for name in ["epoch", "loss", "norm", "errors"]]))
        print(' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8), "None".rjust(8)]))
    
    # Keeping track of errors due to overflow, NaN, singular Hessian matrix
    errors = 0

    while (epoch<max_epochs and norm_logs[-1]>tolerance):

        # The randomy selected samples 
        grad_idx = np.random.choice(np.arange(N), size=grad_batch_size, replace=with_remplacement)
        grad2_idx = np.random.choice(np.arange(N), size=grad2_batch_size, replace=with_remplacement)

        # Computing the gradient of the selected batch
        batch_grad = loss_class.grad(w, grad_idx)

        # Computing the Hessian matrix of the selected batch
        batch_grad2 = loss_class.grad2(w, grad2_idx)

        # Direction
        # if not invertible, then direction = -grad(w) (equivalent to batch stochastic gradient)
        try :
            inv_grad2 = np.linalg.inv(batch_grad2)
        except :
            errors += 1
            inv_grad2 = np.identity(d)
        direction = -(inv_grad2 @ batch_grad)

        # Computing the batch loss (used in the line search)
        batch_loss = loss_class(w, grad_idx)

        # Line search
        if use_line_search :
            alpha = BatchArmijoLineSearch(
                loss_fn=loss_class.__call__,
                batch_loss=batch_loss,
                batch_grad=batch_grad,
                w=w, 
                direction=direction,
                indexes=grad_idx,
                **Armijo_kwargs
            ) 

        # computing L off of a batch at each iteration
        elif alpha_batch_L>0 :
            alpha = alpha_batch_L / loss_class.get_batch_L(grad_idx)
        # decreasing step_size as a function of iter
        elif callable(step_size) :
            alpha = step_size(iter)
        # constant step size
        else :
            alpha = step_size

        # updating w
        w = w + alpha * direction
        
        # Incrementing the iterations and updating the history
        iter += 1
        if iter % iter_per_epoch == 0 :
            if ground_truth is not None :
                norm_logs.append(norm(w-ground_truth, 2))
            else :
                norm_logs.append(norm(loss_class.grad(w, np.arange(N)),2))
            loss_logs.append(loss_class(w, np.arange(N)))
            epoch += 1
            if verbose :
                print(' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8), ("%d" % errors).rjust(8)])) 

    return w, loss_logs, norm_logs

def SGD(
        initial_w,
        loss_class,
        grad_batch_size,
        max_epochs,
        ground_truth=None,
        tolerance=1e-5,
        verbose=True,
        seed=1,
        with_remplacement=True,
        use_line_search=False,
        step_size = 1e-2,
        alpha_batch_L = 0,
        **Armijo_kwargs,
    ):

    # Initialization
    np.random.seed(seed)
    N = loss_class.N
    d = loss_class.d
    w = initial_w.copy()
    epoch = 0
    iter = 0
    iter_per_epoch = round(N/grad_batch_size)

    # loss_logs stores the history of the (full) loss
    loss_logs = [loss_class(initial_w, indexes=np.arange(N))]

    # If known, norm_logs stores the history of the distance between our w and the solution w*
    # Else stores the norm of the gradients (expensive)
    norm_logs = [norm(w-ground_truth,2)] if ground_truth is not None else [norm(loss_class.grad(initial_w, np.arange(N)),2)]
    
    if verbose:
        print(' | '.join([name.center(8) for name in ["epoch", "loss", "norm"]]))
        msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])

    try :
        while (epoch<max_epochs and norm_logs[-1]>tolerance):
            # Beginning of one epoch
            # ----------------------
            iterator = tqdm(range(iter_per_epoch), desc=msg) if verbose else range(iter_per_epoch)
            for iter in iterator:

                # The randomy selected samples 
                grad_idx = np.random.choice(np.arange(N), size=grad_batch_size, replace=with_remplacement)

                # Computing the gradient of the selected batch
                batch_grad = loss_class.grad(w, grad_idx)

                # Direction
                direction = -batch_grad

                # Computing the batch loss for the Line search
                batch_loss = loss_class(w, grad_idx)

                # Line search
                if use_line_search :
                    alpha = BatchArmijoLineSearch(
                        loss_fn=loss_class.__call__,
                        batch_loss=batch_loss,
                        batch_grad=batch_grad,
                        w=w, 
                        direction=direction,
                        indexes=grad_idx,
                        **Armijo_kwargs
                    ) 

                # computing L off of a batch at each iteration
                elif alpha_batch_L>0 :
                    alpha = alpha_batch_L / loss_class.get_batch_L(grad_idx)
                # decreasing step_size as a function of iter
                elif callable(step_size) :
                    alpha = step_size(iter)
                # constant step size
                else :
                    alpha = step_size

                # updating w
                w = w + alpha * direction
            # End of the epoch
            # ----------------
            if ground_truth is not None :
                norm_logs.append(norm(w-ground_truth, 2))
            else :
                norm_logs.append(norm(loss_class.grad(w, np.arange(N)),2))
            loss_logs.append(loss_class(w, np.arange(N)))
            epoch += 1
            if verbose :
                msg = ' | '.join([("%d" % epoch).rjust(8),("%.2e" % loss_logs[-1]).rjust(8),("%.2e" % norm_logs[-1]).rjust(8)])

    except KeyboardInterrupt:
        return w, loss_logs, norm_logs

    return w, loss_logs, norm_logs