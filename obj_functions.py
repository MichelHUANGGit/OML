import numpy as np


class QuadraticFunction():

    def __init__(self) -> None:
        # D is the dimension of w
        self.D = 3
        self.grad_eval = 0
        self.func_eval = 0

    def __call__(self, w : np.array):
        self.func_eval += 1
        w1 = w[0,0]
        w2 = w[1,0]
        w3 = w[2,0]
        return 2*(w1+w2+w3-3)**2 + (w1-w2)**2 + (w2-w3)**2

    def nabla(self, w):
        self.grad_eval += 1
        return (np.array([[6,2,4],
                          [2,8,2],
                          [4,2,6]], dtype='float64') @ w)-12
    
    def nabla2(self, w):
        self.grad_eval += self.D
        return np.array([[6,2,4],
                         [2,8,2],
                         [4,2,6]], dtype='float64')
        

class RosenBrock :
    
    def __init__(self) -> None:
        # D is the dimension of w
        self.D = 2
        self.func_eval = 0
        self.grad_eval = 0

    def __call__(self, w):
        self.func_eval += 1
        w1 = w[0,0]
        w2 = w[1,0]
        return 100*(w2-(w1**2))**2 + (1-w1)**2

    def nabla(self, w):
        self.grad_eval += 1
        w1 = w[0,0]
        w2 = w[1,0]
        return np.array([[400*w1**3 - 400*w1*w2 + 2*w1 - 2],
                        [200*w2 - 200*w1**2]])

    def nabla2(self, w):
        self.grad_eval += self.D
        w1 = w[0,0]
        w2 = w[1,0]
        return np.array([[1200*w1**2 - 400*w2 + 2, -400*w1],
                        [-400*w1                , 200]])
    

if __name__ == '__main__':
    from scipy.optimize import check_grad

    w0 = np.random.randn(2,1)
    r = RosenBrock()
    print(r(w0), "\n", r.nabla(w0), "\n", r.nabla2(w0))