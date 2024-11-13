import numpy as np
import sympy as sp
from sympy import Matrix, symbols
from sympy.utilities.lambdify import lambdify



class EKF:
    
    t : np.ndarray
    r : np.ndarray
    b : np.ndarray
    v : np.ndarray
    om : np.ndarray
    l : np.ndarray
    x_true : np.ndarray
    y_true : np.ndarray
    th_true : np.ndarray
    true_valid : np.ndarray
    d : float
    r_var : float
    b_var : float
    v_var : float
    om_var : float
    
    def __init__(self, dataset):
        
        self.t = dataset["t"]
        self.r = dataset["r"]
        self.b = dataset["b"]
        self.v = dataset['v']
        self.om = dataset["om"]
        self.l = dataset["l"]
        self.x_true = dataset["x_true"]
        self.y_true = dataset["y_true"]
        self.th_true = dataset["th_true"]
        self.true_valid = dataset["true_valid"]
        self.d = dataset["d"].item()
        self.r_var = dataset["r_var"].item()
        self.b_var = dataset["b_var"].item()
        self.v_var = dataset["v_var"].item()
        self.om_var = dataset["om_var"].item()
        
    


    def wrap_to_pi(self, angle):
        """Maps an angle to the range -pi to pi."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def wrap_bearing(self, mat):
        mat[1, 0] = self.wrap_to_pi(mat[1, 0])
        return mat

    def wrap_angle(self, mat):
        mat[2, 0] = self.wrap_to_pi(mat[2, 0])
        return mat




    def run(self, r_max, x_0_correct, CRLB=False):
        
        T = 0.1
        K = len(self.x_true)
        # EKF equations
        _x_k, _y_k, theta_k                     = symbols(r"x_k y_k \theta_k")
        _x_prev, _y_prev, theta_prev            = symbols(r"x_{k-1} y_{k-1} \theta_{k-1}")
        x_k, x_prev                             = Matrix([[_x_k], [_y_k], [theta_k]]), Matrix([[_x_prev], [_y_prev], [theta_prev]])
        v_k, omega_k, w_k1, w_k2, n_k1, n_k2    = symbols(r"v_k \omega_k w_k1 w_k2 n_k1 n_k2")
        u_k, w_k, n_k                           = Matrix([[v_k], [omega_k]]), Matrix([[w_k1], [w_k2]]), Matrix([[n_k1], [n_k2]])
        M                                       = Matrix([[sp.cos(theta_prev), 0], [sp.sin(theta_prev), 0], [0, 1]])
        x_l, y_l                                = symbols("x_l y_l")


        f             = x_prev + T * M * (u_k + w_k) 
        __f__         = lambdify((_x_prev, _y_prev, theta_prev, v_k, omega_k, w_k1, w_k2), f, modules="numpy")
        __F__         = lambdify((_x_prev, _y_prev, theta_prev, v_k, w_k1, w_k2), f.jacobian([_x_prev, _y_prev, theta_prev]), modules="numpy")

        g             = Matrix([[sp.sqrt((x_l - _x_k - self.d * sp.cos(theta_k))**2 + (y_l - _y_k - self.d * sp.sin(theta_k))**2)], 
                            [sp.atan2(y_l - _y_k - self.d * sp.sin(theta_k), x_l - _x_k - self.d * sp.cos(theta_k)) - theta_k]]) + n_k
        __g__         = lambdify((_x_k, _y_k, theta_k, x_l, y_l, n_k1, n_k2), g, modules="numpy")
        G             = g.jacobian([_x_k, _y_k, theta_k])
        __G__         = lambdify((_x_k, _y_k, theta_k, x_l, y_l, n_k1, n_k2), G, modules="numpy")

        Q             = Matrix([[self.v_var, 0], [0, self.om_var]]) 
        __Q__         = lambdify((theta_prev), Q, modules='numpy')
        dQ            = T**2 * M * Q * M.T
        __dQ__        = lambdify((theta_prev), dQ, modules='numpy')



        R = Matrix([[self.r_var, 0], [0, self.b_var]])
        dR = R
        
        X_correct = []
        P_correct = []
        
        P_k_correct = np.diag([1,1,0.1])
        x_k_correct = x_0_correct
        
        
        for k in range(K):
            
            rangesIdx = np.where((self.r[k] < r_max) & (self.r[k] != 0))[0]
            l_k = len(rangesIdx)
            print(f"\rk: {k}", end='', flush=True)
            # Prediction
            
            if not CRLB:
                dF_k = __F__(x_k_correct[0,0], x_k_correct[1,0], x_k_correct[2,0], self.v[k].item(), 0, 0)
                dQ_k = __dQ__(x_k_correct[2,0])
            else:
                dF_k = __F__(self.x_true[k].item(), self.y_true[k].item(), self.th_true[k].item(), self.v[k].item(), 0, 0)
                dQ_k = __dQ__(self.th_true[k].item())
            
            P_k_predict = dF_k @ P_k_correct @ dF_k.T + dQ_k

            x_k_predict = self.wrap_angle(__f__(x_k_correct[0,0], x_k_correct[1,0], x_k_correct[2,0], self.v[k].item(), self.om[k].item(), 0, 0))

            # kalman gain
            if not CRLB:
                G_k_l = lambda ll : __G__(x_k_predict[0,0], x_k_predict[1,0], x_k_predict[2,0], self.l[ll,0], self.l[ll,1], 0, 0)
            else:
                G_k_l = lambda ll : __G__(self.x_true[k].item(), self.y_true[k].item(), self.th_true[k].item(), self.l[ll,0], self.l[ll,1], 0, 0)
            obs = [G_k_l(ll) for ll in rangesIdx]
            
            if obs:
                G_k = np.vstack(obs)
                K_k = P_k_predict @ G_k.T @ np.linalg.inv((G_k @ P_k_predict @ G_k.T + np.diag([self.r_var, self.b_var] * l_k)))

                # corrector
                y_k = np.vstack([np.array([[self.r[k, ll]], [self.b[k, ll]]]) for ll in rangesIdx])
                
                g_k_l = lambda ll : __g__(x_k_predict[0, 0], x_k_predict[1,0], x_k_predict[2,0], self.l[ll,0], self.l[ll,1], 0, 0)

                g_k = np.vstack([self.wrap_bearing(np.array(g_k_l(ll))) for ll in rangesIdx])
                
                P_k_correct = (np.eye(3) - K_k @ G_k) @ P_k_predict
                
                x_k_correct = self.wrap_angle(x_k_predict + K_k @ (y_k - g_k))
                

            else:
                P_k_correct = P_k_predict
                x_k_correct = x_k_predict
                
            X_correct.append(x_k_correct)
            P_correct.append(P_k_correct)
            
        print("\nDone\n")
        return X_correct, P_correct