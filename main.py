from matplotlib import pyplot as plt
import numpy as np
import scipy
import EKF


dataset = scipy.io.loadmat("dataset2.mat")

def q4(ekf: EKF.EKF, x_0_correct, plt_path, CRLB=False):
    
    r_maxes = [1,3,5]
    mats = []


    for r_max in r_maxes:
        X_correct, P_correct = ekf.run(r_max, x_0_correct=x_0_correct, CRLB=CRLB)
        
        mats.append((X_correct, P_correct))
        
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,20))

    for i in range(3):
        
        X_correct, P_correct = mats[i]
        x_correct_mat = np.hstack(X_correct)
        
        for j in range(3):
            
            if j == 0:
                ground_t = ekf.x_true
                y_label = r"$\hat{x} - x_{\text{true}}$"
            elif j == 1:
                ground_t = ekf.y_true
                y_label = r"$\hat{y} - y_{\text{true}}$"
            else:
                ground_t = ekf.th_true
                y_label = r"$\hat{\theta} - \theta_{\text{true}}$"
            
            residuals = x_correct_mat[j] - np.squeeze(ground_t)
            if j == 2:
                residuals = ekf.wrap_to_pi(residuals)

            variances = np.vstack([p.diagonal() for p in P_correct])

            lower_bound = - 3 * np.sqrt(variances[:, j])
            upper_bound = + 3 * np.sqrt(variances[:, j])

            ax[i][j].set_title(f"r_max = {r_maxes[i]}")  # Title for each subplot
            ax[i][j].set_xlabel("time (s)")  # X axis title
            ax[i][j].set_ylabel(y_label)  # Y axis title
        
            ax[i][j].plot(ekf.t.squeeze(), lower_bound, color='purple', linestyle='--', label='Lower Bound (-3 std dev)')
            ax[i][j].plot(ekf.t.squeeze(), upper_bound, color='orange', linestyle='--', label='Upper Bound (+3 std dev)')
            ax[i][j].legend()
            ax[i][j].plot(ekf.t.squeeze(), residuals)

            
            
    plt.tight_layout()
    
    plt.savefig(plt_path, dpi=300)
    
    plt.show()
    
    return mats

def main():
    
    ekf = EKF.EKF(dataset)
    
    # 4. a)
    q4(ekf=ekf, x_0_correct=np.array([ekf.x_true[0], ekf.y_true[0], ekf.th_true[0]]), plt_path="4a.png")
    
    # 4. b)
    q4(ekf=ekf, x_0_correct=np.array([[1], [1], [0.1]]), plt_path="4b.png")
    
    # 4. c)
    q4(ekf=ekf, x_0_correct=np.array([ekf.x_true[0], ekf.y_true[0], ekf.th_true[0]]), plt_path="4c.png", CRLB=True)
    
    

if __name__ == "__main__":
    
    main()