from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import numpy as np
import scipy
import EKF
import sys
import os

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


def update_elipse(ellipse, x, y, covariance):
  
    eigenvalues, eigenvectors = np.linalg.eigh(covariance[:2,:2])
    
  
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]


    width = 2 * 3 * np.sqrt(eigenvalues[0]) 
    height = 2 * 3 * np.sqrt(eigenvalues[1])  


    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse.set_center((x, y))
    ellipse.set_width(width)
    ellipse.set_height(height)
    ellipse.set_angle(angle)

    return ellipse

def q5(ekf: EKF.EKF):
    
    states, covariances = ekf.run(r_max=1, x_0_correct=np.array([ekf.x_true[0], ekf.y_true[0], ekf.th_true[0]]))
    
    fig, ax = plt.subplots()
    state, = ax.plot([], [], 'ro', markersize=1)
    true_pos, = ax.plot([], [], 'bo', markersize=1)
    
    ellipse = Ellipse(xy=(0, 0), width=1, height=1, edgecolor='r', facecolor='none', lw=1)
    ax.add_patch(ellipse)
    
    ax.set_xlim(-5, 15)
    ax.set_ylim(-10, 10)

    states = np.hstack(states)

    ax.scatter(ekf.l[:, 0], ekf.l[:, 1], color='black', marker='o', label="Landmarks")
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left', va='top')
    
    def animate(frame):
        state.set_data([states[0, frame]], [states[1,frame]])
        true_pos.set_data(ekf.x_true[frame], ekf.y_true[frame])
        time_text.set_text(f'Time: {ekf.t[frame].item():.2f} s')
        update_elipse(
            ellipse=ellipse, 
            x=states[0, frame], 
            y=states[1, frame],
            covariance=covariances[frame]
            )
        
        return state, true_pos, ellipse, time_text


    ani = FuncAnimation(fig, animate, frames=len(ekf.t), interval= 100 / 4, repeat=False)


    plt.show()
    save_path = "output_4x.mp4"
    if not os.path.exists(save_path):  
        print(f"Outputing to {save_path}...")
        ani.save(f'{save_path}', writer='ffmpeg')




def main():
    
    ekf = EKF.EKF(dataset)
    
    # 4. a)
    if sys.argv[1] == "q4a" or len(sys.argv) == 1:
        q4(ekf=ekf, x_0_correct=np.array([ekf.x_true[0], ekf.y_true[0], ekf.th_true[0]]), plt_path="4a.png")
    
    # 4. b)
    if sys.argv[1] == "q4b" or len(sys.argv) == 1:
        q4(ekf=ekf, x_0_correct=np.array([[1], [1], [0.1]]), plt_path="4b.png")
    
    # 4. c)
    if sys.argv[1] == "q4c" or len(sys.argv) == 1:
        q4(ekf=ekf, x_0_correct=np.array([ekf.x_true[0], ekf.y_true[0], ekf.th_true[0]]), plt_path="4c.png", CRLB=True)
    
    # 5. 
    if sys.argv[1] == "q5" or len(sys.argv) == 1:
        q5(ekf=ekf)

if __name__ == "__main__":
    
    main()