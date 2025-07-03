#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Kernel Ridge Regression System ID Experiments

import numpy as np
from krr import KernelRidgeRegression, kernel_rbf_function_M, kernel_rbf_function_N
from systems import LinearDCMotorPlant, PredatorPreyPlant, InvertedPendulum


def random_noise_gen(std=0.01):
    while True:
        yield np.random.normal(0, std)


def generate_trajectory(plant, x0, u_seq, eta, dt):
    X = []
    Y = []
    x = x0
    for u in u_seq:
        X.append(x)
        x_next = plant.dynamics(x, u, eta, dt)
        Y.append(x_next)
        x = x_next
    return np.array(X), np.array(Y)


# def run_krr_experiment(plant, kernel_function_M, kernel_function_N, lambda_reg=1e-4, dt=0.01, mode="full", theta=200):
#     x0 = np.array([0.1, 0.1])
#     u_seq = np.random.uniform(-1, 1, size=10000)
#     eta = random_noise_gen(std=0.01)
#     X, Y = generate_trajectory(plant, x0, u_seq, eta, dt)

#     if mode == "landmark":
#         X_landmarks = online_determinantal_landmark_selection(X, theta, kernel_function_M)
#         Y_landmarks = Y[:len(X_landmarks)]
#         X_train, Y_train = X_landmarks, Y_landmarks
#     elif mode == "latest":
#         X_train, Y_train = X[-theta:], Y[-theta:]
#     else:
#         X_train, Y_train = X, Y

#     models = []
#     for i in range(Y.shape[1]):
#         model = KernelRidgeRegression(lambda_reg)
#         model.training(X_train, Y_train[:, i], kernel_function_M)
#         models.append(model)

#     test_eta = random_noise_gen(std=0.01)
#     x = x0
#     predictions = []
#     targets = []
#     for u in u_seq:
#         x_true = plant.dynamics(x, u, test_eta, dt)
#         y_pred = np.array([model.predict(x.reshape(1, -1), kernel_function_N).item() for model in models])
#         predictions.append(y_pred)
#         targets.append(x_true)
#         x = x_true

#     predictions = np.array(predictions)
#     targets = np.array(targets)
#     error = np.mean(np.linalg.norm(predictions - targets, axis=1))
#     print(f"[{mode.upper()}] Mean prediction error: {error:.5f}")
#     return predictions, targets


# def online_determinantal_landmark_selection(X, theta, kernel_function, alpha=1.0):
#     """
#     Implements the Online Determinantal Landmark Selection (ODLS) algorithm.
#     Corrected to unconditionally fill the landmark budget before swapping.
    
#     Args:
#         X (np.ndarray): The full trajectory of states.
#         theta (int): The landmark budget.
#         kernel_function (callable): The kernel function K(X, Y).
#         alpha (float): The aging/decay factor for adaptive memory. 1.0 means no decay.

#     Returns:
#         np.ndarray: The final set of selected landmark states.
#     """
#     landmarks = []  # Corresponds to Θ in the paper
#     K_inv = np.array([])

#     for t, x_t in enumerate(X):
#         x_t = np.array(x_t).reshape(1, -1)

#         # --- THIS IS THE FIX ---
#         # Case 1: The landmark set is not yet full.
#         # Unconditionally add the new point to fill the budget.
#         if len(landmarks) < theta:
#             landmarks.append({'state': x_t.flatten(), 'timestamp': t})
#             # When the set becomes full for the first time, compute the inverse kernel matrix.
#             if len(landmarks) == theta:
#                 final_states = np.array([lm['state'] for lm in landmarks])
#                 K = kernel_function(final_states, final_states)
#                 K_inv = np.linalg.inv(K + 1e-8 * np.eye(len(landmarks)))
#             continue
#         # -----------------------

#         # Case 2: The landmark set is full, consider a swap.
#         current_states = np.array([lm['state'] for lm in landmarks])
#         diag_K_inv = np.diag(K_inv)
        
#         # Step 11: Find the most redundant landmark using the aging factor.
#         aged_redundancies = [alpha**(t - lm['timestamp']) / d_inv for lm, d_inv in zip(landmarks, diag_K_inv)]
#         r_index = np.argmax(aged_redundancies)
        
#         # Step 12 & 13: Create the temporary set Θ' and calculate the new point's score against it.
#         temp_indices = [i for i in range(theta) if i != r_index]
#         temp_states = current_states[temp_indices]
        
#         # This temporary inverse calculation is needed to correctly score the new point.
#         K_temp = kernel_function(temp_states, temp_states)
#         K_temp_inv = np.linalg.inv(K_temp + 1e-8 * np.eye(len(temp_states)))
        
#         v_t_prime = kernel_function(temp_states, x_t)
#         score_vs_temp = kernel_function(x_t, x_t)[0, 0] - (v_t_prime.T @ K_temp_inv @ v_t_prime).item()
        
#         # Step 14: Check if the new point is more informative than the landmark to be retired.
#         redundancy_of_r = aged_redundancies[r_index]
#         if score_vs_temp > redundancy_of_r:
#             landmarks[r_index] = {'state': x_t.flatten(), 'timestamp': t}
#             # Rebuild the main inverse matrix after the swap.
#             final_states = np.array([lm['state'] for lm in landmarks])
#             K = kernel_function(final_states, final_states)
#             K_inv = np.linalg.inv(K + 1e-8 * np.eye(theta))
            
#     return np.array([lm['state'] for lm in landmarks])

def run_krr_experiment(plant, x0, u_seq, eta, kernel_function_M, kernel_function_N, lambda_reg=1e-4, dt=0.01, horizon = 5, mode="full", theta=500, batch_size=50, alpha=1.0):

    # Helper function for batch processing
    def update_landmarks_with_batch(current_landmarks, K_inv, batch_XU, t_start, kernel_function, theta, alpha):
        scores = []
        current_landmark_vectors = np.array([lm['vector'] for lm in current_landmarks])
        for i, xu_candidate in enumerate(batch_XU):
            xu_candidate = xu_candidate.reshape(1, -1)
            v = kernel_function(current_landmark_vectors, xu_candidate)
            score = kernel_function(xu_candidate, xu_candidate)[0, 0] - (v.T @ K_inv @ v).item()
            scores.append(score)

        best_candidate_index_in_batch = np.argmax(scores)
        best_candidate_score = scores[best_candidate_index_in_batch]
        best_candidate_vector = batch_XU[best_candidate_index_in_batch]
        
        diag_K_inv = np.diag(K_inv)
        aged_redundancies = [alpha**((t_start + best_candidate_index_in_batch) - lm['timestamp']) / d_inv for lm, d_inv in zip(current_landmarks, diag_K_inv)]
        r_index_to_replace = np.argmax(aged_redundancies)
        redundancy_of_r = aged_redundancies[r_index_to_replace]

        if best_candidate_score > redundancy_of_r:
            current_landmarks[r_index_to_replace] = {'vector': best_candidate_vector.flatten(), 'timestamp': t_start + best_candidate_index_in_batch}
            final_vectors = np.array([lm['vector'] for lm in current_landmarks])
            K = kernel_function(final_vectors, final_vectors)
            K_inv = np.linalg.inv(K + 1e-8 * np.eye(theta))
            
        return current_landmarks, K_inv
        
    # --- Main function logic ---
    X, Y = generate_trajectory(plant, x0, u_seq, eta, dt)
    # --- Testing Loop ---
    test_eta = random_noise_gen(std=0.01)

    XU = np.hstack((X, u_seq.reshape(-1, 1)))

    models = []
    if mode != "naive":
        X_train, Y_train = None, None
        if mode == "landmark":
            initial_vectors = XU[:theta]
            landmarks = [{'vector': initial_vectors[i].flatten(), 'timestamp': i} for i in range(theta)]
            K = kernel_function_M(initial_vectors, initial_vectors)
            K_inv = np.linalg.inv(K + 1e-8 * np.eye(theta))
            for i in range(theta, len(XU), batch_size):
                batch_end = min(i + batch_size, len(XU))
                data_batch = XU[i:batch_end]
                if len(data_batch) > 0:
                    landmarks, K_inv = update_landmarks_with_batch(landmarks, K_inv, data_batch, t_start=i, kernel_function=kernel_function_M, theta=theta, alpha=alpha)
            X_train = np.array([lm['vector'] for lm in landmarks])
            y_indices = [lm['timestamp'] for lm in landmarks]
            Y_train = Y[y_indices]
        elif mode == "latest":
            X_train, Y_train = XU[-theta:], Y[-theta:]
        else: # mode == "full"
            X_train = XU
            Y_train = Y

        if X_train.shape[0] < theta:
            print(f"[{mode.upper()}] Warning: Not enough training points. Skipping.")
            return

        for i in range(Y.shape[1]):
            model = KernelRidgeRegression(lambda_reg)
            model.training(X_train, Y_train[:, i], kernel_function_M)
            models.append(model)
        
    # --- Testing & Evaluation ---
    test_eta = random_noise_gen(std=0.01)
    
    # 1. One-Step-Ahead Mean Euclidean Error
    x = x0
    predictions_1_step = []
    targets_1_step = []
    for i, u in enumerate(u_seq):
        x_true = plant.dynamics(x, u, test_eta, dt)
        targets_1_step.append(x_true)
        
        y_pred = None
        if mode == "naive":
            y_pred = x
        else:
            current_xu_input = np.hstack((x, u)).reshape(1, -1)
            y_pred = np.array([model.predict(current_xu_input, kernel_function_N).item() for model in models])
        
        predictions_1_step.append(y_pred)
        x = x_true
    
    error_1_step = np.mean(np.linalg.norm(np.array(predictions_1_step) - np.array(targets_1_step), axis=1))

    # 2. Five-Step-Ahead Mean Euclidean Error
    horizon_errors = []
    for i in range(len(X) - horizon):
        x_sim = X[i]
        
        predictions_5_step = []
        for k in range(horizon):
            u_sim = u_seq[i+k]
            
            y_pred_sim = None
            if mode == "naive":
                y_pred_sim = x_sim
            else:
                current_xu_input = np.hstack((x_sim, u_sim)).reshape(1, -1)
                y_pred_sim = np.array([model.predict(current_xu_input, kernel_function_N).item() for model in models])
            
            predictions_5_step.append(y_pred_sim)
            x_sim = y_pred_sim

        ground_truth_5_step = Y[i : i + horizon]
        
        # --- FIXED: Calculate Mean Euclidean Error for the horizon ---
        # This calculates the error magnitude for each of the 5 steps and then averages them.
        horizon_euc_error = np.mean(np.linalg.norm(np.array(predictions_5_step) - ground_truth_5_step, axis=1))
        horizon_errors.append(horizon_euc_error)
        
    error_5_step = np.mean(horizon_errors)
    
    # --- FIXED: Updated print statement for clarity ---
    print(f"[{mode.upper():<8}] 1-Step Mean Error: {error_1_step:.5f} | {horizon}-Step Mean Error: {error_5_step:.5f}")


if __name__ == "__main__":
    motor = LinearDCMotorPlant(J=0.01, Kb=0.01, Kf=0.1, Km=0.01, R=1.0, L=0.5)
    predator_prey = PredatorPreyPlant(a=1.0, b=0.5, c=0.5, d=0.5)
    pendulum = InvertedPendulum(m=1.0, M=5.0, b=0.1, I=0.1, l=0.5)


    print("\nRunning KRR on Linear DC Motor:")
    # signals
    dt = 0.01
    x0 = np.array([0.1, 0.1]) 
    u_seq = np.random.uniform(-1, 1, size=2000)
    eta = random_noise_gen(std=0.01)
    run_krr_experiment(motor, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="full")
    run_krr_experiment(motor, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="latest")
    run_krr_experiment(motor, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="landmark")
    run_krr_experiment(motor, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="naive")


    print("\nRunning KRR on Predator-Prey:")
    # signals
    dt = 0.1
    x0 = np.array([0.1, 0.1]) 
    u_seq = np.random.uniform(-1, 1, size=2000)
    eta = random_noise_gen(std=0.01)
    run_krr_experiment(predator_prey, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="full")
    run_krr_experiment(predator_prey, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="latest")
    run_krr_experiment(predator_prey, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="landmark")
    run_krr_experiment(predator_prey, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="naive")


    print("\nRunning KRR on InvertedPendulum:")
    # signals
    dt = 0.005
    x0 = np.array([0.1, 0.1, 0.1, 0.1])  # Initial state for Inverted Pendulum
    u_seq = np.random.uniform(-1, 1, size=2000)
    eta = random_noise_gen(std=0.01)
    # run_krr_experiment(pendulum, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="full")
    run_krr_experiment(pendulum, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="latest")
    run_krr_experiment(pendulum, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="landmark")
    run_krr_experiment(pendulum, x0, u_seq, eta, kernel_rbf_function_M, kernel_rbf_function_N, lambda_reg=1e-3, dt=dt, mode="naive")
