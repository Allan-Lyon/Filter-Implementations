import numpy as np
import matplotlib.pyplot as plt

# Initialize state
dt = 1.0  # Time step (seconds)
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])  # Constant velocity model

H = np.eye(4)  # Full state observation
R = np.diag([0.1, 0.1, 0.1, 0.1])  # Observation noise
Q = np.diag([0.01, 0.01, 0.1, 0.1])  # Process noise

x = np.array([0, 0, 1, 1])  # Initial state: position (0,0), velocity (1,1)
P = np.eye(4) * 0.1  # Initial covariance

# Simulate delta-v at step 10 (unknown to filter)
maneuver_step = 10
true_delta_v = np.array([0.5, -0.2])

# Logs
states = [x.copy()]
measurements = []
true_states = [x.copy()]

# Simulate and estimate for 20 steps
for k in range(20):
    # Inject maneuver into true dynamics only
    if k == maneuver_step:
        x[2:4] += true_delta_v

    # Propagate true state (simulate dynamics)
    x = F @ x
    true_states.append(x.copy())

    # Simulate measurement (with noise)
    z = x + np.random.multivariate_normal(np.zeros(4), R)
    measurements.append(z)

# Now apply EKF assuming no known maneuver
x_est = np.array([0, 0, 1, 1])
P_est = np.eye(4) * 0.1
estimates = [x_est.copy()]

for k in range(20):
    # Prediction
    x_est = F @ x_est
    P_est = F @ P_est @ F.T + Q

    # Measurement update
    z = measurements[k]
    y = z - H @ x_est
    S = H @ P_est @ H.T + R
    K = P_est @ H.T @ np.linalg.inv(S)
    x_est = x_est + K @ y
    P_est = (np.eye(4) - K @ H) @ P_est

    estimates.append(x_est.copy())

# Plot results
estimates = np.array(estimates)
true_states = np.array(true_states)
measurements = np.array(measurements)

plt.figure(figsize=(10, 5))
plt.plot(true_states[:, 0], true_states[:, 1], label='True Position')
plt.plot(estimates[:, 0], estimates[:, 1], label='Estimated Position')
plt.plot(measurements[:, 0], measurements[:, 1], 'rx', alpha=0.5, label='Measurements')
plt.axvline(x=true_states[maneuver_step, 0], color='gray', linestyle='--', label='Maneuver (Unknown)')
plt.legend()
plt.title('EKF with Unknown Maneuver (Delta-v Estimation via Filter Residuals)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
