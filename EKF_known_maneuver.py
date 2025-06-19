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

B = np.array([
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
])  # Control input affects velocity

H = np.eye(4)  # Full state observation
R = np.diag([0.1, 0.1, 0.1, 0.1])  # Observation noise
Q = np.diag([0.01, 0.01, 0.1, 0.1])  # Process noise

x = np.array([0, 0, 1, 1])  # Initial state: position (0,0), velocity (1,1)
P = np.eye(4) * 0.1  # Initial covariance

# Maneuver setup
maneuver_step = 10
delta_v = np.array([0.5, -0.2])  # Known delta-v (m/s)

# Logs
states = [x.copy()]
measurements = []

# Simulate for 20 steps
for k in range(20):
    # Control input (delta-v at maneuver step only)
    u = delta_v if k == maneuver_step else np.array([0.0, 0.0])
    
    # Prediction
    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    # Simulate measurement (with noise)
    z = x + np.random.multivariate_normal(np.zeros(4), R)
    measurements.append(z)

    # Update
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(4) - K @ H) @ P

    states.append(x.copy())

# Plot position
states = np.array(states)
measurements = np.array(measurements)

plt.figure(figsize=(10, 5))
plt.plot(states[:, 0], states[:, 1], label='Estimated Position')
plt.plot(measurements[:, 0], measurements[:, 1], 'rx', alpha=0.5, label='Measurements')
plt.axvline(x=states[maneuver_step, 0], color='gray', linestyle='--', label='Maneuver')
plt.legend()
plt.title('EKF with Control Maneuver (Delta-v)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
