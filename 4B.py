import matplotlib.pyplot as plt

# Reference trajectory and increments
r = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]
delta_r = [r[k + 1] - r[k] for k in range(12)]

# Fixed disturbance sequence for simulation only
w = [0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0]

# Initial state
e, eta = 0, 0

# Store simulated quantities
e_values = [e]
eta_values = [eta]
y_values = [e + r[0]]
u_values = []

for k in range(12):

    # Select control using only the current state and feedback policy
    u = policy[k][(e, eta)]
    u_values.append(u)

    # Apply the specified disturbance and update the state
    e_next = e + eta - delta_r[k]
    eta_next = eta + u + w[k]

    e, eta = e_next, eta_next

    # Store the new state and actual position
    e_values.append(e)
    eta_values.append(eta)
    y_values.append(e + r[k + 1])

# Plot actual position and reference
time_state = range(13)

plt.figure(figsize=(9, 5))
plt.plot(time_state, y_values, marker="o", label="Actual position $y_k$")
plt.plot(time_state, r, marker="s", linestyle="--", label="Reference $r_k$")
plt.xlabel("Time step k")
plt.ylabel("Lateral position")
plt.title("Actual Position vs Reference")
plt.xticks(time_state)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot control input
time_control = range(12)

plt.figure(figsize=(9, 4))
plt.step(time_control, u_values, where="post", marker="o")
plt.xlabel("Time step k")
plt.ylabel("Control input $u_k$")
plt.title("Feedback Control Inputs")
plt.xticks(time_control)
plt.yticks([-1, 0, 1])
plt.grid(True)
plt.tight_layout()
plt.show()

print("k | e_k | eta_k | r_k | y_k")
for k in range(13):
    print(k, e_values[k], eta_values[k], r[k], y_values[k])

print("Control sequence:", u_values)

stage_cost = sum(
    e_values[k]**2
    + 0.25 * eta_values[k]**2
    + 0.50 * u_values[k]**2
    for k in range(12)
)

terminal_cost = 4 * e_values[12]**2 + eta_values[12]**2

realized_cost = stage_cost + terminal_cost

print("Stage cost:", stage_cost)
print("Terminal cost:", terminal_cost)
print("Realized total cost:", realized_cost)