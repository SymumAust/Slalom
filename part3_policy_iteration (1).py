"""Problem 2, Part 3: Policy iteration on the augmented states.
Run:  python part3_policy_iteration.py
"""

# ---------------- Problem data ----------------
N = 12
r = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]
dr = [r[k+1] - r[k] for k in range(N)]          # Delta r_k = r_{k+1} - r_k
W = [(-1, 0.1), (0, 0.8), (1, 0.1)]             # (w, probability)
ETAS = [-2, -1, 0, 1, 2]
U_ORDER = [-1, 0, 1]                            # tie-break order
TIE = 1e-12
DELTA = "DELTA"                                 # zero-cost absorbing state

def e_grid(k):       return range(-6 - 3*k, 6 + 3*k + 1)
def g(e, eta, u):    return e**2 + 0.25*eta**2 + 0.50*u**2
def gN(e, eta):      return 4*e**2 + eta**2
def admissible(eta): return [u for u in U_ORDER if abs(eta + u) <= 1]
def sign(x):         return (x > 0) - (x < 0)

# Decision states (k < N); the k = N states and DELTA have no control
DECISION = [(k, e, eta) for k in range(N) for e in e_grid(k) for eta in ETAS]

def Q(J, z, u):
    """Cost now + expected cost-to-go using table J."""
    k, e, eta = z
    e_next = e + eta - dr[k]
    return g(e, eta, u) + sum(p * J[(k + 1, e_next, eta + u + w)] for w, p in W)

def evaluate(mu):
    """POLICY EVALUATION: exact backward pass with the controls FIXED (no min)."""
    J = {DELTA: 0.0}
    for e in e_grid(N):
        for eta in ETAS:
            J[(N, e, eta)] = gN(e, eta) + J[DELTA]       # charge g_N once -> DELTA
    for k in range(N - 1, -1, -1):
        for e in e_grid(k):
            for eta in ETAS:
                z = (k, e, eta)
                J[z] = Q(J, z, mu[z])                    # use the policy's control
    return J

def improve(J):
    """POLICY IMPROVEMENT: greedy control at every state, ties -> order -1, 0, 1."""
    new_mu = {}
    for z in DECISION:
        qs = {u: Q(J, z, u) for u in admissible(z[2])}
        best = min(qs.values())
        new_mu[z] = next(u for u in U_ORDER if u in qs and qs[u] <= best + TIE)
    return new_mu

# ---------------- Policy iteration ----------------
mu = {z: -sign(z[2]) for z in DECISION}                  # initial policy u = -sign(eta)
assert all(abs(z[2] + u) <= 1 for z, u in mu.items())    # check feasibility

rounds = 0
hist_J, hist_changed, hist_stage = [], [], []          # for the convergence plot
while True:
    rounds += 1
    J = evaluate(mu)                                     # 1) evaluate
    new_mu = improve(J)                                  # 2) improve
    changed = sum(new_mu[z] != mu[z] for z in DECISION)
    hist_J.append(J); hist_changed.append(changed)
    hist_stage.append([sum(new_mu[z] != mu[z] for z in DECISION if z[0] == k)
                       for k in range(N)])
    print(f"round {rounds}: J_mu(0,0,0) = {J[(0,0,0)]:.6f}, controls changed = {changed}")
    if changed == 0:                                     # 3) stop if nothing changed
        break
    mu = new_mu

# ---------------- Report ----------------
print(f"\nNumber of evaluation/improvement rounds: {rounds}")
print(f"J(0,-2,0) = {J[(0,-2,0)]:.6f}")
print(f"J(0, 0,0) = {J[(0,0,0)]:.6f}")
print(f"J(0, 2,0) = {J[(0,2,0)]:.6f}")
print(f"Policy mu(k,0,0), k=0..11: {[mu[(k,0,0)] for k in range(N)]}")

# ---------------- Convergence diagram ----------------
from pathlib import Path

import matplotlib.pyplot as plt

R = list(range(1, rounds + 1))
J_final = hist_J[-1]
err = [max(abs(Jr[z] - J_final[z]) for z in J_final) for Jr in hist_J]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Policy iteration convergence", fontsize=15, weight="bold")

# (1) value of the start state after each evaluation
a = ax[0, 0]
v = [Jr[(0, 0, 0)] for Jr in hist_J]
a.plot(R, v, "o-", color="tab:blue", lw=3, ms=10)
a.axhline(J_final[(0, 0, 0)], color="green", ls="--", lw=2,
          label=f"optimal = {J_final[(0, 0, 0)]:.2f}")
for x, y in zip(R, v):
    a.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center")
a.set(title="Cost of the current policy at (k,e,η) = (0,0,0)",
      xlabel="round", ylabel="$J_\\mu(0,0,0)$", xticks=R)
a.set_ylim(min(v) - 3, max(v) + 5)
a.grid(alpha=0.3); a.legend()

# (2) number of controls changed by each improvement
a = ax[0, 1]
bars = a.bar(R, hist_changed, color=["tab:red"] * (rounds - 1) + ["green"], edgecolor="black")
for b, c in zip(bars, hist_changed):
    a.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", weight="bold")
a.set_yscale("symlog", linthresh=1)
a.set(title="Controls changed by the improvement step (0 = stop)",
      xlabel="round", ylabel="number of states changed", xticks=R)
a.grid(alpha=0.3, axis="y")

# (3) max value error over ALL states vs. the final (optimal) values
a = ax[1, 0]
FLOOR = 1e-16
a.semilogy(R, [max(x, FLOOR) for x in err], "s-", color="tab:purple", lw=3, ms=10)
for x, y in zip(R, err):
    a.annotate("0 (exact)" if y == 0 else f"{y:.3g}", (x, max(y, FLOOR)),
               textcoords="offset points", xytext=(0, 10), ha="center")
a.set(title="Max value error over all states, $\\max_z |J_\\mu(z) - J^*(z)|$",
      xlabel="round", ylabel="error (log scale)", xticks=R)
a.set_ylim(1e-17, 1e5)
a.grid(alpha=0.3, which="both")

# (4) where the changes happen: stage-by-stage
a = ax[1, 1]
im = a.imshow(hist_stage, cmap="Reds", aspect="auto", origin="upper",
              extent=[-0.5, N - 0.5, rounds + 0.5, 0.5])
for i, row in enumerate(hist_stage):
    for k, c in enumerate(row):
        a.text(k, i + 1, str(c), ha="center", va="center", fontsize=9,
               color="white" if c > 70 else "black")
fig.colorbar(im, ax=a, label="controls changed")
a.set(title="Changes per stage in each round", xlabel="stage k", ylabel="round",
      xticks=range(N), yticks=R)

plt.tight_layout()
output_path = Path(__file__).resolve().with_name("part3_convergence.png")
plt.savefig(output_path, dpi=150)
plt.show()
