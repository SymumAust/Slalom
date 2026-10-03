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
while True:
    rounds += 1
    J = evaluate(mu)                                     # 1) evaluate
    new_mu = improve(J)                                  # 2) improve
    changed = sum(new_mu[z] != mu[z] for z in DECISION)
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
