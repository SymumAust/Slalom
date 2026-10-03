"""Problem 2, Part 2: Synchronous value iteration on the augmented states.
Run:  python part2_value_iteration.py
"""

# ---------------- Problem data ----------------
N = 12
r = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]
dr = [r[k+1] - r[k] for k in range(N)]          # Delta r_k = r_{k+1} - r_k
W = [(-1, 0.1), (0, 0.8), (1, 0.1)]             # (w, probability)
ETAS = [-2, -1, 0, 1, 2]
U_ORDER = [-1, 0, 1]                            # tie-break order
TIE = 1e-12                                     # tie tolerance
TOL = 1e-10                                     # stopping tolerance
DELTA = "DELTA"                                 # zero-cost absorbing state

def e_grid(k):       return range(-6 - 3*k, 6 + 3*k + 1)
def g(e, eta, u):    return e**2 + 0.25*eta**2 + 0.50*u**2
def gN(e, eta):      return 4*e**2 + eta**2
def admissible(eta): return [u for u in U_ORDER if abs(eta + u) <= 1]

# ---------------- Augmented states z = (k, e, eta) ----------------
STATES = [(k, e, eta) for k in range(N + 1) for e in e_grid(k) for eta in ETAS]
ALL = STATES + [DELTA]

def Q(J, z, u):
    """Cost now + expected cost-to-go from the OLD table J (k < N)."""
    k, e, eta = z
    e_next = e + eta - dr[k]
    return g(e, eta, u) + sum(p * J[(k + 1, e_next, eta + u + w)] for w, p in W)

def T(J):
    """Bellman optimality operator. Reads only J, writes a NEW table."""
    TJ = {DELTA: 0.0}                                       # J(DELTA) = 0
    for z in STATES:
        k, e, eta = z
        if k == N:
            TJ[z] = gN(e, eta) + J[DELTA]                   # charge g_N once -> DELTA
        else:
            TJ[z] = min(Q(J, z, u) for u in admissible(eta))
    return TJ

def greedy(J, z):
    """Greedy control at z; ties within 1e-12 broken in order -1, 0, 1."""
    qs = {u: Q(J, z, u) for u in admissible(z[2])}
    best = min(qs.values())
    return next(u for u in U_ORDER if u in qs and qs[u] <= best + TIE)

# ---------------- Synchronous value iteration ----------------
J = {z: 0.0 for z in ALL}                                   # J = 0 everywhere
sweeps = 0
while True:
    TJ = T(J)                                               # one synchronous sweep
    sweeps += 1
    residual = max(abs(TJ[z] - J[z]) for z in ALL)          # ||TJ - J||_inf
    print(f"sweep {sweeps:2d}: residual = {residual:.6e}")
    J = TJ                                                  # replace table AFTER sweep
    if residual < TOL:
        break

mu = {z: greedy(J, z) for z in STATES if z[0] < N}          # greedy policy

# ---------------- Report ----------------
print(f"\nNumber of augmented states: {len(STATES)} (+ DELTA)")
print(f"Number of sweeps: {sweeps}")
print(f"J(0,-2,0) = {J[(0,-2,0)]:.6f}")
print(f"J(0, 0,0) = {J[(0,0,0)]:.6f}")
print(f"J(0, 2,0) = {J[(0,2,0)]:.6f}")
print(f"Greedy policy mu(k,0,0), k=0..11: {[mu[(k,0,0)] for k in range(N)]}")
