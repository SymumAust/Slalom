"""Problem 2, Part 1: Backward DP for the normalized slalom model."""


N = 12
r = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]
dr = [r[k+1] - r[k] for k in range(N)]           # Delta r_k
W = [(-1, 0.1), (0, 0.8), (1, 0.1)]              # (w, prob)
ETAS = [-2, -1, 0, 1, 2]
U_ORDER = [-1, 0, 1]                             # tie-break order
TOL = 1e-12

def e_grid(k):  return list(range(-6 - 3*k, 6 + 3*k + 1))
def g(e, eta, u): return e**2 + 0.25*eta**2 + 0.50*u**2
def gN(e, eta):   return 4*e**2 + eta**2
def admissible(eta): return [u for u in U_ORDER if abs(eta + u) <= 1]

# J[k][(e,eta)] = optimal cost-to-go, MU[k][(e,eta)] = optimal control
J  = [dict() for _ in range(N + 1)]
MU = [dict() for _ in range(N)]
for e in e_grid(N):
    for eta in ETAS:
        J[N][(e, eta)] = gN(e, eta)

for k in range(N - 1, -1, -1):                   # k = 11, 10, ..., 0
    for e in e_grid(k):
        for eta in ETAS:
            e_next = e + eta - dr[k]             # independent of u and w
            Q = {}
            for u in admissible(eta):
                exp_future = 0.0
                for w, p in W:
                    nxt = (e_next, eta + u + w)
                    assert nxt in J[k+1], "successor off grid!"   # no clipping check
                    exp_future += p * J[k+1][nxt]
                Q[u] = g(e, eta, u) + exp_future
            best = min(Q.values())
            u_star = next(u for u in U_ORDER if u in Q and Q[u] <= best + TOL)
            J[k][(e, eta)], MU[k][(e, eta)] = Q[u_star], u_star

if __name__ == "__main__":
    # Worked backup at k = 11, (e, eta) = (0, 0)
    k, e, eta = 11, 0, 0
    print(f"k=11: Delta r_11 = {dr[11]}, e_12 = {e + eta - dr[11]}")
    for u in admissible(eta):
        terms = [(w, p, J[12][(e + eta - dr[k], eta + u + w)]) for w, p in W]
        ef = sum(p * v for _, p, v in terms)
        print(f"  u={u:+d}: g={g(e,eta,u):.2f}, E[J12]={ef:.2f}, Q={g(e,eta,u)+ef:.2f}  {terms}")
    print(f"  J_11(0,0) = {J[11][(0,0)]:.4f}, mu_11(0,0) = {MU[11][(0,0)]}")
    print(f"J_0(0,0) = {J[0][(0,0)]:.6f}, mu_0(0,0) = {MU[0][(0,0)]}")
    print("Optimal mu_k(0,0) for k=0..11:", [MU[k][(0,0)] for k in range(N)])
    print("Grid sizes |X_k|:", [len(e_grid(k)) * 5 for k in range(N + 1)])
