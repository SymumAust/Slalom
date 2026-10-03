"""Problem 2, Part 4: compare backward DP, value iteration and policy iteration.
Standalone file.  Run:  python part4_compare.py
Needs matplotlib:  python -m pip install matplotlib
"""
import time, statistics

# ======================= Problem data =======================
N = 12
r = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0]
dr = [r[k+1] - r[k] for k in range(N)]
W = [(-1, 0.1), (0, 0.8), (1, 0.1)]
ETAS = [-2, -1, 0, 1, 2]
U_ORDER = [-1, 0, 1]
TIE, TOL = 1e-12, 1e-10
DELTA = "DELTA"

def e_grid(k):       return range(-6 - 3*k, 6 + 3*k + 1)
def g(e, eta, u):    return e**2 + 0.25*eta**2 + 0.50*u**2
def gN(e, eta):      return 4*e**2 + eta**2
def admissible(eta): return [u for u in U_ORDER if abs(eta + u) <= 1]
def sign(x):         return (x > 0) - (x < 0)

STATES   = [(k, e, eta) for k in range(N + 1) for e in e_grid(k) for eta in ETAS]
DECISION = [z for z in STATES if z[0] < N]
ALL      = STATES + [DELTA]

def Q(J, z, u):
    """Cost now + expected cost-to-go (the building block of everything)."""
    k, e, eta = z
    e_next = e + eta - dr[k]
    return g(e, eta, u) + sum(p * J[(k + 1, e_next, eta + u + w)] for w, p in W)

def greedy(J, z):
    """BELLMAN BACKUP with argmin: try all controls, keep the best (ties -> -1, 0, 1)."""
    qs = {u: Q(J, z, u) for u in admissible(z[2])}
    best = min(qs.values())
    return next(u for u in U_ORDER if u in qs and qs[u] <= best + TIE)

def terminal_table():
    J = {DELTA: 0.0}
    for e in e_grid(N):
        for eta in ETAS:
            J[(N, e, eta)] = gN(e, eta) + J[DELTA]
    return J

def sup_diff(A, B):
    return max(abs(A[z] - B[z]) for z in ALL)

# ======================= Method 1: backward DP =======================
def backward_dp():
    J, mu = terminal_table(), {}
    for k in range(N - 1, -1, -1):
        for e in e_grid(k):
            for eta in ETAS:
                z = (k, e, eta)
                mu[z] = greedy(J, z)                 # Bellman backup (min over u)
                J[z] = Q(J, z, mu[z])
    return J, mu

# ======================= Method 2: value iteration =======================
def bellman_T(J):
    """Bellman optimality operator: synchronous, reads only the OLD table."""
    TJ = {DELTA: 0.0}
    for z in STATES:
        k, e, eta = z
        TJ[z] = gN(e, eta) + J[DELTA] if k == N else \
                min(Q(J, z, u) for u in admissible(eta))     # Bellman backup
    return TJ

def value_iteration(J_ref=None):
    J, sweeps, hist = {z: 0.0 for z in ALL}, 0, []
    while True:
        TJ = bellman_T(J); sweeps += 1
        residual = sup_diff(TJ, J); J = TJ
        if J_ref is not None: hist.append(sup_diff(J, J_ref))
        if residual < TOL: break
    mu = {z: greedy(J, z) for z in DECISION}
    return J, mu, sweeps, hist

# ======================= Method 3: policy iteration =======================
def evaluate_policy(mu):
    """POLICY EVALUATION: backward pass with the control FIXED (no min)."""
    J = terminal_table()
    for k in range(N - 1, -1, -1):
        for e in e_grid(k):
            for eta in ETAS:
                z = (k, e, eta)
                J[z] = Q(J, z, mu[z])                # policy's control only
    return J

def policy_iteration(J_ref=None):
    mu, rounds, hist = {z: -sign(z[2]) for z in DECISION}, 0, []
    while True:
        rounds += 1
        J = evaluate_policy(mu)
        if J_ref is not None: hist.append(sup_diff(J, J_ref))
        new_mu = {z: greedy(J, z) for z in DECISION}
        if new_mu == mu: return J, mu, rounds, hist
        mu = new_mu

# ======================= (c) simulation =======================
W_SEQ = [0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0]

def simulate(mu, e0=0, eta0=0):
    e, eta = e0, eta0
    ys, us, stage_costs = [e + r[0]], [], []
    for k in range(N):
        u = mu[(k, e, eta)]          # controller sees ONLY the current state
        stage_costs.append(g(e, eta, u))
        w = W_SEQ[k]                 # disturbance revealed AFTER u is chosen
        e, eta = e + eta - dr[k], eta + u + w
        ys.append(e + r[k + 1]); us.append(u)
    return ys, us, stage_costs, gN(e, eta)

def timed(fn, repeats=7):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)

# ======================= Run =======================
if __name__ == "__main__":
    # ---------- (a) ----------
    (J_dp, mu_dp), t_dp = timed(backward_dp)
    (J_vi, mu_vi, n_vi, _), t_vi = timed(value_iteration)
    (J_pi, mu_pi, n_pi, _), t_pi = timed(policy_iteration)
    dis = lambda mu: sum(mu[z] != mu_dp[z] for z in DECISION)

    rows = [("Backward DP", J_dp, mu_dp, "1 backward pass", t_dp),
            ("Value iteration", J_vi, mu_vi, f"{n_vi} sweeps", t_vi),
            ("Policy iteration", J_pi, mu_pi, f"{n_pi} rounds", t_pi)]
    print("(a) Comparison table")
    print(f"{'Method':<17}{'J0(-2,0)':>11}{'J0(0,0)':>11}{'J0(2,0)':>11}"
          f"{'max|J-J_DP|':>13}{'disagree':>10}{'iterations':>18}{'time ms':>10}")
    for name, J, mu, it, t in rows:
        print(f"{name:<17}{J[(0,-2,0)]:>11.6f}{J[(0,0,0)]:>11.6f}{J[(0,2,0)]:>11.6f}"
              f"{sup_diff(J, J_dp):>13.2e}{dis(mu):>10d}{it:>18}{1000*t:>10.2f}")

    n_back = sum(len(admissible(z[2])) for z in DECISION)   # Q evals per full backup
    n_eval = len(DECISION)                                   # Q evals per evaluation
    print(f"\nWork (Q evaluations): DP = {n_back}, VI = {n_vi}x{n_back} = {n_vi*n_back}"
          f" (+{n_back} extraction), PI = {n_pi}x({n_eval}+{n_back}) = {n_pi*(n_eval+n_back)}")

    # ---------- (b) ----------
    _, _, _, h_vi = value_iteration(J_dp)
    _, _, _, h_pi = policy_iteration(J_dp)
    print("\n(b) VI error per sweep:     ", [f"{x:.3g}" for x in h_vi])
    print("    PI error per evaluation:", [f"{x:.3g}" for x in h_pi])

    # ---------- (c) ----------
    print("\n(c) w =", W_SEQ)
    sims = {}
    for name, _, mu, _, _ in rows:
        ys, us, sc, term = simulate(mu)
        sims[name] = (ys, us, sum(sc) + term)
        print(f"  {name:<17} u={us}  y={ys}  cost={sum(sc)+term:.2f}")
    ys, us, sc, term = simulate(mu_dp)
    print("  stage costs:", sc, " terminal:", term, " total:", sum(sc) + term)
    print(f"  J0*(0,0) = {J_dp[(0,0,0)]:.6f}")

    # Check: average the realized cost over ALL 3^12 disturbance sequences
    import itertools
    P = {-1: 0.1, 0: 0.8, 1: 0.1}
    mean, p_le, cmin, cmax = 0.0, 0.0, float("inf"), 0.0
    for seq in itertools.product([-1, 0, 1], repeat=N):
        W_SEQ[:] = seq                                   # temporarily swap the sequence
        _, _, sc_, t_ = simulate(mu_dp); c = sum(sc_) + t_
        p = 1.0
        for w in seq: p *= P[w]
        mean += p * c; cmin = min(cmin, c); cmax = max(cmax, c)
        if c <= 20.25: p_le += p
    W_SEQ[:] = [0, 1, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0]   # restore the given sequence
    print(f"  exact average over all 3^12 sequences = {mean:.6f}")
    print(f"  cheapest run = {cmin}, most expensive run = {cmax}, "
          f"P(cost <= 20.25) = {p_le:.3f}")

    # ---------- plots (not timed) ----------
    import matplotlib.pyplot as plt
    FLOOR = 1e-16
    plt.figure(figsize=(9, 5))
    plt.semilogy(range(1, len(h_vi)+1), [max(x, FLOOR) for x in h_vi], "o-",
                 lw=2.5, ms=7, label=f"Value iteration ({n_vi} sweeps)")
    plt.semilogy(range(1, len(h_pi)+1), [max(x, FLOOR) for x in h_pi], "s-",
                 color="tab:red", lw=2.5, ms=8, label=f"Policy iteration ({n_pi} rounds)")
    plt.axhline(FLOOR, color="gray", ls=":")
    plt.text(1, FLOOR*3, "exactly 0 (drawn at 1e-16)", color="gray")
    plt.title("(b) Value error vs. backward DP: max over all states |J - J_DP|", weight="bold")
    plt.xlabel("iteration (VI sweep / PI policy evaluation)"); plt.ylabel("error (log scale)")
    plt.xticks(range(1, max(len(h_vi), len(h_pi)) + 1)); plt.ylim(1e-17, 1e5)
    plt.grid(alpha=0.3, which="both"); plt.legend(); plt.tight_layout()
    plt.savefig("part4b_value_error.png", dpi=150)

    ks = list(range(N + 1))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    a1.plot(ks, r, "o-", color="red", lw=3, ms=9, label="reference $r_k$")
    st = {"Backward DP": ("tab:blue", "-", "s", 10), "Value iteration": ("tab:green", "--", "^", 7),
          "Policy iteration": ("black", ":", "x", 7)}
    for name, (ys, us, cost) in sims.items():
        c, ls, m, ms = st[name]
        a1.plot(ks, ys, ls, color=c, marker=m, ms=ms, lw=2.5, label=f"{name}: $y_k$ (cost {cost:.2f})")
        a2.step(range(N + 1), us + [us[-1]], where="post", color=c, ls=ls, lw=2.5,
                label=name)                     # u_k is held over [k, k+1), incl. k = 11
        a2.plot(range(N), us, m, color=c, ms=ms)
    for k, w in enumerate(W_SEQ):
        if w:
            a1.axvline(k + 0.5, color="purple", alpha=0.25, lw=6)
            a1.text(k + 0.5, 0.98, f"$w_{{{k}}}$={w:+d}", ha="center", va="top",
                    transform=a1.get_xaxis_transform(), color="purple", weight="bold")
    a1.set_title("(c) Closed-loop run from $x_0=(0,0)$  (purple bands: $w_k$ acts between stage k and k+1)",
                 weight="bold", fontsize=11)
    a1.set_ylabel("lateral position"); a1.grid(alpha=0.3); a1.legend(loc="lower left")
    a2.set_ylabel("control $u_k$"); a2.set_xlabel("stage k"); a2.set_yticks([-1, 0, 1])
    a2.set_xticks(ks); a2.grid(alpha=0.3); a2.legend(loc="lower right")
    plt.tight_layout(); plt.savefig("part4c_simulation.png", dpi=150)
    plt.show()