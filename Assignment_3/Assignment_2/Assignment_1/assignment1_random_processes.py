import numpy as np
import matplotlib.pyplot as plt


# PART 1  Identifying WSS vs non-WSS
#READ THIS 
#Note this code depend on random value this code will produce different value each 
#run but same conclusions

K = 100_000
N = 2000
n = np.arange(1, N + 1)  # 1-indexed to match assignment notation

W = np.random.randn(K, N)

XA = W
XB = (1 + 0.5 * n) * W  # n broadcast over K realizations

#  Task 1: Plot 5 realizations 
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
for i in range(5):
    axes[0].plot(n, XA[i], alpha=0.7, linewidth=0.8)
axes[0].set_title("Part 1  Process A: 5 Realizations")
axes[0].set_xlabel("n"); axes[0].set_ylabel("Amplitude")

for i in range(5):
    axes[1].plot(n, XB[i], alpha=0.7, linewidth=0.8)
axes[1].set_title("Part 1  Process B: 5 Realizations")
axes[1].set_xlabel("n"); axes[1].set_ylabel("Amplitude")
plt.tight_layout()
plt.savefig("part1_realizations.png", dpi=150)
plt.close()

#  Task 2: Ensemble mean at n=100, 1000, 2000 
indices = [99, 999, 1999]  # 0-indexed
print(" Part 1  Ensemble Mean ")
for idx in indices:
    print(f"  n={idx+1}: mean_A = {XA[:, idx].mean():.5f}, mean_B = {XB[:, idx].mean():.5f}")

#  Task 3: Ensemble autocorrelation 
print("\n Part 1  Ensemble Autocorrelation ")
pairs = [(100, 150), (1500, 1550)]
for n1, n2 in pairs:
    rx_A = np.mean(XA[:, n1-1] * XA[:, n2-1])
    rx_B = np.mean(XB[:, n1-1] * XB[:, n2-1])
    print(f"  (n1={n1}, n2={n2}): R_A = {rx_A:.4f}, R_B = {rx_B:.4f}")

# theoretical: R_A(n1,n2) = delta(n1-n2) => 0 when n1!=n2
# R_B(n1,n2) = (1+0.5*n1)*(1+0.5*n2) * delta(n1-n2) => same, 0 when n1!=n2
# But variance R_B(n,n) = (1+0.5n)^2 grows with n  non-WSS

print("\n Part 1  WSS Decision ")
print("Process A: WSS  mean=0 (constant), R_A(n1,n2)=delta(n1-n2) depends only on lag.")
print("Process B: NOT WSS  variance (1+0.5n)^2 grows with n, so autocorrelation is NOT lag-only.")



# PART 2  Fast and Slow WSS Processes


K2 = 200
N2 = 1000
n2 = np.arange(N2)
tau_range = np.arange(-100, 101)

W2 = np.random.randn(K2, N2)

def moving_average(W, M):
    # causal MA filter over each realization
    K, N = W.shape
    out = np.zeros_like(W)
    for i in range(M - 1, N):
        out[:, i] = W[:, i - M + 1:i + 1].mean(axis=1)
    # edge: for n < M-1, use whatever samples exist
    for i in range(M - 1):
        out[:, i] = W[:, :i + 1].mean(axis=1)
    return out

def run_part2(M, suffix):
    XA2 = W2.copy()
    XB2 = moving_average(W2, M)

    # Task 1
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].plot(n2, XA2[0], linewidth=0.8)
    axes[0].set_title(f"Part 2 (M={M})  Process A: one realization")
    axes[1].plot(n2, XB2[0], linewidth=0.8, color='orange')
    axes[1].set_title(f"Part 2 (M={M})  Process B: one realization")
    plt.tight_layout()
    plt.savefig(f"part2_realizations_M{suffix}.png", dpi=150)
    plt.close()

    # Task 2: ensemble mean vs time
    mean_A = XA2.mean(axis=0)
    mean_B = XB2.mean(axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].plot(n2, mean_A, linewidth=0.8)
    axes[0].set_title(f"Part 2 (M={M})  Process A: Ensemble Mean vs Time")
    axes[0].set_ylim(-0.1, 0.1)
    axes[1].plot(n2, mean_B, linewidth=0.8, color='orange')
    axes[1].set_title(f"Part 2 (M={M})  Process B: Ensemble Mean vs Time")
    axes[1].set_ylim(-0.1, 0.1)
    plt.tight_layout()
    plt.savefig(f"part2_mean_M{suffix}.png", dpi=150)
    plt.close()

    # Task 3: ensemble ACF for tau = -100..100
    # R[tau] = (1/K) * sum_k x_k[n0] * x_k[n0+tau], averaged over valid n0
    def ensemble_acf(X, tau_vals):
        acf = np.zeros(len(tau_vals))
        n0 = 500  # fixed reference index
        for j, tau in enumerate(tau_vals):
            idx = n0 + tau
            if 0 <= idx < N2:
                acf[j] = np.mean(X[:, n0] * X[:, idx])
        return acf

    acf_A = ensemble_acf(XA2, tau_range)
    acf_B = ensemble_acf(XB2, tau_range)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].stem(tau_range, acf_A, markerfmt='C0.', linefmt='C0-', basefmt='k-')
    axes[0].set_title(f"Part 2 (M={M})  Process A: ACF")
    axes[1].stem(tau_range, acf_B, markerfmt='C1.', linefmt='C1-', basefmt='k-')
    axes[1].set_title(f"Part 2 (M={M})  Process B: ACF")
    plt.tight_layout()
    plt.savefig(f"part2_acf_M{suffix}.png", dpi=150)
    plt.close()

    # Task 4: PSD via FFT of ACF
    # zero-pad ACF to length 2048 for smoother PSD
    fft_len = 2048
    psd_A = np.abs(np.fft.fftshift(np.fft.fft(acf_A, n=fft_len)))
    psd_B = np.abs(np.fft.fftshift(np.fft.fft(acf_B, n=fft_len)))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_len))

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))
    axes[0].plot(freqs, psd_A, linewidth=0.8)
    axes[0].set_title(f"Part 2 (M={M})  Process A: PSD")
    axes[0].set_xlabel("Normalized Frequency")
    axes[1].plot(freqs, psd_B, linewidth=0.8, color='orange')
    axes[1].set_title(f"Part 2 (M={M})  Process B: PSD")
    axes[1].set_xlabel("Normalized Frequency")
    plt.tight_layout()
    plt.savefig(f"part2_psd_M{suffix}.png", dpi=150)
    plt.close()

    print(f"\n Part 2 M={M} ")
    print(f"  Process A: fast variations (white noise), flat PSD, narrow ACF spike at tau=0")
    print(f"  Process B (M={M}): slower variations, ACF width ~M={M}, PSD narrower (lowpass of width ~1/M)")

run_part2(10, "10")
run_part2(50, "50")

print("\n Part 2  Effect of M ")
print("Larger M => more smoothing => slower time variation => wider ACF => narrower PSD (more lowpass).")
print("ACF width ~ M, PSD bandwidth ~ 1/M. Time variation speed and PSD width are inversely related.")
print("This is the time-bandwidth duality: wider ACF <-> narrower PSD.")

print("\nAll plots saved to this dir")