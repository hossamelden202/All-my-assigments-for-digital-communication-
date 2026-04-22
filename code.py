import numpy as np
import matplotlib.pyplot as plt


# Signal definitions

N = 1000          # number of samples
T = 1.0           # signal duration (sec)
dt = T / N        # time step
t = np.linspace(0, T, N, endpoint=False)

# s1(t): rect pulse, amplitude 1 from 0 to 1
s1 = np.ones(N)

# s2(t): +1 from t=0.25 to 1, -1 from 0 to 0.25
s2 = np.where(t < 0.25, -1.0, 1.0)



# 1.1  Gram-Schmidt Orthogonalization

def GM_Bases(s1, s2, dt):
    """
    Returns orthonormal basis functions phi1, phi2
    using Gram-Schmidt for signals s1 and s2.
    dt  : time step (for numerical integration via sum*dt)
    """
    # --- phi1: normalize s1 ---
    energy1 = np.sum(s1 ** 2) * dt          # <s1, s1>
    phi1 = s1 / np.sqrt(energy1)

    # --- remove phi1 component from s2 ---
    proj = np.sum(s2 * phi1) * dt            # <s2, phi1>
    g2 = s2 - proj * phi1                   # residual

    # --- check if s2 is linearly independent of s1 ---
    energy_g2 = np.sum(g2 ** 2) * dt
    if energy_g2 < 1e-10:                   # linearly dependent
        phi2 = np.zeros(len(s1))
    else:
        phi2 = g2 / np.sqrt(energy_g2)

    return phi1, phi2



# 1.2  Signal Space Representation

def signal_space(s, phi1, phi2, dt):
    """
    Projects signal s onto phi1 and phi2.
    Returns scalar coordinates (v1, v2).
    """
    v1 = np.sum(s * phi1) * dt
    v2 = np.sum(s * phi2) * dt
    return v1, v2



# Requirement 1 — Compute & Plot Bases

phi1, phi2 = GM_Bases(s1, s2, dt)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, phi, label in zip(axes, [phi1, phi2], [r'$\phi_1(t)$', r'$\phi_2(t)$']):
    ax.plot(t, phi, linewidth=2)
    ax.set_title(label, fontsize=14)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend([label], fontsize=11)
    ax.grid(True)
fig.suptitle('Gram-Schmidt Orthonormal Basis Functions', fontsize=15)
plt.tight_layout()
plt.savefig('fig1_bases.png', dpi=150)
plt.show()



# Requirement 2 — Signal Space of s1, s2

v1_s1, v2_s1 = signal_space(s1, phi1, phi2, dt)
v1_s2, v2_s2 = signal_space(s2, phi1, phi2, dt)

print(f"s1 coordinates: ({v1_s1:.4f}, {v2_s1:.4f})")
print(f"s2 coordinates: ({v1_s2:.4f}, {v2_s2:.4f})")

plt.figure(figsize=(6, 6))
plt.scatter([v1_s1], [v2_s1], s=150, color='blue',  zorder=5, label=r'$s_1(t)$')
plt.scatter([v1_s2], [v2_s2], s=150, color='red',   zorder=5, label=r'$s_2(t)$')
plt.axhline(0, color='k', linewidth=0.8)
plt.axvline(0, color='k', linewidth=0.8)
plt.xlabel(r'$v_1$ (projection onto $\phi_1$)', fontsize=12)
plt.ylabel(r'$v_2$ (projection onto $\phi_2$)', fontsize=12)
plt.title('Signal Space Representation of $s_1$ and $s_2$', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('fig2_signal_space.png', dpi=150)
plt.show()



# Requirement 3 — AWGN Scatter Plots

n_samples = 100

# Energy of s1 (s1 and s2 have the same energy here)
E = np.sum(s1 ** 2) * dt
print(f"\nSignal energy E = {E:.4f} J")

snr_dB_list  = [-5, 0, 10]          # E/sigma^2 in dB
snr_lin_list = [10 ** (x / 10) for x in snr_dB_list]

for snr_dB, snr_lin in zip(snr_dB_list, snr_lin_list):
    sigma2 = E / snr_lin             # noise variance
    sigma  = np.sqrt(sigma2)

    coords_r1 = []
    coords_r2 = []

    for _ in range(n_samples):
        # Generate a fresh noise vector for each sample
        w = np.random.normal(0, sigma, N)

        r1 = s1 + w
        r2 = s2 + w

        v1_r1, v2_r1 = signal_space(r1, phi1, phi2, dt)
        v1_r2, v2_r2 = signal_space(r2, phi1, phi2, dt)

        coords_r1.append((v1_r1, v2_r1))
        coords_r2.append((v1_r2, v2_r2))

    coords_r1 = np.array(coords_r1)
    coords_r2 = np.array(coords_r2)

    plt.figure(figsize=(7, 7))

    # Scatter: noisy received signals
    plt.scatter(coords_r1[:, 0], coords_r1[:, 1],
                s=40, color='blue', alpha=0.6, label=r'$r_1(t)$ samples')
    plt.scatter(coords_r2[:, 0], coords_r2[:, 1],
                s=40, color='red',  alpha=0.6, label=r'$r_2(t)$ samples')

    # Mark ideal signal points
    plt.scatter([v1_s1], [v2_s1], s=200, color='blue',
                marker='*', zorder=5, label=r'$s_1$ ideal')
    plt.scatter([v1_s2], [v2_s2], s=200, color='red',
                marker='*', zorder=5, label=r'$s_2$ ideal')

    plt.axhline(0, color='k', linewidth=0.8)
    plt.axvline(0, color='k', linewidth=0.8)
    plt.xlabel(r'$v_1$', fontsize=13)
    plt.ylabel(r'$v_2$', fontsize=13)
    plt.title(f'Signal Space with AWGN  —  E/σ² = {snr_dB} dB', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'fig3_scatter_{snr_dB}dB.png', dpi=150)
    plt.show()



# Requirement 4 — Answer (printed)

print("""
Requirement 4 — Effect of noise on signal space:
- AWGN causes the received signal points to scatter around the ideal signal points.
- The scatter INCREASES as sigma^2 INCREASES (i.e. lower E/sigma^2 ratio).
- At E/sigma^2 = 10 dB  → points cluster tightly near s1 and s2 (low noise).
- At E/sigma^2 = -5 dB  → points are widely spread (high noise), making it
  harder to distinguish between s1 and s2.
""")