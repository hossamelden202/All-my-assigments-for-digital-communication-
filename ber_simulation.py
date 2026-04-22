import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

Fs = 500
T  = 1.0
dt = T / Fs
t  = np.arange(0, T, dt)

g = np.sqrt(3) * t
E = np.trapezoid(g**2, t)

h_matched = g[::-1] / np.sqrt(E)   # h_a(t) = g(T-t)/sqrt(E)
h_rect    = np.ones(Fs)

signal_a = np.trapezoid(g * (g / np.sqrt(E)), t)   # sqrt(E) = 1
signal_b = g[-1]                                     # sqrt(3)
signal_c = np.trapezoid(g, t)                        # sqrt(3)/2

ENo_dB  = np.arange(-10, 21, 1)
ENo_lin = 10 ** (ENo_dB / 10)
N0_arr  = E / ENo_lin

# theory BER:
# (a),(c): noise var = N0/2 * integral(h^2 dt) = N0/2  (both filters have unit energy)
# (b):     noise var = N0/(2*dt)   direct sample, full simulation bandwidth
BER_a_th = Q((signal_a/2) / np.sqrt(N0_arr/2))
BER_b_th = Q((signal_b/2) / np.sqrt(N0_arr/(2*dt)))
BER_c_th = Q((signal_c/2) / np.sqrt(N0_arr/2))

def simulate_ber(ENo, N_bits, rng, chunk=50_000):
    N0      = E / ENo
    sigma_w = np.sqrt(N0 / (2 * dt))
    err_a = err_b = err_c = 0
    done  = 0
    while done < N_bits:
        n     = min(chunk, N_bits - done)
        bits  = rng.integers(0, 2, n)
        noise = rng.normal(0, sigma_w, (n, Fs))
        rx    = np.outer(bits, g) + noise

        y_a = np.trapezoid(rx * (g / np.sqrt(E)), t, axis=1)
        y_b = rx[:, -1]
        y_c = np.trapezoid(rx, t, axis=1)

        err_a += int(np.sum((y_a > signal_a/2) != bits.astype(bool)))
        err_b += int(np.sum((y_b > signal_b/2) != bits.astype(bool)))
        err_c += int(np.sum((y_c > signal_c/2) != bits.astype(bool)))
        done  += n
    return err_a, err_b, err_c, N_bits

MIN_ERRORS  = 50
MAX_BITS    = 500_000
MIN_RELIABLE = 30   # mask sim point if fewer than this many errors observed

rng = np.random.default_rng(42)

errors_a = np.zeros(len(ENo_dB), dtype=int)
errors_b = np.zeros(len(ENo_dB), dtype=int)
errors_c = np.zeros(len(ENo_dB), dtype=int)
n_bits   = np.zeros(len(ENo_dB), dtype=int)

BER_a_sim = np.zeros(len(ENo_dB))
BER_b_sim = np.zeros(len(ENo_dB))
BER_c_sim = np.zeros(len(ENo_dB))

for i, ENo in enumerate(ENo_lin):
    exp_ber = max(float(BER_a_th[i]), 1e-6)
    N = int(min(max(MIN_ERRORS / exp_ber, 20_000), MAX_BITS))

    ea, eb, ec, N = simulate_ber(ENo, N, rng)
    errors_a[i] = ea; errors_b[i] = eb; errors_c[i] = ec; n_bits[i] = N

    BER_a_sim[i] = ea / N
    BER_b_sim[i] = eb / N
    BER_c_sim[i] = ec / N

    print(f"E/N0={ENo_dB[i]:4d} dB  N={N:>7,}  "
          f"err_a={ea:5d} ({BER_a_sim[i]:.2e})  "
          f"err_b={eb:5d} ({BER_b_sim[i]:.2e})  "
          f"err_c={ec:5d} ({BER_c_sim[i]:.2e})")

# mask unreliable sim points (too few errors)
mask_a = errors_a >= MIN_RELIABLE
mask_b = errors_b >= MIN_RELIABLE
mask_c = errors_c >= MIN_RELIABLE

#  filter output waveforms 
sigma_demo = np.sqrt((E/10.0) / (2*dt))
rx_demo    = g + np.random.default_rng(99).normal(0, sigma_demo, Fs)

ya_full = np.convolve(rx_demo, h_matched) * dt
yc_full = np.convolve(rx_demo, h_rect)    * dt
t_conv  = np.arange(len(ya_full)) * dt

fig1, axes = plt.subplots(1, 3, figsize=(14, 4))
fig1.suptitle('Receive filter output  one received "1" bit, E/N$_0$ = 10 dB', fontsize=11)

axes[0].plot(t_conv, ya_full, color='steelblue')
axes[0].axvline(T, color='r', ls='--', lw=1, label=f'sample={ya_full[Fs-1]:.3f}')
axes[0].axhline(signal_a/2, color='k', ls=':', lw=1, label=f'thresh={signal_a/2:.3f}')
axes[0].set_title('(a) matched filter'); axes[0].set_xlabel('time (s)'); axes[0].legend(fontsize=8)

axes[1].plot(t, rx_demo, color='darkorange')
axes[1].axvline(t[-1], color='r', ls='--', lw=1, label=f'sample={rx_demo[-1]:.3f}')
axes[1].axhline(signal_b/2, color='k', ls=':', lw=1, label=f'thresh={signal_b/2:.3f}')
axes[1].set_title('(b) no filter  raw rx'); axes[1].set_xlabel('time (s)'); axes[1].legend(fontsize=8)

axes[2].plot(t_conv, yc_full, color='seagreen')
axes[2].axvline(T, color='r', ls='--', lw=1, label=f'sample={yc_full[Fs-1]:.3f}')
axes[2].axhline(signal_c/2, color='k', ls=':', lw=1, label=f'thresh={signal_c/2:.3f}')
axes[2].set_title('(c) rect filter'); axes[2].set_xlabel('time (s)'); axes[2].legend(fontsize=8)

plt.tight_layout()
fig1.savefig('filter_outputs.png', dpi=150, bbox_inches='tight')
print("saved filter_outputs.png")

#  BER plot 
fig2, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(ENo_dB, BER_a_th,  'b-',  lw=2, label='(a) matched  theory')
ax.semilogy(ENo_dB, BER_b_th,  'r-',  lw=2, label='(b) no filter  theory')
ax.semilogy(ENo_dB, BER_c_th,  'g-',  lw=2, label='(c) rect filter  theory')
# only plot sim where enough errors were observed
ax.semilogy(ENo_dB[mask_a], BER_a_sim[mask_a], 'b^', ms=6, label='(a) matched  sim')
ax.semilogy(ENo_dB[mask_b], BER_b_sim[mask_b], 'rs', ms=6, label='(b) no filter  sim')
ax.semilogy(ENo_dB[mask_c], BER_c_sim[mask_c], 'go', ms=6, label='(c) rect filter  sim')
ax.set_xlabel('E/N$_0$ (dB)', fontsize=12)
ax.set_ylabel('BER',          fontsize=12)
ax.set_title('BER vs E/N$_0$  three receive filter cases', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.set_ylim([1e-6, 1.0])
plt.tight_layout()
fig2.savefig('ber_vs_eno.png', dpi=150, bbox_inches='tight')
print("saved ber_vs_eno.png")
plt.show()