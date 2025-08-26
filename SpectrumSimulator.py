# Streamlit HI Spectrum Sandbox
# Rhysy — simple, parametric, publication‑style HI spectrum toy
# Python ≥ 3.8 recommended for Streamlit, but core maths kept simple and clear.

import math
import numpy
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# -------------------------------
# Helpers
# -------------------------------

def top_hat(v, v0, width, height):
    """Return a top‑hat profile of given centre v0, full width, and peak height (Jy)."""
    half = 0.5 * width
    y = numpy.zeros_like(v)
    mask = (v >= (v0 - half)) & (v <= (v0 + half))
    y[mask] = height
    return y


def mass_to_sint(m_hi, distance_mpc):
    """Convert HI mass (Msun) to integrated flux S_int (Jy km/s) via M_HI = 2.356e5 D^2 S_int.
    Returns S_int.
    """
    if distance_mpc <= 0:
        return 0.0
    return float(m_hi) / (2.356e5 * (distance_mpc ** 2))


def make_ripple(v, amp_jy, period_kms, phase_rad):
    if period_kms <= 0:
        return numpy.zeros_like(v)
    return amp_jy * numpy.sin(2.0 * math.pi * v / period_kms + phase_rad)


def make_polynomial(v, order, amp_jy):
    """A very simple polynomial baseline: amp * (v / vmax) ** order, centred to have mean ≈ 0 for order ≥ 1."""
    if order < 0:
        order = 0
    vmax = max(abs(float(v.min())), abs(float(v.max())), 1.0)
    x = v / vmax
    baseline = amp_jy * (x ** order)
    if order >= 1:
        baseline = baseline - baseline.mean()
    return baseline


def publication_axes(ax, ylabel_left, ylabel_right, title):
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_xlabel("Velocity  [km s$^{-1}$]")
    ax.set_ylabel(ylabel_left)
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.set_ylabel(ylabel_right)
    return ax2


def sn_axis_match(ax_left, ax_right, rms_jy):
    # Map left y‑limits (Jy) to right y‑limits (S/N)
    yl = ax_left.get_ylim()
    if rms_jy <= 0:
        rms_jy = 1.0  # avoid zero division; S/N axis becomes arbitrary
    ax_right.set_ylim(yl[0] / rms_jy, yl[1] / rms_jy)


def set_serif():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


# -------------------------------
# UI
# -------------------------------
set_serif()
st.set_page_config(page_title="HI Spectrum Sandbox", layout="wide")
st.title("HI Spectrum Sandbox — top‑hat + toys")

left, right = st.columns([1, 1])

with left:
    st.subheader("Velocity window and sampling")
    v_span = st.slider("Velocity span (full)  [km s⁻¹]", min_value=100, max_value=5000, value=1500, step=50)
    v_res = st.slider("Velocity resolution  [km s⁻¹ per channel]", min_value=1, max_value=100, value=10, step=1)
    v0 = st.slider("Line centre velocity  v₀  [km s⁻¹]", min_value=-2000, max_value=2000, value=0, step=10)

    st.subheader("Source strength")
    mode = st.radio("Specify by", ["Peak flux", "Integrated flux", "HI mass"], index=0)

    if mode == "Peak flux":
        peak_mjy = st.slider("Peak flux density  [mJy]", min_value=0, max_value=500, value=50, step=1)
        s_peak_jy = peak_mjy / 1000.0
        width = st.slider("Line width (full)  [km s⁻¹]", min_value=10, max_value=1000, value=200, step=5)
        s_int = s_peak_jy * float(width)
    elif mode == "Integrated flux":
        sint = st.slider("Integrated flux S_int  [Jy km s⁻¹]", min_value=0.0, max_value=500.0, value=10.0, step=0.1)
        width = st.slider("Line width (full)  [km s⁻¹]", min_value=10, max_value=1000, value=200, step=5)
        s_peak_jy = sint / float(width) if width > 0 else 0.0
        s_int = sint
    else:
        mhi = st.slider("HI mass  [10^9 M☉]", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        dist = st.slider("Distance  [Mpc]", min_value=1, max_value=300, value=50, step=1)
        width = st.slider("Line width (full)  [km s⁻¹]", min_value=10, max_value=1000, value=200, step=5)
        sint = mass_to_sint(mhi * 1e9, float(dist))
        s_peak_jy = sint / float(width) if width > 0 else 0.0
        s_int = sint

with right:
    st.subheader("Baseline and noise")
    rms_mjy = st.slider("RMS noise per channel  [mJy]", min_value=0, max_value=100, value=5, step=1)
    seed = st.number_input("Random seed (noise)", min_value=0, value=42, step=1)

    st.markdown("**Sinusoidal ripple**")
    ripple_amp_mjy = st.slider("Amplitude  [mJy]", min_value=0, max_value=200, value=0, step=1)
    ripple_period = st.slider("Period  [km s⁻¹]", min_value=10, max_value=2000, value=400, step=10)
    ripple_phase = st.slider("Phase  [rad]", min_value=0.0, max_value=2.0 * math.pi, value=0.0, step=0.05)

    st.markdown("**Polynomial baseline**")
    poly_order = st.slider("Order", min_value=0, max_value=6, value=1, step=1)
    poly_amp_mjy = st.slider("Amplitude scaling  [mJy]", min_value=0, max_value=500, value=0, step=5)

# -------------------------------
# Build the spectrum
# -------------------------------
# Velocity grid, centred about zero by default
half = 0.5 * float(v_span)
nbin = int(max(2, round(v_span / float(v_res))))
v = numpy.linspace(-half, half, nbin)

# Signal (Jy)
y_sig = top_hat(v, v0=v0, width=float(width), height=float(s_peak_jy))

# Baselines (Jy)
ripple = make_ripple(v, amp_jy=ripple_amp_mjy / 1000.0, period_kms=float(ripple_period), phase_rad=float(ripple_phase))
poly = make_polynomial(v, order=int(poly_order), amp_jy=poly_amp_mjy / 1000.0)

# Noise (Jy)
numpy.random.seed(int(seed))
noise = numpy.random.normal(loc=0.0, scale=rms_mjy / 1000.0, size=v.shape)

# Final spectrum (Jy)
y_total = y_sig + ripple + poly + noise

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.plot(v, y_total, lw=1.2, label="Observed")
ax.plot(v, (y_sig + ripple + poly), lw=1.0, alpha=0.9, label="Signal+baseline (no noise)")
ax.plot(v, y_sig, lw=1.0, alpha=0.9, label="Top‑hat line")

publication_axes(ax, ylabel_left="Flux density  [Jy]", ylabel_right="S/N per channel", title="Synthetic HI spectrum (parametric)")
ax.legend(loc="upper right", frameon=False)

# Nice tick density
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Right y‑axis = S/N
ax2 = ax.twinx()
sn_axis_match(ax, ax2, rms_jy=max(rms_mjy / 1000.0, 1e-12))
ax2.set_ylabel("S/N per channel")

st.pyplot(fig, clear_figure=True)

# -------------------------------
# Readouts
# -------------------------------
with st.expander("Numerical summary"):
    st.write({
        "Channels": nbin,
        "Resolution_km_s": float(v_res),
        "Velocity_span_km_s": float(v_span),
        "Width_km_s": float(width),
        "Peak_flux_Jy": float(s_peak_jy),
        "Integrated_flux_Jy_km_s": float(s_int),
        "RMS_Jy": float(rms_mjy) / 1000.0,
    })
