# Streamlit HI Spectrum Sandbox (tidied version)
# Implements requested refinements for UI layout, ranges, smoothing, styling.

import math
import numpy
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# -------------------------------
# Helpers
# -------------------------------

def top_hat(v, v0, width, height):
    half = 0.5 * width
    y = numpy.zeros_like(v)
    mask = (v >= (v0 - half)) & (v <= (v0 + half))
    y[mask] = height
    return y


def mass_to_sint(m_hi, distance_mpc):
    if distance_mpc <= 0:
        return 0.0
    return float(m_hi) / (2.356e5 * (distance_mpc ** 2))


def make_ripple(v, amp_jy, period_kms, phase_rad):
    if period_kms <= 0:
        return numpy.zeros_like(v)
    return amp_jy * numpy.sin(2.0 * math.pi * v / period_kms + phase_rad)


def make_polynomial(v, order, amp_jy):
    if order < 0:
        order = 0
    vmax = max(abs(float(v.min())), abs(float(v.max())), 1.0)
    x = v / vmax
    baseline = amp_jy * (x ** order)
    if order >= 1:
        baseline = baseline - baseline.mean()
    return baseline


def hanning_smooth(y, width):
    if width <= 0:
        return y
    window = numpy.hanning(width * 2 + 1)
    window /= window.sum()
    return numpy.convolve(y, window, mode="same")


def publication_axes(ax, ylabel_left, ylabel_right, title):
    ax.set_xlabel("Velocity  [km s$^{-1}$]")
    ax.set_ylabel(ylabel_left)
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.set_ylabel(ylabel_right)
    return ax2


def sn_axis_match(ax_left, ax_right, rms_jy):
    yl = ax_left.get_ylim()
    if rms_jy <= 0:
        rms_jy = 1.0
    ax_right.set_ylim(yl[0] / rms_jy, yl[1] / rms_jy)


def set_serif():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 0.9,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })

# -------------------------------
# UI
# -------------------------------
set_serif()
st.set_page_config(page_title="HI Spectrum Sandbox", layout="wide")
st.title("HI Spectrum Sandbox — top‑hat + toys")

# Source parameters row
col1, col2, col3, col4 = st.columns(4)
with col1:
    mode = st.radio("Specify by", ["Peak flux", "Integrated flux", "HI mass"], index=0)
with col2:
    width = st.slider("Line width [km s⁻¹]", min_value=10, max_value=1000, value=200, step=5)
with col3:
    v0 = st.slider("Line centre velocity v₀ [km s⁻¹]", min_value=-2000, max_value=2000, value=0, step=10)
with col4:
    if mode == "Peak flux":
        peak_mjy = st.slider("Peak flux [mJy]", min_value=0, max_value=500, value=50, step=1)
        s_peak_jy = peak_mjy / 1000.0
        s_int = s_peak_jy * float(width)
    elif mode == "Integrated flux":
        sint = st.slider("Integrated flux [Jy km s⁻¹]", min_value=0.0, max_value=500.0, value=10.0, step=0.1)
        s_peak_jy = sint / float(width) if width > 0 else 0.0
        s_int = sint
    else:
        mhi = st.number_input("HI mass [M☉]", min_value=0.0, max_value=1e11, value=5e9, step=1e9, format="%.3e")
        dist = st.slider("Distance [Mpc]", min_value=1, max_value=300, value=50, step=1)
        sint = mass_to_sint(mhi, float(dist))
        s_peak_jy = sint / float(width) if width > 0 else 0.0
        s_int = sint

# Velocity window row
col5, col6, col7 = st.columns(3)
with col5:
    v_span = st.slider("Velocity span [km s⁻¹]", min_value=100, max_value=5000, value=1500, step=50)
with col6:
    v_res = st.slider("Resolution [km s⁻¹ per channel]", min_value=1, max_value=100, value=10, step=1)
with col7:
    hann = st.slider("Hanning smoothing width", min_value=0, max_value=15, value=0, step=1)

# Noise and baseline row
col8, col9, col10 = st.columns(3)
with col8:
    rms_mjy = st.slider("RMS [mJy]", min_value=0.001, max_value=100.0, value=5.0, step=0.1)
    seed = st.slider("Random seed", min_value=0, max_value=1000, value=42, step=1)
with col9:
    ripple_amp_mjy = st.slider("Ripple amplitude [mJy]", min_value=0, max_value=200, value=0, step=1)
    ripple_period = st.slider("Ripple period [km s⁻¹]", min_value=10, max_value=2000, value=400, step=10)
    ripple_phase = st.slider("Ripple phase [rad]", min_value=0.0, max_value=2.0 * math.pi, value=0.0, step=0.05)
with col10:
    poly_order = st.slider("Polynomial order", min_value=0, max_value=6, value=1, step=1)
    poly_amp_mjy = st.slider("Polynomial amplitude [mJy]", min_value=0, max_value=500, value=0, step=5)

# -------------------------------
# Build the spectrum
# -------------------------------
half = 0.5 * float(v_span)
nbin = int(max(2, round(v_span / float(v_res))))
v = numpy.linspace(-half, half, nbin)

y_sig = top_hat(v, v0=v0, width=float(width), height=float(s_peak_jy))

ripple = make_ripple(v, amp_jy=ripple_amp_mjy / 1000.0, period_kms=float(ripple_period), phase_rad=float(ripple_phase))
poly = make_polynomial(v, order=int(poly_order), amp_jy=poly_amp_mjy / 1000.0)

if seed != 0:
    numpy.random.seed(int(seed))
noise = numpy.random.normal(loc=0.0, scale=rms_mjy / 1000.0, size=v.shape)

# Final spectrum (Jy)
y_total = y_sig + ripple + poly + noise
y_total = hanning_smooth(y_total, hann)

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.plot(v, y_total, lw=0.9, label="Observed")
ax.plot(v, (y_sig + ripple + poly), lw=0.8, alpha=0.9, label="Source and baseline")
ax.plot(v, y_sig, lw=0.8, alpha=0.9, label="Source profile")

publication_axes(ax, ylabel_left="Flux density [Jy]", ylabel_right="S/N per channel", title="Synthetic HI spectrum")
ax.legend(loc="upper right", frameon=False, fontsize=9)

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax2 = ax.twinx()
sn_axis_match(ax, ax2, rms_jy=max(rms_mjy / 1000.0, 1e-12))
ax2.set_ylabel("S/N per channel")

st.pyplot(fig, clear_figure=True)

# -------------------------------
# Summary
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
