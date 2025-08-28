# Streamlit HI Spectrum Sandbox (compact, S/N fix)
# Implements compact UI rows, capped RMS, ALFALFA-style integrated S/N, and fixes duplicate S/N axis.

import math
import numpy
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 1) SUBROUTINES TO GENERATE THE SPECTRA

# Top-hat profile for the source itself
def top_hat(v, v0, width, height):
    half = 0.5 * width
    y = numpy.zeros_like(v)
    mask = (v >= (v0 - half)) & (v <= (v0 + half))
    y[mask] = height
    return y

# Convert HI mass to total flux
def mass_to_sint(m_hi, distance_mpc):
    if distance_mpc <= 0:
        return 0.0
    return float(m_hi) / (2.356e5 * (distance_mpc ** 2))

# Sinusoidal ripple in the baseline
def make_ripple(v, amp_jy, period_kms, phase_rad):
    if period_kms <= 0:
        return numpy.zeros_like(v)
    return amp_jy * numpy.sin(2.0 * math.pi * v / period_kms + phase_rad)

# Polynomial component of the baseline
def make_polynomial(v, order, amp_jy):
    if order < 0:
        order = 0
    vmax = max(abs(float(v.min())), abs(float(v.max())), 1.0)
    x = v / vmax
    baseline = amp_jy * (x ** order)
    if order >= 1:
        baseline = baseline - baseline.mean()
    return baseline

# Apply Hanning smoothing to an input spectrum
def hanning_smooth(y, width):
    # Symmetric Hanning of integer half-width; total window = 2*width+1.
    # Width 0 disables
    if width <= 0:
        return y
    window = numpy.hanning(width * 2 + 1)
    window /= window.sum()
    return numpy.convolve(y, window, mode="same")

# Use a serif font (why is this done here and not just at the start ?) and set the axes font and size
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

# Claculate the integrated S/N
def alfalpha_sn(s_int_jykms, width_kms, rms_mjy, dv_kms):
    """ALFALFA-style integrated S/N (idealised):
    S/N = (1000 * (S_int / W)) / rms_mJy * sqrt(W / (2 * dv)).
    """
    if width_kms <= 0 or dv_kms <= 0 or rms_mjy <= 0:
        return 0.0
    term1 = 1000.0 * (s_int_jykms / width_kms) / rms_mjy
    term2 = math.sqrt(width_kms / (2.0 * dv_kms))
    if width_kms > 400.0:
        term1 = 1000.0 * (s_int_jykms / width_kms) / rms_mjy
        term2 = math.sqrt(400.0 / (2.0 * dv_kms))
    return float(term1 * term2)


# 2)GUI

# Apply serif fonts, display the title and basic description
set_serif()
st.set_page_config(page_title="HI Spectrum Sandbox", layout="wide")
st.title("Spectrum Simulator")
st.write('### Generate example HI spectra based on simple parameters for the source and noise properties')
st.write('Demonstrates the typical appearance of an HI spectrum of specified properties. You can adjust the source width, flux, change the properties of the noise, and apply different smoothing levels. Sources always have top-hat profiles. You can enter their parameters either as physical (HI mass with distance) or observationally (peak or total flux). Changing their parameters shows how source detectability can vary.')
st.write('Note that by default the noise is purely random-Gaussian. Set the "seed" value to be above zero to keep the noise fixed, otherwise it will be rengenerated every time you update any input parameters. You can also make the baseline more realistic by adding a sinusoid and/or a polynomial.')
st.write('Currently the source profile is a simple top-hat function - about 25% of galaxies show such profiles. Other profiles will be added eventually, maybe.')

# Row 1: Input the source parameters 
col1, col2, col3, col4, col5 = st.columns([1.1, 1, 1, 1.2, 1.2])
with col1:
    mode = st.radio("Specify by", ["Peak flux", "Integrated flux", "HI mass"], index=0)
with col2:
    width = st.slider("Line width [km s⁻¹]", min_value=10, max_value=1000, value=200, step=5, help='Observed line width of the source')
with col3:
    v0 = st.slider("Line centre v₀ [km s⁻¹]", min_value=-2000, max_value=2000, value=0, step=10, help='Shift the line centre relative to the centre of the spectrum')
with col4:
    if mode == "Peak flux":
        peak_mjy = st.slider("Peak flux [mJy]", min_value=0, max_value=500, value=50, step=1, help='Peak flux of the source')
        s_peak_jy = peak_mjy / 1000.0
        s_int = s_peak_jy * float(width)
    elif mode == "Integrated flux":
        sint = st.slider("Integrated flux [Jy km s⁻¹]", min_value=0.0, max_value=500.0, value=10.0, step=0.1, help='Total flux in the source')
        s_peak_jy = sint / float(width) if width > 0 else 0.0
        s_int = sint
    else:
        mhi = st.number_input("HI mass [M☉]", min_value=0.0, max_value=1e11, value=5e9, step=1e7, format="%.3e", help='Total HI mass of the source. Be sure to set the distance value as well !')
        dist = st.slider("Distance [Mpc]", min_value=1, max_value=300, value=50, step=1, help='Set the distance to the source, if entering the mass')
        sint = mass_to_sint(mhi, float(dist))
        s_peak_jy = sint / float(width) if width > 0 else 0.0
        s_int = sint
with col5:
    st.markdown("\n")  # spacer for alignment
    st.markdown("**Peak**: {:.3f} Jy  \
**Total**: {:.3f} Jy km s⁻¹".format(s_peak_jy, s_int))

# Row 2: Input baseline and Gaussian noise components
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    v_span = st.slider("Total baseline [km s⁻¹]", min_value=100, max_value=5000, value=1500, step=50, help='Change the length of the baseline shown')
with c2:
    v_res = st.slider("Resolution [km s⁻¹]", min_value=1, max_value=100, value=10, step=1, help='Set the spectral resolution')
with c3:
    hann = st.slider("Hanning width", min_value=0, max_value=15, value=0, step=1, help='Set the level of Hanning smoothing (0 to disable)')
with c4:
    rms_mjy = st.slider("RMS [mJy]", min_value=0.001, max_value=50.0, value=5.0, step=0.1, help='Set the base noise level. Note that this does NOT include the effects of the (optional) ripple and/or polynomial')
with c5:
    seed = st.slider("Seed (0 = random)", min_value=0, max_value=1000, value=0, step=1, help='Set the random seed for generating the Gaussian noise. If zero, the noise will be regenerated whenever any parameters are updated')

# Row 3: Input parameters for additional noise as a sinusoid and polynomial
b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    ripple_amp_mjy = st.slider("Ripple amplitude [mJy]", min_value=0, max_value=200, value=0, step=1, help='Strength of the optional baseline ripple, calculated a sinusoid')
with b2:
    ripple_period = st.slider("Ripple period [km s⁻¹]", min_value=10, max_value=2000, value=400, step=10, help='Period of the optional baseline sinusoidal ripple')
with b3:
    ripple_phase = st.slider("Phase [radians]", min_value=0.0, max_value=2.0 * math.pi, value=0.0, step=0.05, help='Phase offset of the ripple at the central velocity')
with b4:
    poly_order = st.slider("Polynomial order", min_value=0, max_value=6, value=1, step=1, help='Order of the polynomial offset applied to the baseline')
with b5:
    poly_amp_mjy = st.slider("Polynomial amplitude [mJy]", min_value=0, max_value=500, value=0, step=5, help='Strength of the polynomial offset')

# Row 4 : toggle plotting the different spectral components
f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    show_components = st.toggle('Plot components', value=False, help='If enabled, also shows the individual model components of the baseline and signal')


# Now we can actually generate the spectrum
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

# Plot the results !
fig, ax = plt.subplots(figsize=(8.6, 4.4))
if show_components == True:
    ax.plot(v, y_sig, lw=0.5, alpha=1.0, label="Source profile", color='green')
    ax.plot(v, (y_sig + ripple + poly), lw=0.5, alpha=0.9, label="Source and baseline", color='orange')
ax.plot(v, y_total, lw=0.5, label="Observed", color='blue')

# Publication-style labels and ticks
ax.set_xlabel("Velocity  [km s$^{-1}$]")
ax.set_ylabel("Flux density [Jy]")
ax.set_title("Synthetic HI spectrum")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Second (right hand) axis for equivalent S/N
ax2 = ax.twinx()
yl = ax.get_ylim()
ax2.set_ylim(yl[0] / max(rms_mjy / 1000.0, 1e-12), yl[1] / max(rms_mjy / 1000.0, 1e-12))
ax2.set_ylabel("S/N per channel")

st.pyplot(fig, clear_figure=True)

st.write('The spectrum shows what would be observed (blue), the raw source profile (green), and also the combination of source plus the baseline without the random noise (orange).')
st.write('Numerical summary of the inputs - these do NOT account for the noise in measuring the source. The integrated S/N is defined in Saintonge 2007; values above 6.5 tend to be reliable detections.')

# Finally, print a simple numerical summary of the inputs
int_sn = alfalpha_sn(s_int, float(width), float(rms_mjy), float(v_res))
st.markdown(
    "**Channels**: {}  |  **Resolution**: {:.1f} km s⁻¹  |  **Span**: {:.0f} km s⁻¹  |  **Width**: {:.0f} km s⁻¹  |  "
    "**Peak flux**: {:.3f} Jy  |  **Total flux**: {:.3f} Jy km s⁻¹  |  **RMS**: {:.3f} mJy  |  **Integrated S/N**: {:.2f}".format(
        nbin, float(v_res), float(v_span), float(width), float(s_peak_jy), float(s_int), float(rms_mjy), float(int_sn)
    )
)
