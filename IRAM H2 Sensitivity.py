
import math
import streamlit as st

# Basic conversion factors
ARCSEC_TO_RAD = 1.0 / 206265.0
ARCMIN_TO_RAD = ARCSEC_TO_RAD * 60.0

def angle_to_rad(value, unit):
    if unit == "arcsec":
        return value * ARCSEC_TO_RAD
    elif unit == "arcmin":
        return value * ARCMIN_TO_RAD
    else:
        raise ValueError("Unit must be 'arcsec' or 'arcmin'.")

def distance_to_pc(value, unit):
    if unit == "pc":
        return value
    elif unit == "kpc":
        return value * 1_000.0
    elif unit == "Mpc":
        return value * 1_000_000.0
    else:
        raise ValueError("Distance unit must be 'pc', 'kpc', or 'Mpc'.")

def beam_physical_fwhm_pc(theta_fwhm_rad, distance_pc):
    # Linear FWHM in pc
    return theta_fwhm_rad * distance_pc

def gaussian_beam_area_pc2(fwhm_pc):
    # A_beam = 1.133 * FWHM^2  (Gaussian beam area in physical units)
    return 1.133 * (fwhm_pc ** 2)

def tmb_rms_required(mh2_msun, alpha_co, dv_kms, area_pc2, n_sigma):
    # T_mb (1σ, per channel) required such that Nσ * T_rms reaches the target mass per channel
    # Tmb_rms = MH2 / (alpha_co * dv_kms * area_pc2 * N_sigma)
    if alpha_co <= 0 or dv_kms <= 0 or area_pc2 <= 0 or n_sigma <= 0:
        return float("nan")
    return mh2_msun / (alpha_co * dv_kms * area_pc2 * n_sigma)

def ta_star_from_tmb(tmb, feff, beff):
    # T_mb = (F_eff / B_eff) * T_A*  =>  T_A* = T_mb * (B_eff / F_eff)
    if feff <= 0:
        return float("nan")
    return tmb * (beff / feff)

def surface_density_1sigma(alpha_co, tmb_rms, dv_kms):
    # Σ_1σ (per channel) = α_CO * Tmb_rms * ΔV
    return alpha_co * tmb_rms * dv_kms

def mk(value_K):
    # convert K to mK
    return value_K * 1e3

def nice_float(x, digits=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    fmt = f"{{:.{digits}g}}"
    return fmt.format(x)


# GUI
st.set_page_config(page_title="CO Sensitivity Calculator", layout="wide")
st.title("Mass-Based CO Sensitivity Calculator")
st.write('### Calculate H₂ sensitivity levels using physical parameters')
st.write('CO sensitivity estimates often work in surface brightness and temperature, which makes working out how much H₂ mass you can calculate extremely tiresome. This tool is an attempt to get around that. Provide the total mass, the line width enclosing the mass (this could be the velocity resolution of the observations or the line width of the source, depending on what you want to detect), and the area of the beam, and it will do the hard work for you.')
st.write("This calculator outputs various parameters you can use in any telescope to calculate how much observing time you'll need. It also outputs the antenna temperature for the IRAM 30m single-dish, which you can enter directly into the [IRAM Exposure Time Calculator](https://oms.iram.fr/tse/). Works for two lines at once -- by default these are CO(1-0) and CO(2-1), but you can adjust their parameters manually.")

st.info(
    "**Reference notes**\n"
    "- $\\Sigma_{\\rm H₂} = \\alpha_{\\rm CO}\\,I_{\\rm CO}$  with $I_{\\rm CO} = \\int T_{\\rm mb}\\,dv$ (K km s⁻¹)"+"\n"
    "In essence, the intensity _I_ value is just the temperature multipled by the line width)"+"\n"
    "- $T_{\\rm mb} = \\dfrac{M_{\\rm H_2}}{\\alpha_{\\rm CO}\\,\\Delta V\\,A_{\\rm beam}\\,N_\\sigma}$"+"\n"
    "Where the temperature is that for the sensitivty to detect the CO per the above equation, and N$\sigma$ is the requested peak S/N for a detection."+"\n"
    "- $T_A^* = T_{\\rm mb}\\,(B_{\\rm eff}/F_{\\rm eff})$"+"\n"
    "This converts the above main-beam temperature to the actual observed antenna temperature needed for sensitivity calculations. The $alpha$ factors convert between CO and H₂ and are metallicity dependent; note also that the relative strength of the CO lines depends on the physical properties of the gas."
)

# ------------------------------
# Inputs
# ------------------------------
with st.container():
    st.subheader("Source properties")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        mh2 = st.number_input("Target H₂ mass [M$_{☉}$]", min_value=0.0, value=1e6, step=1e5, format="%.6g", help='Total H₂ within the beam in the specified velocity width')
    with c2:
        dv = st.number_input("Velocity width ΔV [km s⁻¹]", min_value=0.001, value=5.0, step=0.5, format="%.6g", help='This can be the velocity resolution of the instrument (if you need the line resolved, i.e. detected in every channel), or a smoothed width e.g. the total width of the line (if you just need sheer sensitivity for a detection)')
    with c3:
        nsig = st.number_input("Peak S/N", min_value=0.1, value=5.0, step=0.5, format="%.3g", help='Nσ in the above equation. Sets the statistical significance of the detection')
    with c4:
        dist_val = st.number_input("Source distance", min_value=0.0, value=17.0, step=1.0, format="%.6g", help='Distance to the souce, used in calculating the physical area of the beam')
    with c5:
        dist_unit = st.selectbox("Distance unit", options=["pc", "kpc", "Mpc"], index=2)

    distance_pc = distance_to_pc(dist_val, dist_unit)

# Horizontal line
#st.markdown("---")

# Two-line inputs: CO(1–0) and CO(2–1)
st.subheader("Telescope and line parameters")

colA, colB, colC, colD = st.columns(4)
# Beam : Unit
# alpha, F, B

with colA:
    st.markdown("### CO(1–0)")
    b10_val = st.number_input("Beam FWHM", min_value=0.0, value=22.4, step=0.1, format="%.6g", key="b10_val", help='Default is the IRAM 30m resolution in arcseconds at z=0')
with colB:
    st.markdown("### ", unsafe_allow_html=True)
    b10_unit = st.selectbox("Beam unit", options=["arcsec", "arcmin"], index=0, key="b10_unit")

with colC:
    st.markdown("### CO(2–1)")
    b21_val = st.number_input("Beam FWHM", min_value=0.0, value=10.7, step=0.1, format="%.6g", key="b21_val",  help='Default is the IRAM 30m resolution in arcseconds at z=0')
with colD:
    st.markdown("### ", unsafe_allow_html=True)
    b21_unit = st.selectbox("Beam unit", options=["arcsec", "arcmin"], index=0, key="b21_unit")

colE, colF, colG, colH, colI, colJ = st.columns(6)

with colE:
    a10 = st.number_input("α$_{\\rm CO}$", min_value=0.0, value=4.35, step=0.05, format="%.6g", key="a10", help='Essentially a conversion factor relating the CO to H₂ intensity. Depends on the nature of the system; default 4.35 is typical for CO(1-0) in the Milky Way. Units are [M☉ pc⁻² (K km s⁻¹)⁻¹]')
with colF:
    fe10 = st.number_input("F$_{eff}$", min_value=0.0, max_value=1.0, value=0.94, step=0.01, format="%.3f", key="fe10", help='Forward efficiency at this frequency, used for converting main beam temperature to antenna temperature. Default is for IRAM 30 m at CO(1-0) frequencies')
with colG:
    be10 = st.number_input("B$_{eff}$", min_value=0.0, max_value=1.0, value=0.78, step=0.01, format="%.3f", key="be10", help='Main beam efficiency at this frequency, used for converting main beam temperature to antenna temperature. Default is for IRAM 30 m at CO(1-0) frequencies')

with colH:
    a21 = st.number_input("α$_{\\rm CO}$", min_value=0.0, value=6.70, step=0.05, format="%.6g", key="a21", help='Essentially a conversion factor relating the CO to H₂ intensity. Depends on the nature of the system; default 6.70 is typical for CO(2-1) in the Milky Way. Units are [M☉ pc⁻² (K km s⁻¹)⁻¹]')
with colI:
    fe21 = st.number_input("F$_{eff}$", min_value=0.0, max_value=1.0, value=0.92, step=0.01, format="%.3f", key="fe21", help='Forward efficiency at this frequency, used for converting main beam temperature to antenna temperature. Default is for IRAM 30 m at CO(2-0) frequencies')
with colJ:
    be21 = st.number_input("B$_{eff}$", min_value=0.0, max_value=1.0, value=0.59, step=0.01, format="%.3f", key="be21", help='Main beam efficiency at this frequency, used for converting main beam temperature to antenna temperature. Default is for IRAM 30 m at CO(2-1) frequencies')     

st.markdown("---")

# ------------------------------
# Calculations
# ------------------------------
def compute_block(beam_val, beam_unit, alpha_co, feff, beff):
    # Angles & physical scales
    theta_rad = angle_to_rad(beam_val, beam_unit)
    fwhm_pc = beam_physical_fwhm_pc(theta_rad, distance_pc)
    area_pc2 = gaussian_beam_area_pc2(fwhm_pc)

    # Temperatures
    tmb_rms = tmb_rms_required(mh2, alpha_co, dv, area_pc2, nsig)  # K
    ta_rms = ta_star_from_tmb(tmb_rms, feff, beff)                  # K

    # Surface density sensitivity (1σ, per channel)
    sigma_1sigma = surface_density_1sigma(alpha_co, tmb_rms, dv)    # M_sun / pc^2

    return {
        "theta_rad": theta_rad,
        "fwhm_pc": fwhm_pc,
        "area_pc2": area_pc2,
        "tmb_rms_K": tmb_rms,
        "ta_rms_K": ta_rms,
        "sigma_1sigma": sigma_1sigma,
    }

res10 = compute_block(b10_val, b10_unit, a10, fe10, be10)
res21 = compute_block(b21_val, b21_unit, a21, fe21, be21)

# ------------------------------
# Outputs
# ------------------------------
st.subheader("Results")

def render_results(title, res):
    tmb_rms_mK = mk(res["tmb_rms_K"])
    ta_rms_mK = mk(res["ta_rms_K"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**{title}**")
        st.write(f"Beam FWHM : **{nice_float(res['fwhm_pc'], 4)} pc**")
        st.write(f"Beam area : **{nice_float(res['area_pc2'], 4)} pc²**")
    with c2:
        st.write(f"Surface-density sensitivity (1σ, per spectral resolution element): **{nice_float(res['sigma_1sigma'], 4)} M<sub>☉</sub> pc⁻²**", unsafe_allow_html=True)
        st.write(f"Main-beam temperature $rms$: **{nice_float(res['tmb_rms_K'], 4)} K**  ({nice_float(tmb_rms_mK, 4)}&nbsp;mK)")
    with c3:
        st.markdown(f"**IRAM 30m antenna temperatures for the ETC :**")
        st.markdown(f"### **{nice_float(ta_rms_mK, 4)} mK**")
        st.caption(f"{nice_float(res['ta_rms_K'], 4)} K")

render_results("CO(1–0)", res10)
render_results("CO(2–1)", res21)

st.caption("The beam size is dependent on the source frequency. To calculate the exact value use [this online tool](https://whosespectrallineisitanyway.streamlit.app/). Enter this in the (IRAM Exposure Time Calculator)[https://oms.iram.fr/tse/] : the spatial resolution is included in the 'Results' box on the left. Enter the calculated antenna values in the same ETC to calculate how much observing time is required. Official IRAM beam efficiencies are available [here](https://publicwiki.iram.es/Iram30mEfficiencies).")
st.caption("Note that for an interferometer, it can be difficult and potentially meaningless to calculate a sensitivity value for a specified mass. For this you would assume the CO fills the synthesised beam (angular resolution). Given the typically very small beam, this can mean an interferometer like ALMA or NOEMA is sensitive to much smaller total masses than a single dish --  but this is only true if the gas is clumpy ! If the same mass as is detectable in a single synthesised beam is more spread out, it may not be detected. See the explanatory [cheat sheet here](http://www.rhysy.net/.cm4all/mediadb/Resources/Cheat%20Sheet.pdf).")
st.caption("Reference integration times at the IRAM 30m :")
st.caption("CO(1-0) at 10, 5, 1 mK = 35 minutes, 2.3 hours, 58 hours (5 km/s resolution)")
st.caption("CO(2-1) at 10, 5, 1 mK = 5 minutes, 20 minutes, 8 hours (5 km/s resolution)")
           
