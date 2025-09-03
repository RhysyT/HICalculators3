
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
    "- $\\Sigma_{\\rm H₂} = \\alpha_{\\rm CO}\\,I_{\\rm CO}$  with $I_{\\rm CO} = \\int T_{\\rm mb}\\,dv$ (K km s⁻¹) (i.e. the intensity _I_ value is just the temperature multipled by the line width)\n"
    "- $T_{\\rm mb} = \\dfrac{M_{\\rm H_2}}{\\alpha_{\\rm CO}\\,\\Delta V\\,A_{\\rm beam}\\,N_\\sigma}$ Where the temperature is that for the sensitivty to detect the CO per the above equation.\n"
    "- $T_A^* = T_{\\rm mb}\\,(B_{\\rm eff}/F_{\\rm eff})$ : this converts the above main-beam temperature to the actual observed antenna temperature needed for sensitivity calculations.\n\n"
)

# ------------------------------
# Inputs
# ------------------------------
with st.container():
    st.subheader("Global inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mh2 = st.number_input("Target H₂ mass **per beam, per channel** (M☉)", min_value=0.0, value=1e6, step=1e5, format="%.6g")
    with c2:
        dv = st.number_input("Velocity resolution ΔV (km s⁻¹)", min_value=0.001, value=5.0, step=0.5, format="%.6g")
    with c3:
        nsig = st.number_input("Detection level Nσ", min_value=0.1, value=5.0, step=0.5, format="%.3g")
    with c4:
        dist_val = st.number_input("Source distance value", min_value=0.0, value=17.0, step=1.0, format="%.6g")
        dist_unit = st.selectbox("Distance unit", options=["pc", "kpc", "Mpc"], index=2)

    distance_pc = distance_to_pc(dist_val, dist_unit)

st.markdown("---")

# Two-line inputs: CO(1–0) and CO(2–1)
st.subheader("Line-specific inputs")

col10, col21 = st.columns(2)

with col10:
    st.markdown("### CO(1–0)")
    b10_val = st.number_input("Beam FWHM — CO(1–0)", min_value=0.0, value=22.0, step=0.1, format="%.6g", key="b10_val")
    b10_unit = st.selectbox("Beam unit (1–0)", options=["arcsec", "arcmin"], index=0, key="b10_unit")
    a10 = st.number_input("α_CO (1–0) [M☉ pc⁻² (K km s⁻¹)⁻¹]", min_value=0.0, value=4.35, step=0.05, format="%.6g", key="a10")
    fe10 = st.number_input("F_eff (1–0)", min_value=0.0, max_value=1.0, value=0.94, step=0.01, format="%.3f", key="fe10")
    be10 = st.number_input("B_eff (1–0)", min_value=0.0, max_value=1.0, value=0.78, step=0.01, format="%.3f", key="be10")

with col21:
    st.markdown("### CO(2–1)")
    b21_val = st.number_input("Beam FWHM — CO(2–1)", min_value=0.0, value=11.0, step=0.1, format="%.6g", key="b21_val")
    b21_unit = st.selectbox("Beam unit (2–1)", options=["arcsec", "arcmin"], index=0, key="b21_unit")
    a21 = st.number_input("α_CO (2–1) [M☉ pc⁻² (K km s⁻¹)⁻¹]", min_value=0.0, value=6.70, step=0.05, format="%.6g", key="a21")
    fe21 = st.number_input("F_eff (2–1)", min_value=0.0, max_value=1.0, value=0.92, step=0.01, format="%.3f", key="fe21")
    be21 = st.number_input("B_eff (2–1)", min_value=0.0, max_value=1.0, value=0.59, step=0.01, format="%.3f", key="be21")

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
        st.write(f"Beam FWHM (physical): **{nice_float(res['fwhm_pc'], 4)} pc**")
        st.write(f"Beam area (Gaussian): **{nice_float(res['area_pc2'], 4)} pc²**")
    with c2:
        st.write(f"Surface-density sensitivity (1σ, per channel): **{nice_float(res['sigma_1sigma'], 4)} M☉ pc⁻²**")
        st.write(f"Main-beam temperature RMS (1σ): **{nice_float(res['tmb_rms_K'], 4)} K**  ({nice_float(tmb_rms_mK, 4)} mK)")
    with c3:
        st.markdown(f"**Enter into IRAM ETC (T_A* RMS, per channel):**")
        st.markdown(f"### **{nice_float(ta_rms_mK, 4)} mK**")
        st.caption(f"(= {nice_float(res['ta_rms_K'], 4)} K)")

render_results("CO(1–0)", res10)
render_results("CO(2–1)", res21)

st.caption(
    "Notes: Results are **per synthesized beam** and **per velocity channel ΔV**. "
    "If your source uniformly fills many beams, total mass sensitivity scales roughly with the number of beams covering the source; "
    "for clumpy emission, detectability is governed by peak surface brightness per synthesized beam. "
    "Defaults used here: α_CO(1–0)=4.35, α_CO(2–1)=6.70 (assuming R21≈0.65), F_eff(115 GHz)=0.94, B_eff(115 GHz)=0.78; "
    "F_eff(230 GHz)=0.92, B_eff(230 GHz)=0.59."
)
