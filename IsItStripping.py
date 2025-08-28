# app.py
# Streamlit app to compute required ram pressure (p_def) and local ram pressure (p_loc)
# following the methodology of Köppen et al. (2018).
#
# Units throughout:
# - Radii: kpc
# - Masses: Msun
# - Surface densities for user-visible steps: Msun/pc^2
# - Speeds: km/s
# - Pressures reported in: cm^-3 (km/s)^2, and in [1000 cm^-3 (km/s)^2]

import math
import streamlit as st

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    submitted = True  # compute once on first load

# ---------- Physical constants ----------
G = 4.300917270e-6  # gravitational constant in (kpc * (km/s)^2) / Msun
mp_g = 1.67262192369e-24  # proton mass [g]
MSUN_PC2_TO_G_CM2 = 2.089e-4  # 1 Msun/pc^2 to g/cm^2
KPC_TO_CM = 3.085677581e21  # 1 kpc in cm

# The "shortcut" conversion constant K that maps:
#   p[cm^-3 (km/s)^2] = K * (Sigma[Msun/pc^2] * v^2[km^2/s^2] / r[kpc]) * (1 + a e^{-r/R0})
# when the geometrical factor g is included separately as 1/g.
# Derivation (see chat): K = (MSUN_PC2_TO_G_CM2) / (KPC_TO_CM * mp_g)
K_CONV = MSUN_PC2_TO_G_CM2 / (KPC_TO_CM * mp_g)  # ~ 0.040484... (before dividing by g)
# We'll divide by g explicitly in the formula, so no g folded into K_CONV.

# Mathematical functions
def f_nfw(x: float) -> float:
    """f(x) = ln(1+x) - x/(1+x)."""
    if x <= 0:
        return 0.0
    return math.log1p(x) - x / (1.0 + x)

def phi_nfw(r_kpc: float, M200: float, c: float, R200_kpc: float | None = None) -> float:
    """
    NFW potential with zero at infinity (untruncated), in (km/s)^2.
    Phi(r) = - G * M200 / r * ln(1 + r/rs) / f(c)
    where rs = R200 / c and f(c) = ln(1+c) - c/(1+c).
    If R200_kpc is not provided, it is computed from M200 via the critical-density definition
    with H0=70 km/s/Mpc and Delta=200. (Only needed to get rs from M200 and c.)

    Returns Phi(r) in (km/s)^2 (because G has those units).
    """
    if r_kpc <= 0:
        r_kpc = 1e-6
    if R200_kpc is None:
        # Compute R200 from M200 using: M200 = (4/3)*pi*200*rho_crit*R200^3
        # We'll use rho_crit = 3 H0^2 / (8 pi G_N). In these astrophysical units it's messy,
        # so instead use a practical approximate relation:
        #   R200 ≈ 206.0 * (M200/1e12 Msun)^(1/3) kpc   for H0 ~ 70 km/s/Mpc
        # (This is a common rule-of-thumb; good enough for our v_esc estimate.)
        R200_kpc = 206.0 * (M200 / 1e12) ** (1.0 / 3.0)

    rs = R200_kpc / c
    fc = f_nfw(c)
    if fc == 0:
        return 0.0
    return -G * M200 / r_kpc * math.log(1.0 + r_kpc / rs) / fc

def v_escape_nfw(r_kpc: float, M200: float, c: float, R200_kpc: float | None = None) -> float:
    """
    Escape speed at radius r for NFW with zero potential at infinity:
    v_esc = sqrt(-2 * Phi(r)).
    Returns km/s.
    """
    phi = phi_nfw(r_kpc, M200, c, R200_kpc)
    val = max(0.0, -2.0 * phi)
    return math.sqrt(val)

# Koppen+2018 formulae
def r_strip_from_def(deficiency: float, R_kpc: float, rmax_kpc: float) -> float:
    """
    Inverse of eq. (28):
    10^{-def} = [1 - (1 + (r/R)^2)^(-1/2)] / [1 - (1 + (rmax/R)^2)^(-1/2)]
    Solve analytically for r.
    """
    denom = 1.0 - (1.0 + (rmax_kpc / R_kpc) ** 2) ** -0.5
    X = 1.0 - (10.0 ** (-deficiency)) * denom
    # Safety clamp
    X = min(max(X, 1e-12), 0.999999999999)
    r_over_R = math.sqrt(X ** -2 - 1.0)
    return R_kpc * r_over_R

def sigma0_msun_per_kpc2(M0_msun: float, R_kpc: float, rmax_kpc: float) -> float:
    """
    Central surface density for Miyamoto–Nagai-like 2D profile used in the paper:
    Sigma_0 = M0 / [ 2π R^2 * (1 - (1 + (rmax/R)^2)^(-1/2)) ]   (with radii in kpc)
    Returns Msun/kpc^2.
    """
    denom = 2.0 * math.pi * (R_kpc ** 2) * (1.0 - (1.0 + (rmax_kpc / R_kpc) ** 2) ** -0.5)
    return M0_msun / denom

def sigma_at_r_msun_per_pc2(sigma0_msun_per_kpc2: float, r_kpc: float, R_kpc: float) -> float:
    """
    Sigma(r) = Sigma0 / (1 + (r/R)^2)^{3/2}.
    Convert Msun/kpc^2 -> Msun/pc^2 at the end (1 kpc^2 = 1e6 pc^2).
    """
    factor = (1.0 + (r_kpc / R_kpc) ** 2) ** 1.5
    sigma_r_msun_per_kpc2 = sigma0_msun_per_kpc2 / factor
    return sigma_r_msun_per_kpc2 / 1.0e6  # to Msun/pc^2

def p_def_cm3_kms2(
    M_HI_msun: float,
    deficiency: float,
    v_rot_kms: float,
    R_opt_kpc: float,
    rmax_over_R: float = 1.5,
    g_geom: float = 2.0,
    a_mol: float = 15.0,
) -> tuple[float, dict]:
    """
    Compute the 'required' ram pressure (from eq. 27), assuming deficiency-based r_strip.
    Returns p_def in cm^-3 (km/s)^2 and a dict of intermediate values for debugging.
    """
    R_kpc = R_opt_kpc
    rmax_kpc = rmax_over_R * R_kpc
    # Initial HI mass before stripping (M0) from deficiency:
    M0_msun = M_HI_msun * (10.0 ** deficiency)

    # Central surface density
    sigma0 = sigma0_msun_per_kpc2(M0_msun, R_kpc, rmax_kpc)

    # Stripping radius from deficiency (eq. 28 inverted)
    r_strip_kpc = r_strip_from_def(deficiency, R_kpc, rmax_kpc)

    # Sigma at r_strip (Msun/pc^2)
    sigma_r = sigma_at_r_msun_per_pc2(sigma0, r_strip_kpc, R_kpc)

    # Molecular enhancement term: R0 = 2 kpc * (R_opt / 15 kpc)
    R0_kpc = 2.0 * (R_kpc / 15.0)
    mol_boost = 1.0 + a_mol * math.exp(-r_strip_kpc / R0_kpc)

    # Final eq. (27) using the pre-derived conversion K_CONV, with explicit division by g
    # p_def [cm^-3 (km/s)^2] = (K_CONV/g) * (Sigma[Msun/pc^2]*v_rot^2[km^2/s^2]/r[kpc]) * mol_boost
    pdef = (K_CONV / g_geom) * (sigma_r * (v_rot_kms ** 2) / r_strip_kpc) * mol_boost

    debug = {
        "M0_msun": M0_msun,
        "Sigma0_Msun_kpc2": sigma0,
        "r_strip_kpc": r_strip_kpc,
        "Sigma_r_Msun_pc2": sigma_r,
        "R0_kpc": R0_kpc,
        "mol_boost": mol_boost,
    }
    return pdef, debug

def p_def_from_rstrip_cm3_kms2(
    M0_msun: float,
    v_rot_kms: float,
    R_opt_kpc: float,
    r_strip_kpc: float,
    rmax_over_R: float = 1.5,
    g_geom: float = 2.0,
    a_mol: float = 15.0,
) -> tuple[float, dict]:
    """
    Alternative path: if the user has a directly measured stripping radius r_strip.
    """
    R_kpc = R_opt_kpc
    rmax_kpc = rmax_over_R * R_kpc
    sigma0 = sigma0_msun_per_kpc2(M0_msun, R_kpc, rmax_kpc)
    sigma_r = sigma_at_r_msun_per_pc2(sigma0, r_strip_kpc, R_kpc)
    R0_kpc = 2.0 * (R_kpc / 15.0)
    mol_boost = 1.0 + a_mol * math.exp(-r_strip_kpc / R0_kpc)
    pdef = (K_CONV / g_geom) * (sigma_r * (v_rot_kms ** 2) / r_strip_kpc) * mol_boost
    debug = {
        "Sigma0_Msun_kpc2": sigma0,
        "Sigma_r_Msun_pc2": sigma_r,
        "R0_kpc": R0_kpc,
        "mol_boost": mol_boost,
    }
    return pdef, debug

def n_icm_beta_cm3(R_kpc: float, n0_cm3: float, Rc_kpc: float, beta: float) -> float:
    """ICM number density from a β-profile: n(R) = n0 * [1 + (R/Rc)^2]^(-3β/2)."""
    return n0_cm3 * (1.0 + (R_kpc / Rc_kpc) ** 2) ** (-1.5 * beta)

def p_loc_cm3_kms2(
    Rproj_kpc: float,
    los_offset_kpc: float,
    v_kms: float,
    n0_cm3: float,
    Rc_kpc: float,
    beta: float,
) -> tuple[float, float]:
    """
    Local ram pressure (upper limit per paper): p_loc = n_ICM(R_3D) * v^2
    using 3D radius R_3D = sqrt(Rproj^2 + z^2), with z = los_offset_kpc (user-controlled).
    Returns (p_loc, R_3D).
    """
    R3D_kpc = math.sqrt(max(0.0, Rproj_kpc**2 + los_offset_kpc**2))
    n = n_icm_beta_cm3(R3D_kpc, n0_cm3, Rc_kpc, beta)
    return n * (v_kms**2), R3D_kpc

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Ram Pressure Stripping Calculator", page_icon="🌀", layout="wide")
st.title("🌀 Is It Stripping ?")

st.markdown(
    """Uses the prescription of [Köppen+2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.4367K/abstract) to estimate 
whether a cluster galaxy is likely to be currently losing gas (or not yet lost much at all) or has done so in the past. This
works by estimating (1) the ram pressure needed to reach the current HI deficiency and (2) the probable current local pressure
a galaxy is actually experiencing, given a model of the cluster's mass profile and ICM density.""")
st.markdown("""A galaxy is deemed to likely be a current, active stripper if the **required pressure** is comparable or less
than the **local pressure**; i.e. it hasn't lost as much gas yet as the current pressure should be able to remove, therefore
it's still losing gas. Conversely, if the pressure required to reach the current deficiency is much greater than the current
estimated value, the galaxy is likely to be past stripper.""")
st.markdown("""Users can alter the parameters of a β–model for the ICM, and either set a fixed speed for the galaxy or use
an NFW-profile to use the locale escape velocity. By default, both use parameters based on the Virgo cluster.""")

with st.sidebar:
    st.header("🎰 Galaxy inputs")
    M_HI = st.number_input("[Optional] Current HI mass  [M$_{☉}$]", value=2.1e7, min_value=0.0, step=1.0E7, key="M_HI", format="%.3e", help='Not actually used in the ram pressure calculation - provided for convenience only, to show the original gas content and how much has been lost')
    deficiency = st.number_input("HI deficiency", value=0.7, min_value=0.0, max_value=2.0, step=0.05, format="%.2f", help='Logarithmic measure of the current compared to orginal gas content. 1.0 means the galaxy has 10% of its original content remaining, 2.0 means it has only 1%, etc.')
    Ropt = st.number_input("Optical radius R$_{opt}$ [kpc]", value=1.175, min_value=0.01, step=0.05, format="%.3f", help='HI detections are usually unresolved, so the optical radius can be used to estimate the original HI extent in these cases')
    vrot = st.number_input("Rotation speed v$_{rot}$ [km/s]", value=15.0, min_value=1.0, step=1.0, format="%.1f", help='Rotation speed after correcting the measurements for inclination and whatnot')
    rmax_over_R = st.slider("Initial HI extent compared to optical", min_value=1.2, max_value=3.0, value=1.5, step=0.1, help='Multiplies the optical radius to get the estimated initial HI extent, typically reckoned to be ~1.5-1.7x the optical')
    
    st.header("📐 Geometry")
    Rproj_mpc = st.number_input("Projected distance [Mpc]", value=1.03, min_value=0.01, step=0.05, format="%.2f", help='Current projected distance from the galaxy to the cluster centre')
    los_offset = st.number_input("Line-of-sight offset [Mpc]", value=0.0, min_value=0.0, step=0.25, format="%.2f", help='If known, here you can set the line-of-sight offset distance of the galaxy from the cluster centre, so giving the true 3D position for greater accuracy in the pressure calculation. Leave as zero to use only the projected distance')
    
    speed_mode = st.radio("Galaxy velocity", ["Fixed velocity", "Escape velocity (NFW)"], index=0, help='Choose whether to set a simple velocity for the galaxy or whether to calculate this given its current position, assuming an NFW model for the cluster potential. Both assume a radial orbit')
    if speed_mode == "Fixed velocity":
        vgal = st.number_input("Galaxy velocity [km/s]", value=1300.0, min_value=100.0, step=50.0, format="%.0f", help='Set the current speed of the galaxy at its current position in its radial orbit')
        M200 = None
        c_conc = None
        R200_kpc = None
    else:
        st.subheader("NFW halo parameters")
        M200 = st.number_input("M$_{200}$ [M$_{☉}$]", value=4.0e14, min_value=1e12, step=1e13, format="%.3e", help='Total virial mass of the halo')
        c_conc = st.number_input("Concentration", value=5.0, min_value=2.0, max_value=10.0, step=0.5, format="%.1f", help='Halo concentration, estimated as 10-15 for the Milky Way, typically reckoned as 3-6 for Virgo-mass clusters')
        # Optional explicit R200 if you prefer to override the scaling:
        R200_kpc = st.number_input("R$_{200}$ [kpc] (optional; 0 = auto)", value=0.0, min_value=0.0, step=10.0, format="%.1f", help='Virial radius of the cluster')
        if R200_kpc <= 0:
            R200_kpc = None
        vgal = None
    
    st.header("💫 Cluster model")
    n0 = st.number_input("n$_{0}$ [cm$^{-3]}$", value=0.04, min_value=1e-4, step=0.01, format="%.4f", help='Central ICM density')
    Rc = st.number_input("Core radius [kpc]", value=13.4, min_value=1.0, step=0.5, format="%.1f", help='Core radius of the ICM, used in the beta-profile model of the density')
    beta = st.number_input("β", value=0.47, min_value=0.1, max_value=1.5, step=0.01, format="%.2f", help='Beta factor used in the density profile; effectively this sets how steeply the density declines')

    st.header("💫 Other (optional)")
    g_geom = st.number_input("Molecular correction g", value=2.0, min_value=1.0, step=0.1, format="%.2f", help='Corrects for the molecular gas when calculating the vertical restoring force. The default factor 2 should be reasonable')
    a_mol = st.number_input("Molecular boost 'a'", value=15.0, min_value=0.0, step=1.0, format="%.1f", help='Molecular scale length. The default value of 15 gives a significant contribution in the inner regions for the restoring force, but negligible in the outskirts')


# ---------- Calculations ----------
# Required pressure from deficiency:
pdef, dbg = p_def_cm3_kms2(
    M_HI_msun=M_HI,
    deficiency=deficiency,
    v_rot_kms=vrot,
    R_opt_kpc=Ropt,
    rmax_over_R=rmax_over_R,
    g_geom=g_geom,
    a_mol=a_mol,
)

# Local pressure:
Rproj_kpc = Rproj_mpc * 1000.0
los_kpc = los_offset * 1000.0

if speed_mode == "Fixed velocity":
    p_loc, R3D = p_loc_cm3_kms2(
        Rproj_kpc=Rproj_kpc,
        los_offset_kpc=los_kpc,
        v_kms=vgal,
        n0_cm3=n0,
        Rc_kpc=Rc,
        beta=beta,
    )
else:
    # Use NFW escape speed at R3D
    R3D = math.sqrt(Rproj_kpc**2 + los_kpc**2)
    vesc = v_escape_nfw(R3D, M200, c_conc, R200_kpc=R200_kpc)
    p_loc, _ = p_loc_cm3_kms2(
        Rproj_kpc=Rproj_kpc,
        los_offset_kpc=los_kpc,
        v_kms=vesc,
        n0_cm3=n0,
        Rc_kpc=Rc,
        beta=beta,
    )

# ---------- Display ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Required pressure from H I deficiency (eq. 27)")
    st.metric("p_def  [cm⁻³ (km/s)²]", f"{pdef:,.3f}")
    st.caption(f"= {pdef/1000.0:.3f}  × 1000 cm⁻³ (km/s)²")

    with st.expander("Inputs & intermediate values"):
        st.write(f"R_opt = **{Ropt:.3f}** kpc,  r_max = **{rmax_over_R:.2f} × R** = **{rmax_over_R*Ropt:.3f}** kpc")
        st.write(f"M_HI = **{M_HI:.3e}** Msun,  def = **{deficiency:.3f}**,  v_rot = **{vrot:.1f}** km/s")
        st.write(f"g = **{g_geom:.2f}**,  a (molecular) = **{a_mol:.1f}**")
        st.write("---")
        st.write(f"M0 (pre-strip) = **{dbg['M0_msun']:.3e}** Msun")
        st.write(f"Sigma0 = **{dbg['Sigma0_Msun_kpc2']:.3e}** Msun/kpc²")
        st.write(f"r_strip = **{dbg['r_strip_kpc']:.3f}** kpc")
        st.write(f"Sigma(r_strip) = **{dbg['Sigma_r_Msun_pc2']:.4f}** Msun/pc²")
        st.write(f"R0 (mol scale) = **{dbg['R0_kpc']:.4f}** kpc → boost = **{dbg['mol_boost']:.4f}**")

with col2:
    st.subheader("Local ram pressure (β–model × speed)")
    if speed_mode == "Fixed velocity":
        st.write(f"Speed v = **{vgal:.0f}** km/s (user fixed)")
    else:
        st.write(f"Escape speed v_esc(NFW) = **{v_escape_nfw(R3D, M200, c_conc, R200_kpc):.0f}** km/s")
        st.write(f"NFW: M200 = **{M200:.3e}** Msun,  c = **{c_conc:.1f}**,  R200 = **{(R200_kpc if R200_kpc else 206.0*(M200/1e12)**(1/3)):.1f}** kpc")

    st.write(f"3D radius R = **{R3D:.0f}** kpc (from R_proj = {Rproj_kpc:.0f} kpc, z = {los_kpc:.0f} kpc)")
    st.metric("p_loc  [cm⁻³ (km/s)²]", f"{p_loc:,.3f}")
    st.caption(f"= {p_loc/1000.0:.3f}  × 1000 cm⁻³ (km/s)²")

# Classification
st.markdown("---")
ratio = pdef / p_loc if p_loc > 0 else math.inf
if ratio <= 3.0 and ratio >= 1/3:
    verdict = "Active (pressures comparable)"
elif ratio < 1/3:
    verdict = "Active (local >> required)"
else:
    verdict = "Past (required >> local)"

st.subheader("Classification")
st.write(f"Ratio p_def / p_loc = **{ratio:.3f}** → **{verdict}**")

st.info(
    "Notes: (i) p_loc uses the β–model density at *3D* radius and your chosen speed; "
    "(ii) using escape speed gives an upper-limit flavor (as in the paper); "
    "(iii) substructure (e.g., M49 region) can raise the true local density by factors of a few."
)

st.caption("Built from the paper’s equations: r_strip(def), Σ0 and Σ(r) for the HI disc, "
           "p_def (eq. 27) with g≈2 and molecular boost 1+15 e^{-r/R0}, and p_loc = n_ICM v^2 from a β–model.")
