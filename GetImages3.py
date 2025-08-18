# Streamlit GUI wrapper for HIPS2FITS image retrieval, based on the example HIP2FITS retrieval script provided online.
# Re-written by ChatGPT-5. Runs in Python 3.9+ (tested with Streamlit Cloud). Minimal error handling by design.

import io
import math
import numpy
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle, Longitude, Latitude
from astropy.wcs import WCS
from astropy.io import fits as pyfits
from astroquery.hips2fits import hips2fits

# STYLE
# Remove the menu button
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Remove vertical whitespace padding
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.write('<style>div.block-container{padding-bottom:0rem;}</style>', unsafe_allow_html=True)


# Check for a previously retrieved image in the session state. This is used to prevent updates to the GUI from removing the current
# image from the display
if "preview_image" not in st.session_state:
    st.session_state["last_preview_png"] = None   # bytes
    #st.session_state["last_caption"] = ""
    #st.session_state["last_downloads"] = {}       # {"png": bytes, "fits": bytes}

preview_slot = st.empty()    # Used for (re)drawing the preview image every new run

st.set_page_config(page_title="HIPS2FITS Viewer", layout="wide")

st.title("HIPS2FITS Viewer")


# 1) Functions called by the GUI modules below
# Convert RA to degrees, allowing both decimal and sexigesimal input. Astropy has in-built modules to handle this !
def parse_ra(ra_text):
    # Assume decimal degrees by default
    try:   
        return Longitude(float(ra_text) * u.deg)
    # Otherwise assume sexigesimal formatting
    except Exception:
        # Sexagesimal path (assume hourangle)
        ang = Angle(ra_text, unit=u.hourangle)
        return Longitude(ang.to(u.deg))
    # *** ADD A FAILSAFE HERE !!! ***

# As above but for declination
def parse_dec(dec_text):
    try:
        return Latitude(float(dec_text) * u.deg)
    except Exception:
        ang = Angle(dec_text, unit=u.deg)
        return Latitude(ang)

# Convert input field of view to degrees - if in degrees, it doesn't do anything so no changes are made
def fov_to_deg(value, unit_label):
    if unit_label == "arcsec":
        return (value * u.arcsec).to(u.deg).value
    if unit_label == "arcmin":
        return (value * u.arcmin).to(u.deg).value
    return float(value)  # degrees

# As above but for the pixel scale
def pixscale_to_deg(value, unit_label):
    if unit_label == "arcsec / pixel":
        return (value * u.arcsec).to(u.deg).value
    return (value * u.arcmin).to(u.deg).value

# Construct a simple TAN WCS matching the requested center, size, and pixel grid.
def build_wcs(ra_deg, dec_deg, width, height, fov_deg):
    # Pixel scale in degrees/pixel along the widest dimension (square here)
    cdelt = fov_deg / float(width)
    w = WCS(naxis=2)
    # CRPIX is 1-indexed in FITS convention
    w.wcs.crpix = [ (width + 1) / 2.0, (height + 1) / 2.0 ]
    # RA decreases to the right in images -> negative CDELT1
    w.wcs.cdelt = numpy.array([-cdelt, cdelt])
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w

# Convert a normal array of bytes into a PNG image 
def to_png_bytes_from_array(img_array):
    # Expect uint8 RGB array; if float, normalize
    if img_array.dtype != numpy.uint8:
        arr = numpy.clip(img_array, 0, 255).astype(numpy.uint8)
    else:
        arr = img_array
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf

# Show the image either as a plain image (st.image) or with WCS axes (matplotlib WCSAxes).
def render_with_optional_wcs_axes(img_array, wcs_obj, show_axes, caption):
    if not show_axes:
        st.session_state["last_preview_png"] = st.image(img_array, caption=caption, use_column_width=True)
        st.image(img_array, caption=caption, use_column_width=True)
        return None

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection=wcs_obj)
    ax.imshow(img_array, origin="lower")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    st.pyplot(fig, clear_figure=True)
    return fig


# 2) Set up the GUI
# Row 1: Input coordinates and FOV
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])    # ChatGPT preferred 1.2, 1.2 for the first, but this is asymmetrical and weird
with c1:
    ra_text = st.text_input("RA  —  decimal degrees or HH:MM:SS", "191.1332558422000", help="Must be J2000, but fairly liberal. Enter something sensible and astropy will try its best")
with c2:
    dec_text = st.text_input("Dec  —  decimal degrees or DD:MM:SS", "11:11:25.74", help="Must be J2000, but fairly liberal. Enter something sensible and astropy will try its best")
with c3:
    fov_value = st.number_input("Field of view value", min_value=0.001, value=3.5, step=0.5, format="%.3f", help="Assumes a simple square field of view")
with c4:
    fov_unit = st.selectbox("FOV units", ["arcsec", "arcmin", "deg"], index=1)

# Row 2: Pixel scale and target name
c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
with c5:
    pix_value = st.number_input("Pixel scale value", min_value=0.001, value=1.0, step=0.1, format="%.3f", help="Pixel scale in the specified units. May break if too far outside the survey's native resolution")
with c6:
    pix_unit = st.selectbox("Pixel scale units", ["arcsec / pixel", "arcmin / pixel"], index=0)
with c7:
    show_axes = st.checkbox("Show WCS axes", value=False, help="Shows WCS axes in the preview image")
with c8:
    name_tag = st.text_input("Output basename", "Target", help="Optional, but useful for specifying the name of the file ahead of time for downloads")

# Row 3: Survey and band selection
# Keep to DESI, SDSS, GALEX as requested — offer color composites or individual bands.
survey_col, band_col, format_col = st.columns([1, 1, 1])    # ChatGPT preferred some weird asymmetrical four-column format for some reason !

SURVEYS = {
    "DESI Legacy Surveys DR10": {
        "color": "CDS/P/DESI-Legacy-Surveys/DR10/color",
        "bands": {
            "g": "CDS/P/DESI-Legacy-Surveys/DR10/g",
            "r": "CDS/P/DESI-Legacy-Surveys/DR10/r",
            "z": "CDS/P/DESI-Legacy-Surveys/DR10/z",
        },
    },
    "SDSS DR9": {
        "color": "CDS/P/SDSS9/color",
        "bands": {
            "u": "CDS/P/SDSS9/u",
            "g": "CDS/P/SDSS9/g",
            "r": "CDS/P/SDSS9/r",
            "i": "CDS/P/SDSS9/i",
            "z": "CDS/P/SDSS9/z",
        },
    },
    "GALEX GR6 AIS": {
        "color": "CDS/P/GALEXGR6/AIS/color",
        "bands": {
            "FUV": "CDS/P/GALEXGR6/AIS/FUV",
            "NUV": "CDS/P/GALEXGR6/AIS/NUV",
        },
    },
}

with survey_col:
    survey_name = st.selectbox("Survey", list(SURVEYS.keys()), index=0, help="Specify which survey to use. Not necessarily a full list of every survey supported by HIP2FITS !")

with band_col:
    mode = st.selectbox("Mode", ["Color composite", "Single band"], index=0, help="Choose whether to have a multi-colour image (e.g. RGB, where available) or a single waveband - the latter can be in FITS format")
    band_choice = None
    if mode == "Single band":
        band_choice = st.selectbox("Band", list(SURVEYS[survey_name]["bands"].keys()), help="If a single waveband was requested, here you can specify it. Survey-dependent.")

with format_col:
    # For color we’ll fetch JPG/PNG; for single band we default to FITS for science download.
    out_format = st.selectbox("Download format", ["PNG", "FITS"], index=0 if mode == "Color composite" else 1, help="Format for the downloaded image. A preview image will be shown below, regardless of the format selected")

# Row 4: Comment
st.write('After the image is retrieved, scroll to the bottom for download options.')

# Row 5 : Run the script !
final_col, _, _, _ = st.columns([1, 1, 1, 1])
with final_col:
    fetch = st.button("Retrieve image")

st.markdown("---")

# ---------------------------
# Action: fetch image
# ---------------------------
if fetch:
    # Button pressed, so now we clear the session state
    st.session_state["last_preview_png"] = None
    # Parse coordinates
    ra = parse_ra(ra_text)
    dec = parse_dec(dec_text)

    # Derive sizes
    fov_deg = fov_to_deg(fov_value, fov_unit)
    pix_deg = pixscale_to_deg(pix_value, pix_unit)
    width = max(1, int(round(fov_deg / pix_deg)))
    height = width  # square output per the guide

    # Select HIPS identifier
    if mode == "Color composite":
        hips_id = SURVEYS[survey_name]["color"]
        request_format = "jpg"
    else:
        hips_id = SURVEYS[survey_name]["bands"][band_choice]
        # For single-band, request FITS to enable science download and precise WCS from header if desired.
        # We'll preview from FITS data directly.
        request_format = "fits" if out_format == "FITS" else "fits"  # always retrieve FITS for single-band

    # Query HIPS2FITS
    # Note: Following user's base script, we request TAN projection and full-range cut,
    # and flip the color image vertically for correct display orientation.
    result = hips2fits.query(
        hips=hips_id,
        width=width,
        height=height,
        ra=ra,
        dec=dec,
        fov=Angle(fov_deg * u.deg),
        projection="TAN",
        get_query_payload=False,
        format=request_format,
        min_cut=0.0,
        max_cut=100.0,
    )

    # Prepare WCS (used for optional axes overlay; for FITS we could also read header WCS)
    wcs_for_axes = build_wcs(ra.to_value(u.deg), dec.to_value(u.deg), width, height, fov_deg)

    if mode == "Color composite":
        # result is an RGB image array; flip vertically for correct display as in the user's script
        img = numpy.flip(result, axis=0)
        caption = f"{survey_name}  —  color  —  {width} × {height} px  —  FOV {fov_value} {fov_unit}"
        render_with_optional_wcs_axes(img, wcs_for_axes, show_axes, caption=caption)

        # Downloads
        png_buf = to_png_bytes_from_array(img)
        st.download_button(
            label="Download PNG",
            data=png_buf,
            file_name=f"{name_tag}_{survey_name.replace(' ', '')}_color.png",
            mime="image/png",
        )

    else:
        # Single-band FITS: preview and download FITS; optional PNG export as convenience.
        # Convert FITS HDUList / PrimaryHDU to array
        if hasattr(result, "data"):
            data = result.data
            header = getattr(result, "header", None)
        else:
            # If hips2fits returns an HDUList
            data = result[0].data
            header = result[0].header

        # Simple display stretch: percentile clip to 1..99
        lo, hi = numpy.nanpercentile(data, [1.0, 99.0])
        stretched = numpy.clip((data - lo) / (hi - lo + 1e-12), 0.0, 1.0)

        # Use grayscale preview with optional WCS axes
        fig = None
        if show_axes:
            fig = plt.figure(figsize=(7, 7))
            # Prefer WCS from header if present and plausible; otherwise use constructed WCS
            try:
                wcs_hdr = WCS(header) if header is not None else wcs_for_axes
            except Exception:
                wcs_hdr = wcs_for_axes
            ax = plt.subplot(111, projection=wcs_hdr)
            im = ax.imshow(stretched, origin="lower", cmap="gray")
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
            st.pyplot(fig, clear_figure=True)
        else:
            st.session_state["last_preview_png"] = st.image(stretched, caption=f"{survey_name}  —  {band_choice}  —  FITS preview", use_column_width=True, clamp=True)
            st.image(stretched, caption=f"{survey_name}  —  {band_choice}  —  FITS preview", use_column_width=True, clamp=True)

        # Downloads
        # FITS file
        fits_buf = io.BytesIO()
        if header is None:
            hdu = pyfits.PrimaryHDU(data=data)
        else:
            hdu = pyfits.PrimaryHDU(data=data, header=header)
        hdul = pyfits.HDUList([hdu])
        hdul.writeto(fits_buf)
        fits_buf.seek(0)

        st.download_button(
            label="Download FITS",
            data=fits_buf,
            file_name=f"{name_tag}_{survey_name.replace(' ', '')}_{band_choice}.fits",
            mime="application/fits",
        )

        # Optional PNG export for quicklook
        # Convert stretched grayscale to 8-bit
        png8 = (stretched * 255.0).astype(numpy.uint8)
        png_buf = to_png_bytes_from_array(png8)
        st.download_button(
            label="Download PNG",
            data=png_buf,
            file_name=f"{name_tag}_{survey_name.replace(' ', '')}_{band_choice}.png",
            mime="image/png",
        )

# Footer hint (kept minimal per instructions)
#st.caption("HIPS2FITS powered — minimal UI, no heavy error trapping. Happy hunting for photons.")
