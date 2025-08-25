# Streamlit GUI wrapper for HIPS2FITS image retrieval, based on the example HIP2FITS retrieval script provided online.
# Re-written by ChatGPT-5. Runs in Python 3.9+ (tested with Streamlit Cloud). Minimal error handling by design.

import io
import math
import numpy
from PIL import Image
import streamlit as st
import matplotlib
from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle, Longitude, Latitude
from astropy.wcs import WCS
from astropy.io import fits as pyfits
from astroquery.hips2fits import hips2fits
from astroquery.mocserver import MOCServer

# Matplotlib configuations to use a nicer font for the axes
matplotlib.rcParams['font.family'] = 'serif'                     # FALLBACK FAMILY
matplotlib.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']  # TRY TNR FIRST
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'          # MATH TEXT LOOKS OK WITH SERIF


# Streamlit style preferences
# Remove the menu button
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Remove vertical whitespace padding
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.write('<style>div.block-container{padding-bottom:0rem;}</style>', unsafe_allow_html=True)


# Check if the coordinates are already present in memory, and if not, set them to some defaults (VCC 2000)
# Setting them as a session_state parameter means we can easily update the GUI values from the name resolver
if 'coord_ra_val' not in st.session_state:
    st.session_state.coord_ra_val = "191.1332558422000"
if 'coord_dec_val' not in st.session_state:
    st.session_state.coord_dec_val = "11:11:25.74"


# Check for a previously retrieved image in the session state. This is used to prevent updates to the GUI from removing the current
# image from the display. No need to do anyything if the parameter already exists.
if "last_preview_png" not in st.session_state:
    st.session_state["last_preview_png"] = False   # If it hasn't been set, then no preview image has been created yet
    # Similarly set other parameters, one to hold the image and one to hold its caption
    if "preview_png_bytes" not in st.session_state:
        st.session_state["preview_png_bytes"] = None
    if "preview_caption" not in st.session_state:
        st.session_state["preview_caption"] = ""


preview_slot = st.empty()    # Used for (re)drawing the preview image every new run

st.set_page_config(page_title="HIPS2FITS Viewer", layout="wide")

st.title("HIPS2FITS Don't Lie")
st.write('### Retrieve and preview astronomical survey data using the HIP2FITS service')
st.write('Shows images from a variety of astronomical data sets at a specified field of view and resolution. Uses the astroquery HIPS2FITS service. Yes, you can do this on the HIP2FITS website directly, but this one lets you use different units and has drop-down menus and is just generally friendlier.') 
st.write("Note that the preview image does not show or update until you press 'Retrieve image'. Be careful in setting both the pixel scale and field of view, as large images will take longer to retrieve.")
st.write("If you need to download a FITS image for photometry, note that HIPS2FITS doesn't conserve the flux when rescaling - be sure to use the original pixel scale of the survey.")

# 1) Functions called by the GUI modules below
# Convert RA to degrees, allowing both decimal and sexigesimal input. Astropy has in-built modules to handle this !
def parse_ra(ra_text):
    # Assume decimal degrees by default
    try:   
        return Longitude(float(ra_text) * u.deg)
    # Otherwise assume sexigesimal formatting
    except Exception:
        try:
            # Sexagesimal path (assume hourangle)
            ang = Angle(ra_text, unit=u.hourangle)
            return Longitude(ang.to(u.deg))
        # If this doesn't work, it must be in a format astropy can't handle
        except:
            return 'Invalid RA'

# As above but for declination
def parse_dec(dec_text):
    try:
        return Latitude(float(dec_text) * u.deg)
    except Exception:
        try:
            ang = Angle(dec_text, unit=u.deg)
            return Latitude(ang)
        except:
            return 'Invalid Dec'

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

# Gamma adjustments
def apply_gamma_01(arr01, gamma):
    # arr01 in [0,1] float -> apply 1/gamma curve; returns float in [0,1]
    g = float(gamma)
    if g <= 0:
        g = 1.0
        
    return numpy.clip(arr01, 0.0, 1.0) ** (1.0 / g)

def apply_gamma_rgb_uint8(rgb_uint8, gamma):
    # RGB uint8 [0..255] -> gamma corrected uint8
    arr01 = rgb_uint8.astype(numpy.float32) / 255.0
    out01 = apply_gamma_01(arr01, gamma)
    
    return (out01 * 255.0).round().astype(numpy.uint8)

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
    # Save the image data to the permanent array
    st.session_state["preview_png_bytes"] = buf.getvalue()
    
    return buf

# Show the image either as a plain image (st.image) or with WCS axes (matplotlib WCSAxes).
def render_with_optional_wcs_axes(img_array, wcs_obj, show_axes, caption):
    # If the user doesn't want to show the axes :
    if not show_axes:
        st.image(img_array, caption=caption, use_container_width=True)    # The preview image itself
        st.session_state["last_preview_png"] = 'image'                    # Sets that a preview image has now been shown
        buf = io.BytesIO()
        Image.fromarray(img_array).save(buf, format="PNG")
        buf.seek(0)

        # Save the image and its caption to session_state parameters
        st.session_state["preview_png_bytes"] = buf.getvalue()
        st.session_state["preview_caption"] = caption

        return None

    # More complex case where user does want the axes
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection=wcs_obj)
    ax.tick_params(axis='both', which='major', direction='out', length=7,  width=1.4)
    ax.tick_params(axis='both', which='minor', direction='out', length=5,  width=1.1)

    # Slightly easier way of dealing with the axes, by name instead of number
    ra  = ax.coords[0]
    dec = ax.coords[1]

    ra.set_axislabel('Right Ascension [J2000]',  fontsize=15, minpad=0.8)
    dec.set_axislabel('Declination [J2000]', fontsize=15, minpad=0.8)

    # Tick sizes and frequencies
    ra.set_ticklabel(size=10)
    dec.set_ticklabel(size=10)
    ra.set_minor_frequency(5)
    dec.set_minor_frequency(5)

    # Sexigesimal tick formatting
    ra.set_major_formatter('hh:mm:ss')
    dec.set_major_formatter('dd:mm:ss')

    ax.imshow(img_array, origin="lower")
    #ax.set_xlabel("Right Ascension [J2000]")
    #ax.set_ylabel("Declination [J2000]")
    st.pyplot(fig, clear_figure=True)                # The preview image itself
    st.session_state["last_preview_png"] = 'matplot'      # Sets that a preview image has now been shown

    # Save the image and its caption to session_state parameters
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.session_state["preview_png_bytes"] = buf.getvalue()
    st.session_state["preview_caption"] = caption
    
    return fig

# Name resolver
def update_coords():
    try:
        coords = SkyCoord.from_name(name_tag)
        #st.write(coords.ra.to_string(unit=u.hour, sep=':'), coords.dec.to_string(unit=u.deg, sep=':'))
        st.session_state.coord_ra_val  = coords.ra.to_string(unit=u.hour, sep=':')
        st.session_state.coord_dec_val = coords.dec.to_string(unit=u.deg, sep=':')
        # Force GUI update
        #st.rerun()
    except:
        pass

# Retrieve the native data resolution
def hips_scale_arcsec_per_pix(hips_id):
    # Ask the MOCServer for this HiPS record and return hips_pixel_scale
    tbl = MOCServer.query_hips(criteria="ID=%s" % hips_id,
                               fields=["ID", "hips_pixel_scale"])
    if len(tbl) == 0 or "hips_pixel_scale" not in tbl.colnames:
        return None
    scale_deg = float(tbl["hips_pixel_scale"][0])           # degrees / pixel
    
    return (scale_deg * u.deg).to(u.arcsec).value           # arcsec / pixel


# 2) Set up the GUI
# Row 1: Input coordinates and FOV
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])    # ChatGPT preferred 1.2, 1.2 for the first, but this is asymmetrical and weird
# Coordinate defaults are set to the session_state values
with c1:
    coord_ra = st.text_input("RA  —  decimal degrees or HH:MM:SS", key='coord_ra_val', help="Must be J2000, but fairly liberal. Enter something sensible and astropy will try its best")
with c2:
    coord_dec = st.text_input("Dec  —  decimal degrees or DD:MM:SS", key='coord_dec_val', help="Must be J2000, but fairly liberal. Enter something sensible and astropy will try its best")
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
    name_tag = st.text_input("Name resolver / download file name", "Target", help="Enter the name of an object and press the adjacent 'Resolve coordinates' button to try and automatically update the RA and Dec. Also sets the prefix in the file name when downloading images")
with c8:
    # Add an empty line to ensure vertical alignment of the button with the text boxes. We could also do st.write(""), but this doesn't
    # add exactly the right amount of space
    st.markdown("<br>", unsafe_allow_html=True) 
    resolve = st.button("Resolve coordinates", help='Attempts to find the coordinates of the named object (equatorial only, J2000)', use_container_width=True, on_click=update_coords)



# Row 3: Survey and band selection
# Keep to DESI, SDSS, GALEX as requested — offer colour composites or individual bands.
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
    "DSS2": {
        "color": "CDS/P/DSS2/color",
        "bands": {
            "NIR": "CDS/P/DSS2/NIR",
            "Blue": "CDS/P/DSS2/blue",
            "Red": "CDS/P/DSS2/red",
        },
    }, 
    "2MASS": {
        "color": "CDS/P/2MASS/color",
        "bands": {
            "H": "CDS/P/2MASS/H",
            "J": "CDS/P/2MASS/J",
            "K": "CDS/P/2MASS/K",
        },
    },  
    "HI4PI": {
        "color": "CDS/P/HI4PI/NHI",
        "bands": {
            "NHI": "CDS/P/HI4PI/NHI",
        },
    },   
    "PanSTARS DR1": {
        "color": "CDS/P/PanSTARRS/DR1/color-i-r-g",
        "bands": {
            "g": "CDS/P/PanSTARRS/DR1/g",
            "i": "CDS/P/PanSTARRS/DR1/i",
            "r": "CDS/P/PanSTARRS/DR1/r",
            "y": "CDS/P/PanSTARRS/DR1/y",
            "z": "CDS/P/PanSTARRS/DR1/z",
        },
    }, 
}

with survey_col:
    survey_name = st.selectbox("Survey", list(SURVEYS.keys()), index=0, help="Specify which survey to use. Not necessarily a full list of every survey supported by HIP2FITS !")

with band_col:
    mode = st.selectbox("Mode", ["Colour composite", "Single band"], index=0, help="Choose whether to have a multi-colour image (e.g. RGB, where available) or a single waveband - the latter can be in FITS format")
    band_choice = None
    if mode == "Single band":
        band_choice = st.selectbox("Band", list(SURVEYS[survey_name]["bands"].keys()), help="If a single waveband was requested, here you can specify it. Survey-dependent.")

with format_col:
    # For color we’ll fetch JPG/PNG; for single band we default to FITS for science download.
    out_format = st.selectbox("Download format", ["PNG", "FITS"], index=0 if mode == "Colour composite" else 1, help="Format for the downloaded image. A preview image will be shown below, regardless of the format selected. FITS downloads only available for single-band images")


# Row 4 - print the selected data resolution
# Easiest to use a for loop here rather than a dictionary, this way it's band-independent. Also sets a fallback value in case
# the HIPS2FITS cannot retrieve the pixel scale
if 'DESI' in survey_name:
    hips_id = 'CDS/P/DESI-Legacy-Surveys/DR10/g'
    fallback = 0.262
if 'SDSS DR9' in survey_name:
    hips_id = 'CDS/P/SDSS9/g'
    fallback = 0.396
if 'GALEX' in survey_name:
    hips_id = 'CDS/P/GALEXGR6/AIS/NUV'
    fallback = 1.5
if '2MASS' in survey_name:
    hips_id = 'CDS/P/2MASS/J'
    fallback = 1.0
if 'PanSTARS' in survey_name:
    hips_id = 'CDS/P/PanSTARRS/DR1/g'
    fallback = 0.257
if 'HI4PI' in survey_name:
    hips_id = 'CDS/P/HI4PI/P_HI4PI_NHI'
    fallback = 972.0
if 'DSS2' in survey_name:
    hips_id = 'CDS/P/DSS2/blue'
    fallback = 1.0    # ?


# Get the pixelscale using HIPSFITS. Returns None if not found, in which case use the default
used_default = False
pixelscale = hips_scale_arcsec_per_pix(hips_id)
if pixelscale is None:
    pixescale = fallback
    used_default = True

# Print the resolution and its source to the screen
if 'HI4PI' not in survey_name:
    if used_default == False:
        st.write('Native pixel scale :', pixelscale, 'arcsec/pixel according to the data')
    if used_default == True:
        st.write('Native pixel scale :', pixelscale, 'arcsec/pixel (default value)')
if 'HI4PI' in survey_name:
    if used_default == False:
        st.write('Native resolution :', pixelscale, 'arcsec/pixel according to the data')
    if used_default == True:
        st.write('Native resolution :', pixelscale, 'arcsec/pixel (default value)')
    
    
    
st.write(survey_name)
#
#st.write("Pan-STARRS g:", hips_scale_arcsec_per_pix("CDS/P/PanSTARRS/DR1/g"), "arcsec/pixel")
#st.write("HI4PI NHI:", hips_scale_arcsec_per_pix("CDS/P/HI4PI/P_HI4PI_NHI"), "arcsec/pixel")


# Row 5 : Window dressing
gamma_col, axes_col, grid_col = st.columns([1, 1, 1])
with gamma_col:
    gamma = st.slider("Preview gamma",min_value=0.2, max_value=3.0, value=1.0, step=0.05, help="Gamma correction for the PNG images (does not affect FITS downloads). 1.0 is the survey default. Lower values darkern, higher values brighten")

with axes_col:
    st.markdown("<br>", unsafe_allow_html=True)     # Extra space for vertical alignment
    show_axes = st.checkbox("Show WCS axes", value=False, help="Shows WCS axes in the preview image. As with other buttons, the image needs to be retrieved again to update the display")

with grid_col:
    st.markdown("<br>", unsafe_allow_html=True)     # Extra space for vertical alignment
    show_grid = st.checkbox("Show grid", value=False, help="Overlays a subtle grid on the image (only if the axes are also shown)")
    

# Row 6: Comment
st.write('After the image is retrieved, scroll to the bottom for download options. Note that the download buttons do not save the images - use the right-click download option instead if you need to preserve the axes.')

# Row 7 : Run the script !
final_col, _, _, _ = st.columns([1, 1, 1, 1])
with final_col:
    fetch = st.button("Retrieve image")

st.markdown("---")


# Check if the coordinates are safe to proceed
safetoproceed = False
if parse_ra(coord_ra) != 'Invalid RA' and parse_dec(coord_dec) != 'Invalid Dec':
    safetoproceed = True

if fetch == True and safetoproceed == False:
    st.write('Invalid coordinate(s), please check RA and Dec values !')

# Main button : fetch the image !
if fetch == True and safetoproceed == True:
    st.write('Attempting image retrieval... _if I HIPS, I FITS..._')
    # Parse coordinates
    ra = parse_ra(coord_ra)
    dec = parse_dec(coord_dec)

    # Derive sizes
    fov_deg = fov_to_deg(fov_value, fov_unit)
    pix_deg = pixscale_to_deg(pix_value, pix_unit)
    width = max(1, int(round(fov_deg / pix_deg)))
    height = width  # square output per the guide

    # Select HIPS identifier
    if mode == "Colour composite":
        hips_id = SURVEYS[survey_name]["color"]
        request_format = "jpg"
    else:
        hips_id = SURVEYS[survey_name]["bands"][band_choice]
        # For single-band, request FITS to enable science download and precise WCS from header if desired.
        # We'll preview from FITS data directly.
        request_format = "fits" if out_format == "FITS" else "fits"  # always retrieve FITS for single-band

    # Query HIPS2FITS
    # Note: Following user's base script, we request TAN projection and full-range cut, and flip the color 
    # image vertically for correct display orientation.
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

    if mode == "Colour composite":
        # Wesult is an RGB image array; flip vertically for non-matplotlib (no axes) display
        colour_img = numpy.flip(result, axis=0)
        colour_img_gamma = apply_gamma_rgb_uint8(result, gamma)
        # Also need a flipped version for matplotlib
        colour_img_gamma_axes = numpy.flip(colour_img_gamma, axis=0)
        
        caption = f"{survey_name}  —  color  —  {width} × {height} px  —  FOV {fov_value} {fov_unit}"
        # Save the caption to the permament array
        st.session_state["preview_caption"] = caption
        # Update the image preview flag
        st.session_state["last_preview_png"] = 'matplot' if show_axes else 'image'
        # DON'T DRAW THE IMAGE HERE - only at the end when we know if needs to be redrawn or not

        ## Downloads
        #png_buf = to_png_bytes_from_array(colour_img)
        #st.download_button(
        #    label="Download PNG",
        #    data=png_buf,
        #    file_name=f"{name_tag}_{survey_name.replace(' ', '')}_color.png",
        #    mime="image/png",
        #)

        if show_axes:
            fig = plt.figure(figsize=(7, 7))
            ax = plt.subplot(111, projection=wcs_for_axes)

            # Optionally, show a subtle grid
            if show_grid == True:
                ax.grid(color='0.5', alpha=0.5, linewidth=0.6, linestyle=':')
            
            ax.tick_params(axis='both', which='major', direction='out', length=7,  width=1.4)

            # Slightly easier way of dealing with the axes, by name instead of number
            ra  = ax.coords[0]
            dec = ax.coords[1]

            ra.set_axislabel('Right Ascension [J2000]',  fontsize=15, minpad=0.8)
            dec.set_axislabel('Declination [J2000]', fontsize=15, minpad=0.8)

            # Tick sizes and frequencies
            ra.set_ticklabel(size=10)
            dec.set_ticklabel(size=10)
            ra.set_minor_frequency(5)
            dec.set_minor_frequency(5)

            # Sexigesimal formatting
            ra.set_major_formatter('hh:mm:ss')
            dec.set_major_formatter('dd:mm:ss')

            ax.imshow(colour_img_gamma_axes, origin="lower")
            
            #ax.set_xlabel("RA"); ax.set_ylabel("Dec")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            png_bytes = buf.getvalue()
            st.session_state["preview_png_bytes"] = png_bytes
        else:
            png_buf = to_png_bytes_from_array(colour_img_gamma)     # also sets session_state["preview_png_bytes"]
            png_bytes = png_buf.getvalue()

        # DOWNLOAD BUTTON (USE THE SAME BYTES WE JUST SAVED)
        st.download_button(
            label="Download PNG",
            data=png_bytes,
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
        stretched_gamma = apply_gamma_01(stretched, gamma)

        # Use grayscale preview with optional WCS axes
        fig = None
        # If the user wants to show the axes :
        if show_axes:
            fig = plt.figure(figsize=(7, 7))
            # Prefer WCS from header if present and plausible; otherwise use constructed WCS
            try:
                wcs_hdr = WCS(header) if header is not None else wcs_for_axes
            except Exception:
                wcs_hdr = wcs_for_axes
            ax = plt.subplot(111, projection=wcs_hdr)
            im = ax.imshow(stretched, origin="lower", cmap="gray", norm=PowerNorm(gamma=gamma))
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
            # DON'T DRAW THE IMAGE HERE, only at the end !!!
            st.session_state["last_preview_png"] = 'matplot'  # Sets that an image has now been shown
        # Otherwise, don't show the axes
        else:
            # As above, show the image and then update the preview parameter
            # DON'T DRAW THE IMAGE HERE
            #st.image(stretched, caption=f"{survey_name}  —  {band_choice}  —  FITS preview", use_container_width=True, clamp=True)
            st.session_state["last_preview_png"] = 'image'

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
        png8 = (stretched_gamma * 255.0).astype(numpy.uint8)
        png_buf = to_png_bytes_from_array(png8)
        st.download_button(
            label="Download PNG",
            data=png_buf,
            file_name=f"{name_tag}_{survey_name.replace(' ', '')}_{band_choice}.png",
            mime="image/png",
        )

# Draw the image preview only here at the end, rather than as the code goes along. This makes it easy to preseve the existing
# image, if there is one 
# If there's an image in the buffer, draw that one
if (st.session_state["preview_png_bytes"] is not None):
    st.image(st.session_state["preview_png_bytes"], caption=st.session_state["preview_caption"], use_container_width=True)

# DEPRECATED
# If there's no existing image, draw a new one
#if st.session_state["last_preview_png"] is None:
#    # Now there are four possibilities and different ways to draw the image. Firstly the two cases of colour composites :
#    if st.session_state["last_preview_png"] == 'image':
#        # 1,2) Colour composite with and without axes. Both cases handled by the render_ subroutine. All input parameters
#        # have been set above in "fetch".
#        if mode == "Colour composite":
#            render_with_optional_wcs_axes(colour_img, wcs_for_axes, show_axes, caption=caption)
#
#    # Next the cases of a greyscale preview of a FITS file.
#    # 3) Greyscale, no axes
#    if st.session_state["last_preview_png"] == 'matplot' and show_axes == False:
#        st.image(stretched, caption=f"{survey_name}  —  {band_choice}  —  FITS preview", use_container_width=True, clamp=True)
#    # 4) Greyscale, show axes
#    if st.session_state["last_preview_png"] == 'matplot' and show_axes == True:     
#        st.pyplot(fig, clear_figure=True)
