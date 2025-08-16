# -*- coding: utf-8 -*-

# Code from GPT-5 for downloading astronomical images using HIPs2FITS

import io
import math
import tempfile
from collections import OrderedDict

import numpy
import streamlit as st

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Longitude, Latitude, Angle
from astropy.io import fits
from astropy.visualization import (ImageNormalize, MinMaxInterval, PercentileInterval,
                                   AsinhStretch, LinearStretch, LogStretch, SqrtStretch)
from astropy.wcs import WCS

from astroquery.hips2fits import hips2fits


# ----------------------------
# Utility: data set registry
# ----------------------------

# RGB HiPS datasets (rendered server-side; downloadable as JPG/PNG)
RGB_DATASETS = OrderedDict([
    # Label                             HiPS base (CDS/Aladin)
    ("SDSS (DR12, default colours)",    "http://alasky.u-strasbg.fr/SDSS/DR12/color"),
    ("SDSS (DR9, alternative colours)", "http://alasky.u-strasbg.fr/SDSS/DR9/color"),
    ("DSS2 coloured",                   "http://alasky.u-strasbg.fr/DSS/DSSColor"),
    ("GALEX GR6/7 coloured",            "http://alasky.u-strasbg.fr/GALEX/GR6-07/AllSkyColor"),
    ("DESI Legacy Survey (DR9) grz",    "http://alasky.u-strasbg.fr/DESI-Legacy-Survey/DR9/color"),
])

# Single-band HiPS datasets for FITS downloads + display with WCS axes
# Map: group -> list of (label, hips_url)
SINGLE_BAND = OrderedDict([
    ("SDSS (DR12)", [
        ("u", "http://alasky.u-strasbg.fr/SDSS/DR12/u"),
        ("g", "http://alasky.u-strasbg.fr/SDSS/DR12/g"),
        ("r", "http://alasky.u-strasbg.fr/SDSS/DR12/r"),
        ("i", "http://alasky.u-strasbg.fr/SDSS/DR12/i"),
        ("z", "http://alasky.u-strasbg.fr/SDSS/DR12/z"),
    ]),
    ("DSS2", [
        ("blue",  "http://alasky.u-strasbg.fr/DSS/DSS2-blue"),
        ("red",   "http://alasky.u-strasbg.fr/DSS/DSS2-red"),
        ("ir",    "http://alasky.u-strasbg.fr/DSS/DSS2-infrared"),
    ]),
    ("GALEX GR6/7", [
        ("FUV", "http://alasky.u-strasbg.fr/GALEX/GR6-07/FUV"),
        ("NUV", "http://alasky.u-strasbg.fr/GALEX/GR6-07/NUV"),
    ]),
    ("DESI Legacy Survey (DR9)", [
        ("g", "http://alasky.u-strasbg.fr/DESI-Legacy-Survey/DR9/g"),
        ("r", "http://alasky.u-strasbg.fr/DESI-Legacy-Survey/DR9/r"),
        ("z", "http://alasky.u-strasbg.fr/DESI-Legacy-Survey/DR9/z"),
    ]),
])

# ----------------------------
# Helpers
# ----------------------------

def parse_coords(coord_text):
    """
    Parse equatorial coordinates in sexagesimal or decimal, many formats allowed.
    Returns (ra_deg, dec_deg).
    """
    c = SkyCoord(coord_text, unit=(u.hourangle, u.deg), frame='icrs')
    return float(c.ra.deg), float(c.dec.deg)


def fov_from_pixels_wcs(w, naxis1, naxis2):
    """
    Estimate field of view in degrees from a FITS WCS + image size.
    """
    # sample four corners and compute max separation along RA/Dec
    rr = []
    cc = [(1, 1), (naxis1, 1), (1, naxis2), (naxis1, naxis2)]
    world = w.pixel_to_world_values(numpy.array([p[0] for p in cc]),
                                    numpy.array([p[1] for p in cc]))
    ras = numpy.array(world[0])
    decs = numpy.array(world[1])

    # approximate FOV as max of spans
    # handle RA wrap
    dr = numpy.max(numpy.mod(ras - numpy.min(ras) + 540.0, 360.0) - 180.0)
    dd = float(numpy.max(decs) - numpy.min(decs))
    fov = float(max(abs(dr), abs(dd)))
    # guard against weird headers
    if not numpy.isfinite(fov) or fov <= 0:
        fov = 0.1
    return fov


def center_from_wcs(w, naxis1, naxis2):
    cx = (naxis1 + 1.0) / 2.0
    cy = (naxis2 + 1.0) / 2.0
    ra, dec = w.pixel_to_world_values(cx, cy)
    return float(ra), float(dec)


def stretch_from_name(name):
    if name == "linear":
        return LinearStretch()
    if name == "log":
        return LogStretch()
    if name == "asinh":
        return AsinhStretch()
    if name == "sqrt":
        return SqrtStretch()
    return LinearStretch()


def build_axes(fig, wcs, tick_format):
    import matplotlib.pyplot as plt  # imported late so Streamlit can manage backend
    # Single WCS axes
    ax = plt.gca()
    ax = plt.subplot(111, projection=wcs)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    if tick_format == "sexigesimal":
        ax.coords[0].set_format_unit(u.hourangle)
        ax.coords[0].set_major_formatter('hh:mm:ss')
        ax.coords[1].set_major_formatter('dd:mm:ss')
    else:
        ax.coords[0].set_format_unit(u.deg)
        ax.coords[0].set_major_formatter('d.dddd')
        ax.coords[1].set_major_formatter('d.dddd')
    return ax


def rgba_to_png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="HIPS2FITS Browser", layout="wide")

st.title("HIPS2FITS browser  —  RGB & FITS downloader")

# Sections are stacked vertically; inside each we use rows/columns to keep things tight.

with st.container():
    st.subheader("1. Region selection")

    c1, c2 = st.columns([3, 2])

    with c1:
        coord_text = st.text_input(
            "Equatorial coordinates  —  sexigesimal or decimal",
            value="10:00:00 +02:00:00",
            help="Examples : 10:00:00 +02:00:00  —  150.0 +2.0  —  12h30m00s -45d00m00s"
        )
        fov_deg = st.number_input(
            "Field of view (degrees)",
            min_value=0.001, max_value=30.0, value=0.2, step=0.001,
            help="Used unless a FITS file is uploaded below"
        )

    with c2:
        uploaded = st.file_uploader(
            "Alternatively, upload a FITS file defining the region",
            type=["fits", "fz", "fit"], accept_multiple_files=False
        )
        use_uploaded_center = st.checkbox("Use uploaded FITS to set center + FoV", value=False)

    center_ra = None
    center_dec = None
    if use_uploaded_center and uploaded is not None:
        with fits.open(uploaded) as hdul:
            hdu = hdul[0]
            w = WCS(hdu.header)
            naxis1 = int(hdu.header.get("NAXIS1", hdu.data.shape[1] if hdu.data is not None else 0))
            naxis2 = int(hdu.header.get("NAXIS2", hdu.data.shape[0] if hdu.data is not None else 0))
            center_ra, center_dec = center_from_wcs(w, naxis1, naxis2)
            fov_deg = fov_from_pixels_wcs(w, naxis1, naxis2)
            st.info("Center and FoV taken from uploaded FITS : RA %.6f  Dec %.6f  FoV %.4f deg" % (center_ra, center_dec, fov_deg))

    if center_ra is None:
        try:
            center_ra, center_dec = parse_coords(coord_text)
        except Exception as e:
            st.error("Could not parse coordinates : %s" % str(e))
            st.stop()

ra_angle = Longitude(center_ra, u.deg)
dec_angle = Latitude(center_dec, u.deg)
fov_angle = Angle(fov_deg, u.deg)
rot_angle = Angle(0.0, u.deg)

with st.container():
    st.subheader("2. Data set & output")

    mode = st.radio("What to fetch  ?", ["RGB composite (JPG/PNG)", "Single waveband (FITS)"], horizontal=True)

    if mode == "RGB composite (JPG/PNG)":
        rgb_choice = st.selectbox("RGB dataset", list(RGB_DATASETS.keys()))
        rgb_url = RGB_DATASETS[rgb_choice]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            out_w = st.number_input("Width (pixels)", min_value=64, max_value=4096, value=800, step=16)
        with c2:
            out_h = st.number_input("Height (pixels)", min_value=64, max_value=4096, value=800, step=16)
        with c3:
            stretch_name = st.selectbox("Tone mapping", ["linear", "log", "asinh", "sqrt"], index=2)
        with c4:
            gamma = st.slider("Gamma", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

        # intensity controls for server-side render
        c1, c2 = st.columns(2)
        with c1:
            min_cut = st.text_input("Min cut (leave blank to auto)", value="")
        with c2:
            max_cut = st.text_input("Max cut (leave blank to auto)", value="")

        # Note on axes
        st.caption("Axes overlays are not available on JPG/PNG tiles returned by HIPS2FITS . For axes, use the FITS mode below .")

        # action row
        b1, b2, b3 = st.columns(3)
        fetch_preview = b1.button("Fetch preview")
        dl_jpg = b2.button("Download JPG")
        dl_png = b3.button("Download PNG")

        def query_rgb(return_format):
            kwargs = dict(
                hips=rgb_url,
                ra=ra_angle,
                dec=dec_angle,
                width=int(out_w),
                height=int(out_h),
                fov=fov_angle,
                projection="TAN",
                format=return_format,
                stretch=stretch_name,
            )
            if min_cut.strip() != "":
                kwargs["min_cut"] = float(min_cut)
            if max_cut.strip() != "":
                kwargs["max_cut"] = float(max_cut)
            return hips2fits.query(**kwargs)

        if fetch_preview:
            try:
                img = query_rgb("jpg")
                st.image(img, caption=rgb_choice, use_column_width=True)
            except Exception as e:
                st.error("HIPS2FITS error : %s" % str(e))

        if dl_jpg:
            try:
                payload = query_rgb("jpg")
                st.download_button("Save JPG", data=payload, file_name="hips_%s.jpg" % rgb_choice.replace(" ", "_"))
            except Exception as e:
                st.error("Download failed : %s" % str(e))

        if dl_png:
            try:
                payload = query_rgb("png")
                st.download_button("Save PNG", data=payload, file_name="hips_%s.png" % rgb_choice.replace(" ", "_"))
            except Exception as e:
                st.error("Download failed : %s" % str(e))

    else:
        band_group = st.selectbox("Survey", list(SINGLE_BAND.keys()))
        band_label, band_url = st.selectbox(
            "Waveband",
            SINGLE_BAND[band_group],
            format_func=lambda pair: pair[0]
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            out_w = st.number_input("Width (pixels)", min_value=64, max_value=8192, value=800, step=16)
        with c2:
            out_h = st.number_input("Height (pixels)", min_value=64, max_value=8192, value=800, step=16)
        with c3:
            show_axes = st.checkbox("Plot with axes", value=True)
        with c4:
            tick_fmt = st.selectbox("Axes format", ["sexigesimal", "decimal"], index=0)

        # local display contrast controls (affects on-screen plot only)
        c1, c2, c3 = st.columns(3)
        with c1:
            p_lo = st.slider("Percentile low", min_value=0.0, max_value=25.0, value=1.0, step=0.1)
        with c2:
            p_hi = st.slider("Percentile high", min_value=75.0, max_value=100.0, value=99.5, step=0.1)
        with c3:
            stretch_name = st.selectbox("Stretch", ["linear", "log", "asinh", "sqrt"], index=2)

        # action row
        b1, b2 = st.columns(2)
        fetch_preview = b1.button("Fetch preview (FITS)")
        dl_fits = b2.button("Download FITS")

        def query_fits():
            return hips2fits.query(
                hips=band_url,
                ra=ra_angle,
                dec=dec_angle,
                width=int(out_w),
                height=int(out_h),
                fov=fov_angle,
                projection="TAN",
                format="fits"
            )

        if fetch_preview:
            try:
                payload = query_fits()
                with fits.open(io.BytesIO(payload)) as hdul:
                    hdu = hdul[0]
                    data = hdu.data
                    wcs = WCS(hdu.header)

                    # normalize for display
                    interval = PercentileInterval(p_hi, lower_percentile=p_lo)
                    stretch = stretch_from_name(stretch_name)
                    norm = ImageNormalize(vmin=None, vmax=None, stretch=stretch, interval=interval)

                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(6, 6), dpi=150)
                    if show_axes:
                        build_axes(fig, wcs, tick_fmt)
                        plt.imshow(data, origin='lower', norm=norm, cmap='gray')
                    else:
                        plt.imshow(data, origin='lower', norm=norm, cmap='gray')
                        plt.axis('off')

                    st.pyplot(fig, clear_figure=True, use_container_width=True)

            except Exception as e:
                st.error("HIPS2FITS or FITS display error : %s" % str(e))

        if dl_fits:
            try:
                payload = query_fits()
                st.download_button(
                    "Save FITS",
                    data=payload,
                    file_name="hips_%s_%s.fits" % (band_group.replace(" ", "_"), band_label)
                )
            except Exception as e:
                st.error("Download failed : %s" % str(e))

st.markdown("---")
st.caption("Tip : RGB images are rendered by the HIPS2FITS service itself (you can tweak stretch / gamma / cuts before download). "
           "Single-band FITS are downloaded and displayed locally here with your chosen stretch . And they all lived happily ever after .")
