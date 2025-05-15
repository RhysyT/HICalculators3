# Code from ChatGPT to provide a GUI wrapper for HIPS2FITS

import streamlit as st
from io import BytesIO
from astropy.coordinates import SkyCoord	
from astropy import units as u
from astroquery.hips2fits import Hips2Fits
from astropy.io import fits
import numpy as np
from PIL import Image

st.set_page_config(page_title="HIPS2FITS Browser", layout="wide")
st.title("HIPS2FITS Image Retriever")

# Sidebar inputs
st.sidebar.header("Query Parameters")
mode = st.sidebar.radio("Input mode", ["Coordinates", "Upload FITS Region"])

if mode == "Coordinates":
	coord_str = st.sidebar.text_input(
		"Coordinates (RA Dec, sexagesimal or decimal)", "10h00m00s +02d00m00s")
	fov = st.sidebar.slider("Field of view (arcmin)", min_value=1.0, max_value=180.0, value=30.0)
	if coord_str:
		try:
			coord = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
		except Exception:
			coord = SkyCoord(*[float(x) for x in coord_str.split()], unit=(u.deg, u.deg))
		region_center = coord
else:
	fits_file = st.sidebar.file_uploader("Upload FITS file for region", type="fits")
	if fits_file:
		hdul = fits.open(fits_file)
		header = hdul[0].header
		wcs = WCS(header)
		region_center = None  # could extract from WCS if desired

# Dataset selection
datasets = {
	"SDSS RGB": "CDS/P/SDSS9/color",
	"SDSS Alternate": "CDS/P/SDSS9",
	"DSS2 Color": "DSS2color",
	"GALEX": "GALEX",
	"DESI Legacy Survey": "LS-DR9",
}
selection = st.sidebar.selectbox("Data set", list(datasets.keys()))
dataset_id = datasets[selection]

# Waveband options
wavebands = st.sidebar.multiselect(
	"Wavebands (leave empty for full-color)",
	["u", "g", "r", "i", "z"] if "SDSS" in selection else ["FUV", "NUV"] if selection == "GALEX" else []
)

# Display options
st.sidebar.header("Display Options")
show_axes = st.sidebar.checkbox("Show axes", value=True)
axis_format = st.sidebar.radio("Axis format", ["Sexagesimal", "Decimal"])

st.sidebar.header("Download Options")
if not wavebands:
	img_format = st.sidebar.selectbox("Image format", ["png", "jpg"])
else:
	img_format = "fits"

# Query button
if st.sidebar.button("Retrieve Image"):
	hips = Hips2Fits()
	try:
		if mode == "Coordinates":
			result = hips.get_image(
				hips_service=datasets[selection],
				hips_frame=f"{region_center.ra.deg},{region_center.dec.deg},{fov}"  # ra,dec,fov in degrees
			)
		else:
			result = hips.get_image(
				hips_service=datasets[selection],
				fits_file=fits_file
			)
		hdulist = fits.open(BytesIO(result))
		data = hdulist[0].data
		# Normalize and apply sliders
		vmin, vmax = st.sidebar.slider(
			"Intensity range", float(np.min(data)), float(np.max(data)),
			(float(np.min(data)), float(np.max(data)))
		)
		img_arr = np.clip((data - vmin) / (vmax - vmin), 0, 1)
		if img_format in ["png", "jpg"]:
			img = Image.fromarray((img_arr * 255).astype(np.uint8))
			st.image(img, use_column_width=True)
			buf = BytesIO()
			img.save(buf, format=img_format.upper())
			buf.seek(0)
			st.download_button(
				label=f"Download {img_format.upper()}",
				data=buf,
				file_name=f"hips_image.{img_format}",
				mime=f"image/{img_format}"
			)
		else:
			# FITS
			buf = BytesIO()
			hdulist.writeto(buf)
			buf.seek(0)
			st.download_button(
				label="Download FITS file",
				data=buf,
				file_name="hips_image.fits",
				mime="application/fits"
			)
	except Exception as e:
		st.error(f"Error retrieving image: {e}")
