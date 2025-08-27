# Calculate distance, speed or travel time in different, friendly units

import streamlit as st
import math
from math import pi as pi
import importlib as imp

# EXTERNAL SCRIPTS IMPORTED AS FUNCTIONS
# "nicenumber" function returns human-readable versions of numbers, e.g, comma-separated or scientific notation depending
# on size
import NiceNumber
imp.reload(NiceNumber)
from NiceNumber import nicenumber


# STYLE
# Remove the menu button
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Remove vertical whitespace padding
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.write('<style>div.block-container{padding-bottom:0rem;}</style>', unsafe_allow_html=True)


# MAIN CODE
# Unit conversions
pc = 3.0856775812799588E16 # 1 pc in m
yr = 365.0*24.0*3600.0 # 1 yr in seconds
myr = 365.0*24.0*3600.0*1000000.0 # 1 Myr in seconds


# Three different calculations : time, distance, and speed. Since each use the same parameters, but the whole script is re-run each time
# a value is updated, give them different variable names to avoid conflicts. E.g. dist_ft = distance for time, fd = for distance, fs = 
# for speed


st.write("# Compute travel time")
st.write('Calculates time, speed, or distance travelled, depending on input parameters, in astronomy-friendly units.')


if st.checkbox('Calculate travel time'):

	left_column, right_column = st.columns(2)

	with left_column:
		# Distance, row 1
		dist_ft = st.number_input("Distance", format="%.3f", key="distfortime")
		
		# Speed, row 2
		speed_ft = st.number_input("Speed", format="%.3f", key="speedfortime")
		
	with right_column:
		# Distance unit widget, row 1 
		distunit_ft = st.selectbox('Distance unit', ('m', 'pc', 'kpc', 'Mpc', 'm'), index=2, key="munitfortime")

		# Speed unit widget, row 2 
		speedunit_ft = st.selectbox('Speed unit', ('km/s', 'pc/yr', 'kpc/Myr'), key="sunitfortime")
				
	# Get everything into SI units
	if distunit_ft == 'm':
		distance_ft = dist_ft

	if distunit_ft == 'pc':
		distance_ft = dist_ft*pc
		
	if distunit_ft == 'kpc':
		distance_ft = dist_ft*1000.0*pc	
		
	if distunit_ft == 'Mpc':
		distance_ft = dist_ft*1000000.0*pc

	if speed_ft > 0.0:		
		if speedunit_ft == 'km/s':
			velocity_ft = speed_ft*1000.0
		
		if speedunit_ft == 'pc/yr':
			velocity_ft = speed_ft*pc / yr
		
		if speedunit_ft == 'kpc/Myr':
			velocity_ft = speed_ft*1000.0*pc / myr
		
		sitime_ft = distance_ft/ velocity_ft
	
		myrtime_ft = sitime_ft / myr
	
	
		st.write("#### Time taken = ", nicenumber(myrtime_ft),' in Myr, or ', nicenumber(sitime_ft),' in seconds.') 
		st.write('Exact values are '+str(myrtime_ft)+' in Myr and '+str(sitime_ft)+' in seconds.')


if st.checkbox('Calculate distance travelled'):

	left_column, right_column = st.columns(2)
	
	with left_column:
		# Speed, row 1
		speed_fd = st.number_input("Speed", format="%.3f", key="speedfordist")
		
		# Time, row 2
		time_fd = st.number_input("Time", format="%.3f", key="timefordist")

	with right_column:
		# Speed unit widget, row 1 
		speedunit_fd = st.selectbox('Speed unit', ('km/s', 'pc/yr', 'kpc/Myr'), key="sunitfordist")
		
		# Time unit widget, row 2 
		timeunit_fd = st.selectbox('Time unit', ('Seconds', 'Years', 'Myrs', 'Gyrs'), index=2, key="tunitfordist")

	# Get into SI units
	if speedunit_fd == 'km/s':
		velocity_fd = speed_fd*1000.0
		
	if speedunit_fd == 'pc/yr':
		velocity_fd = speed_fd*pc / yr
		
	if speedunit_fd == 'kpc/Myr':
		velocity_fd = speed_fd*1000.0*pc / myr
		
	if timeunit_fd == 'Seconds':
		sitime_fd = time_fd
	
	if timeunit_fd == 'Years':
		sitime_fd = time_fd * yr
		
	if timeunit_fd == 'Myrs':
		sitime_fd = time_fd * myr	
		
	if timeunit_fd == 'Gyrs':
		sitime_fd = time_fd * (1000.0*myr)
			
	sidist_fd = velocity_fd * sitime_fd	
	stnddist_fd = sidist_fd / (1000.0*pc)
			
	st.write("#### Distance travelled = ", nicenumber(stnddist_fd),' in kpc, or ', nicenumber(sidist_fd),' in metres.') 
	st.write('Exact values are '+str(stnddist_fd)+' in kpc and '+str(sidist_fd)+' in metres.')


if st.checkbox('Calculate average speed'):
	
	left_column, right_column = st.columns(2)
	
	with left_column:
		# Distance, row 1
		dist_fs = st.number_input("Distance", format="%.3f", key="distforspeed")
		
		# Time, row 2
		time_fs = st.number_input("Time", format="%.3f", key="timeforspeed")
				
	with right_column:
		# Distance unit widget, row 1 
		distunit_fs = st.selectbox('Distance unit', ('m', 'pc', 'kpc', 'Mpc'), index=2, key="munitforspeed")

		# Time unit widget, row 2 
		timeunit_fs = st.selectbox('Time unit', ('Seconds', 'Years', 'Myrs', 'Gyrs'), index=2, key="tunitforspeed")

	# Get into SI units
	if distunit_fs == 'm':
		distance_fs = dist_fs	
	
	if distunit_fs == 'pc':
		distance_fs = dist_fs*pc
		
	if distunit_fs == 'kpc':
		distance_fs = dist_fs*1000.0*pc	
		
	if distunit_fs == 'Mpc':
		distance_fs = dist_fs*1000000.0*pc
		
	if timeunit_fs == 'Seconds':
		sitime_fs = time_fs
	
	if timeunit_fs == 'Years':
		sitime_fs = time_fs * yr
		
	if timeunit_fs == 'Myrs':
		sitime_fs = time_fs * myr	
		
	if timeunit_fs == 'Gyrs':
		sitime_fs = time_fs * (1000.0*myr)
		
	if sitime_fs > 0.0:
		# Calculate speed in km/s
		sispeed = (distance_fs/1000.) / sitime_fs
		# And also in kpc / Myr
		stnspeed = (distance_fs/ (1000.0*pc)) / (sitime_fs / myr)
		
		st.write("#### Average speed = ", nicenumber(sispeed),' in km/s, or ', nicenumber(stnspeed),' kpc/Myr.') 	
		st.write('Exact values are '+str(sispeed)+' in km/s and '+str(stnspeed)+' in kpc/Myr.')
