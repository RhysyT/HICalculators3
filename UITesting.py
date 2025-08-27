# minimal_number_input_test.py
import streamlit as st

# Streamlit style preferences
# Remove the menu button
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Remove vertical whitespace padding
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.write('<style>div.block-container{padding-bottom:0rem;}</style>', unsafe_allow_html=True)

st.set_page_config(page_title="Minimal number_input test", layout="wide")

#with st.sidebar:
#    st.header("Sidebar")
#    a = st.number_input("Sidebar number", value=0.0, step=1.0, format="%.3f", key="sidebar_num")

st.header("Main area")
b = st.number_input("Main number", value=0.0)

st.write("You typed:", b)





