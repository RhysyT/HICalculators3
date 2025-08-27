# minimal_number_input_test.py
import streamlit as st

st.set_page_config(page_title="Minimal number_input test", layout="wide")

st.title("Minimal number_input test — sidebar vs main")

#with st.sidebar:
#    st.header("Sidebar")
#    a = st.number_input("Sidebar number", value=0.0, step=1.0, format="%.3f", key="sidebar_num")

st.header("Main area")
b = st.number_input("Main number", value=0.0, step=0.01, format="%.3f", key="main_num")

st.write("You typed:", b)


