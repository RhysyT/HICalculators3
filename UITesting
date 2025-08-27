# diagnose_number_input.py
import streamlit as st, os, pathlib

st.title("number_input behaviour matrix")

st.caption(f"Streamlit {st.__version__}")
st.caption("CWD  —  " + os.getcwd())
st.caption("Project config.toml present  —  " + str(pathlib.Path(".streamlit/config.toml").exists()))

st.subheader("Main area")

c1, c2 = st.columns(2)
with c1:
    st.write("A  —  float, with format, step 1.0  —  often locks")
    a = st.number_input("A", value=0.0, step=1.0, format="%.3f", key="A")
    st.write(a)

    st.write("B  —  float, no format, no step  —  usually fine")
    b = st.number_input("B", value=0.0, key="B")
    st.write(b)

with c2:
    st.write("C  —  int, step 1  —  usually fine")
    c = st.number_input("C", value=0, step=1, key="C")
    st.write(c)

    st.write("D  —  float, tiny step 0.001, no format  —  usually fine")
    d = st.number_input("D", value=0.0, step=0.001, key="D")
    st.write(d)

st.subheader("Inside a form — typing should buffer until submit")
with st.form("F"):
    e = st.number_input("E  —  like A", value=0.0, step=1.0, format="%.3f", key="E")
    f = st.number_input("F  —  like B", value=0.0, key="F")
    g = st.number_input("G  —  like C", value=0, step=1, key="G")
    h = st.number_input("H  —  like D", value=0.0, step=0.001, key="H")
    submitted = st.form_submit_button("Submit")
st.write({"E": e, "F": f, "G": g, "H": h})
