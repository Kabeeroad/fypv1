import streamlit as st

st.set_page_config(page_title="About", page_icon="🌎")

st.markdown("# About")
st.write("This is a research project that aims to tackle fake news by utilising Machine Learning and Large Language models.")

#st.sidebar.header("About")




tab1, tab2 = st.tabs(["Purpose", "Technical Specifications"])


tab1.subheader("Purpose")
tab1.write("Fake News is becoming so prevalent in our societies and this Fake News Detector is a frontline response in classifying if articles are false or real!")


tab2.subheader("Technical Specifications")
tab2.write("Test")


