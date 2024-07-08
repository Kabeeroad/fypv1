import streamlit as st

st.set_page_config(page_title="About", page_icon="🌎")

st.markdown("# About")
st.write("This is a webapp that has implemented cutting edge tools from our research aiming to tackle fake news by utilising Machine Learning and Large Language models.")
#st.write("See the research paper here -> TBC")





tab1, tab2 = st.tabs(["Purpose", "Technical Specifications"])


tab1.subheader("Purpose")
tab1.write("Fake News is becoming so prevalent in our societies and has been proven to influence political views and even cause harm to individuals and organisations. This situation has been escalated since Large Language Models (e.g.ChatGPT) can massively produce fake news written in the style of fake news. We need to ensure what we read online is true.")
tab1.write("This Fake News Detector is a frontline response in classifying if articles are false or real!")


tab2.subheader("Technical Specifications")
tab2.write("The fake news detector framework works like this:")
tab2.image("pages/images/Framework.png")
tab2.write("This uses a Logistic Regression algorithim trained on around 76,000 news articles achieving 0.95 accuracy.")
tab2.write("To fact check the claims, Meta's llama3-8b LLM model with the tools of duckduckgo and wikipedia are used.")

