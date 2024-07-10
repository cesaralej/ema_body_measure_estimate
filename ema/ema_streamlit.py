import streamlit as st
from ema import getMeasurements

#with open("style_title.css") as css:
 #   st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.title("ema")
st.write("""_Welcome to **ema**, we are transforming fashion with virtual try-ons_""")

with open("style_body.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

st.write("""*Please upload an image of your **full body** without cutting any part, preferable one-color background*""")

input_image = st.file_uploader("image", type=["jpg", "jpeg", "png"])

st.write("""_Please input your **height in cm**_""")

height = st.number_input("height", min_value=0, max_value=300, value=170)

# get user gender
st.write("""_Please select the gender with which you identify the most_""")

# dropdown list with gender
gender = st.selectbox("gender", options=("women","men","prefer not to say"))

# button to get measurements
if st.button("Get measurements"):
    st.write(getMeasurements(input_image, height, gender))