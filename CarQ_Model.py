import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CarQ | AI Vehicle Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CARQ UI STYLE
# =========================================================

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#eef6ff 0%,#f8fbff 50%,#e8f2ff 100%); color:#102a43; }
.block-container { max-width:1150px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { color:#0b2d5c !important; }
p,label,.stMarkdown,.stCaption { color:#334e68; }
[data-testid="stMetric"] { background:#fff; border:1px solid #c9def5; border-radius:14px; padding:16px; box-shadow:0 4px 14px rgba(30,90,150,.08); }
[data-testid="stMetricLabel"] { color:#627d98 !important; }
[data-testid="stMetricValue"] { color:#0b4ea2 !important; }
[data-testid="stFileUploader"] { background:#fff; border:2px dashed #82b7ee; border-radius:14px; padding:12px; }
.stButton > button { width:100%; border-radius:10px; background:linear-gradient(90deg,#0b5ed7,#1683e8); color:white; border:1px solid #0b5ed7; font-weight:600; box-shadow:0 4px 12px rgba(11,94,215,.20); }
.stButton > button:hover { background:linear-gradient(90deg,#084cba,#0b6fd3); color:white; }
div[data-testid="stImage"] { border-radius:14px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#c9def5; }
[data-testid="stFileUploaderDropzone"] { background:#f7fbff; border-radius:12px; }
a { color:#0b5ed7 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "best_car_model.keras"
    )


model = load_model()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Altroz",
    "Baleno",
    "Bolero",
    "Brezza",
    "Creta",
    "Dzire",
    "Exter",
    "Grand Vitara",
    "Harrier",
    "I20",
    "Nexon",
    "Punch",
    "Scorpio-N",
    "Swift",
    "Thar",
    "Tiago",
    "Venue",
    "Verna",
    "XUV 3XO",
    "XUV 700"
]


# =========================================================
# BRAND MAPPING
# =========================================================

brand_mapping = {

    "Altroz": "Tata",
    "Baleno": "Maruti Suzuki",
    "Bolero": "Mahindra",
    "Brezza": "Maruti Suzuki",
    "Creta": "Hyundai",
    "Dzire": "Maruti Suzuki",
    "Exter": "Hyundai",
    "Grand Vitara": "Maruti Suzuki",
    "Harrier": "Tata",
    "I20": "Hyundai",
    "Nexon": "Tata",
    "Punch": "Tata",
    "Scorpio-N": "Mahindra",
    "Swift": "Maruti Suzuki",
    "Thar": "Mahindra",
    "Tiago": "Tata",
    "Venue": "Hyundai",
    "Verna": "Hyundai",
    "XUV 3XO": "Mahindra",
    "XUV 700": "Mahindra"
}


# =========================================================
# VEHICLE DATABASE
# =========================================================

vehicle_specs = {

    "Swift": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 24 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },

    "Dzire": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 24 km/l",
        "Price": "₹6.5 - ₹10 Lakh"
    },

    "Baleno": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 22 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },

    "Brezza": {
        "Engine": "1.5L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 19 km/l",
        "Price": "₹8.5 - ₹14 Lakh"
    },

    "Grand Vitara": {
        "Engine": "1.5L Petrol",
        "Fuel": "Petrol / Hybrid",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 21 km/l",
        "Price": "₹11 - ₹20 Lakh"
    },

    "Creta": {
        "Engine": "1.5L Petrol / Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 17-21 km/l",
        "Price": "₹11 - ₹20 Lakh"
    },

    "Venue": {
        "Engine": "1.2L / 1.0L",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 18-20 km/l",
        "Price": "₹8 - ₹14 Lakh"
    },

    "Exter": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 19 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },

    "I20": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / CVT",
        "Seating": "5",
        "Mileage": "Approx. 20 km/l",
        "Price": "₹7 - ₹11 Lakh"
    },

    "Verna": {
        "Engine": "1.5L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 18-20 km/l",
        "Price": "₹11 - ₹18 Lakh"
    },

    "Altroz": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / DCA",
        "Seating": "5",
        "Mileage": "Approx. 19 km/l",
        "Price": "₹6.5 - ₹11 Lakh"
    },

    "Nexon": {
        "Engine": "1.2L Turbo Petrol",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 17-24 km/l",
        "Price": "₹8 - ₹15 Lakh"
    },

    "Punch": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 18-20 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },

    "Harrier": {
        "Engine": "2.0L Diesel",
        "Fuel": "Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 16 km/l",
        "Price": "₹15 - ₹25 Lakh"
    },

    "Tiago": {
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 20 km/l",
        "Price": "₹5 - ₹8 Lakh"
    },

    "Thar": {
        "Engine": "2.0L Petrol / 2.2L Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "4",
        "Mileage": "Approx. 10-15 km/l",
        "Price": "₹11 - ₹18 Lakh"
    },

    "Scorpio-N": {
        "Engine": "2.0L Petrol / 2.2L Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "6 / 7",
        "Mileage": "Approx. 14-16 km/l",
        "Price": "₹14 - ₹25 Lakh"
    },

    "XUV 3XO": {
        "Engine": "1.2L Turbo Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 18-21 km/l",
        "Price": "₹8 - ₹16 Lakh"
    },

    "XUV 700": {
        "Engine": "2.0L Petrol / 2.2L Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5 / 7",
        "Mileage": "Approx. 12-16 km/l",
        "Price": "₹14 - ₹27 Lakh"
    },

    "Bolero": {
        "Engine": "1.5L Diesel",
        "Fuel": "Diesel",
        "Transmission": "Manual",
        "Seating": "7",
        "Mileage": "Approx. 16 km/l",
        "Price": "₹10 - ₹12 Lakh"
    }
}


# =========================================================
# HEADER
# =========================================================

st.title("🚗 CarQ")
st.caption("AI-Powered Vehicle Recognition & Information System")

st.divider()

# =========================================================
# UPLOAD
# =========================================================

st.subheader("🔍 Identify Your Vehicle")

uploaded_file = st.file_uploader(
    "Upload a clear image of the car",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, upload a clear front, side or 3/4 view."
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("CarQ AI is analyzing your vehicle..."):

        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(
            img_array,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(predictions))
        predicted_model = class_names[predicted_index]

        confidence = float(
            predictions[predicted_index] * 100
        )

        predicted_brand = brand_mapping[predicted_model]

    specs = vehicle_specs.get(predicted_model, {})

    # =====================================================
    # IMAGE + AI RESULT
    # =====================================================

    st.divider()
    st.subheader("🤖 AI Recognition")

    image_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )

    with image_col:
        st.image(
            image,
            caption="Uploaded Vehicle",
            use_container_width=True
        )

    with result_col:
        st.markdown("### Identified Vehicle")
        st.markdown(f"## 🚘 {predicted_model}")
        st.write(f"**Manufacturer:** {predicted_brand}")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(confidence / 100, 1.0)
        )

        if confidence >= 80:
            st.success("High-confidence prediction")
        elif confidence >= 60:
            st.warning("Moderate-confidence prediction")
        else:
            st.error("Low-confidence prediction — try a clearer image")

    # =====================================================
    # VEHICLE INFORMATION
    # =====================================================

    st.divider()
    st.subheader("📋 Vehicle Information")

    if specs:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("⚙️ Engine", specs["Engine"])

        with col2:
            st.metric("⛽ Fuel Type", specs["Fuel"])

        with col3:
            st.metric("⚡ Transmission", specs["Transmission"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("👥 Seating Capacity", specs["Seating"])

        with col2:
            st.metric("🛣️ Mileage", specs["Mileage"])

        with col3:
            st.metric("💰 Estimated Price", specs["Price"])

    # =====================================================
    # TOP 3 PREDICTIONS
    # =====================================================

    st.divider()
    st.subheader("📊 AI Prediction Analysis")

    top_indices = np.argsort(predictions)[-3:][::-1]

    for rank, index in enumerate(top_indices, start=1):

        model_name = class_names[index]
        model_confidence = float(predictions[index] * 100)

        st.write(
            f"**{rank}. {model_name}** — "
            f"{model_confidence:.2f}%"
        )

        st.progress(
            min(float(predictions[index]), 1.0)
        )

    # =====================================================
    # INSIGHT
    # =====================================================

    st.info(
        f"💡 **CarQ Insight:** The model identified this vehicle "
        f"as **{predicted_brand} {predicted_model}** with "
        f"**{confidence:.2f}%** confidence. "
        f"Vehicle specifications are retrieved from the CarQ "
        f"vehicle information database."
    )

    # =====================================================
    # ANALYZE ANOTHER VEHICLE
    # =====================================================

    if st.button("🔄 Analyze Another Vehicle"):
        st.rerun()

else:

    st.info(
        "🚘 Upload a vehicle image above to start AI-powered recognition."
    )

    st.markdown("""
    ### How CarQ Works

    **1. Upload Image** → **2. AI Recognition** → **3. Brand & Model**
    → **4. Vehicle Specifications** → **5. Estimated Price**
    """)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CarQ • Deep Learning • Computer Vision • Vehicle Intelligence"
)

