from io import BytesIO
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from gtts import gTTS

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
# GRAD-CAM HELPERS
# =========================================================

def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            idx = model.layers.index(layer)
            for prev in reversed(model.layers[:idx]):
                shape = getattr(prev, "output_shape", None)
                if shape is not None and len(shape) == 4:
                    return prev.name
    return "top_activation"


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_index: int | None = None,
    last_conv_layer_name: str | None = None,
) -> tuple[np.ndarray, int]:
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    target_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    img_tensor = tf.cast(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if isinstance(predictions, list):
            predictions = predictions[0]
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), int(class_index)


def overlay_gradcam(
    original_rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> np.ndarray:
    h, w, _ = original_rgb_image.shape
    heatmap_pil = Image.fromarray(np.uint8(255 * np.clip(heatmap, 0.0, 1.0)))
    heatmap_resized = np.array(
        heatmap_pil.resize((w, h), resample=Image.Resampling.BICUBIC)
    ) / 255.0

    if hasattr(mpl, "colormaps"):
        colormap = mpl.colormaps[colormap_name]
    else:
        colormap = plt.get_cmap(colormap_name)

    heatmap_colored_rgb = np.uint8(255 * colormap(heatmap_resized)[:, :, :3])

    superimposed = (
        original_rgb_image.astype(np.float32) * (1.0 - alpha)
        + heatmap_colored_rgb.astype(np.float32) * alpha
    )
    return np.clip(superimposed, 0, 255).astype(np.uint8)


def render_split_comparison(before_rgb: np.ndarray, after_rgb: np.ndarray, height: int = 380):
    """Interactive draggable before/after split slider between input image and Grad-CAM."""
    import base64

    def to_b64(arr: np.ndarray) -> str:
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    b64_before = to_b64(before_rgb)
    b64_after = to_b64(after_rgb)

    html = f"""
    <div id="split-box" style="position:relative; width:100%; max-width:520px; height:{height}px; margin:0 auto; border-radius:14px; overflow:hidden; border:1px solid #c9def5; user-select:none;">
      <img src="data:image/png;base64,{b64_after}" style="position:absolute; inset:0; width:100%; height:100%; object-fit:cover;" />
      <div id="before-wrap" style="position:absolute; inset:0; width:50%; overflow:hidden;">
        <img src="data:image/png;base64,{b64_before}" style="width:100%; height:100%; max-width:none; object-fit:cover;" />
      </div>
      <div id="divider" style="position:absolute; top:0; bottom:0; left:50%; width:3px; background:#ffffff; box-shadow:0 0 8px rgba(0,0,0,0.45); cursor:ew-resize;">
        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); background:#0b5ed7; color:#fff; width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:14px; font-weight:700; box-shadow:0 2px 8px rgba(0,0,0,0.35);">↔</div>
      </div>
      <span style="position:absolute; top:10px; left:10px; background:rgba(11,45,92,0.78); color:#fff; padding:3px 9px; border-radius:8px; font-size:12px;">Original</span>
      <span style="position:absolute; top:10px; right:10px; background:rgba(11,45,92,0.78); color:#fff; padding:3px 9px; border-radius:8px; font-size:12px;">Grad-CAM</span>
    </div>
    <script>
      const box = document.getElementById('split-box');
      const wrap = document.getElementById('before-wrap');
      const divLine = document.getElementById('divider');
      const beforeImg = wrap.querySelector('img');
      let dragging = false;

      function syncSizes() {{
        beforeImg.style.width = box.clientWidth + 'px';
        beforeImg.style.height = box.clientHeight + 'px';
      }}
      syncSizes();
      window.addEventListener('resize', syncSizes);

      function setPos(clientX) {{
        const rect = box.getBoundingClientRect();
        let pct = ((clientX - rect.left) / rect.width) * 100;
        pct = Math.max(5, Math.min(95, pct));
        wrap.style.width = pct + '%';
        divLine.style.left = pct + '%';
      }}

      box.addEventListener('mousedown', (e) => {{ dragging = true; setPos(e.clientX); }});
      window.addEventListener('mouseup', () => {{ dragging = false; }});
      window.addEventListener('mousemove', (e) => {{ if (dragging) setPos(e.clientX); }});

      box.addEventListener('touchstart', (e) => {{ dragging = true; if (e.touches[0]) setPos(e.touches[0].clientX); }});
      window.addEventListener('touchend', () => {{ dragging = false; }});
      window.addEventListener('touchmove', (e) => {{ if (dragging && e.touches[0]) setPos(e.touches[0].clientX); }});
    </script>
    """
    components.html(html, height=height + 16)

# =========================================================
# BUYER PERSONA MATCHING (HATKE INSIGHT)
# =========================================================

def compute_buyer_fit(specs: dict) -> dict[str, int]:
    segment = specs.get("Segment", "").lower()
    seating = str(specs.get("Seating", "5"))
    mileage = specs.get("Mileage", "").lower()
    price = specs.get("Price", "").lower()

    city_score = 72
    family_score = 70
    highway_score = 70
    value_score = 72

    if "hatchback" in segment or "micro" in segment:
        city_score += 20
        value_score += 18
        family_score -= 10
    elif "compact sedan" in segment:
        city_score += 14
        family_score += 14
        value_score += 14
    elif "compact suv" in segment:
        city_score += 12
        family_score += 12
        highway_score += 12
    elif "mid-size" in segment:
        family_score += 18
        highway_score += 18
        city_score -= 5
    elif "3-row" in segment or "utility" in segment:
        family_score += 24
        highway_score += 14
        city_score -= 12
    elif "off-road" in segment:
        highway_score += 22
        city_score -= 15
        family_score -= 12

    if "24" in mileage or "22" in mileage or "21" in mileage or "20" in mileage:
        city_score += 6
        value_score += 8

    if "6 / 7" in seating or "7" in seating:
        family_score += 6

    if "₹5" in price or "₹6" in price or "₹6.5" in price or "₹7" in price:
        value_score += 8

    clamp = lambda x: max(25, min(98, int(x)))
    return {
        "🏙️ Daily City Commuter": clamp(city_score),
        "👨‍👩‍👧‍👦 Family & Group Travel": clamp(family_score),
        "🛣️ Highway & Road Trips": clamp(highway_score),
        "💡 Value & Running Cost": clamp(value_score),
    }

# =========================================================
# DETAILED MALE VOICE ASSISTANT HELPERS
# =========================================================

def format_top_predictions_spoken(
    predictions: np.ndarray,
    class_names: list[str],
    brand_mapping: dict[str, str],
) -> str:
    top_idx = np.argsort(predictions)[-3:][::-1]
    parts = []
    for rank, idx in enumerate(top_idx, start=1):
        m_name = class_names[idx]
        b_name = brand_mapping.get(m_name, "")
        conf = float(predictions[idx] * 100)
        parts.append(f"number {rank}, {b_name} {m_name} at {conf:.1f} percent")
    return ", ".join(parts)


def build_detailed_briefing_text(
    brand: str,
    model_name: str,
    confidence: float,
    specs: dict,
    predictions: np.ndarray,
    class_names: list[str],
    brand_mapping: dict[str, str],
) -> str:
    top_summary = format_top_predictions_spoken(predictions, class_names, brand_mapping)
    seating_val = specs.get("Seating", "N/A")
    segment_val = specs.get("Segment", "passenger vehicle")
    fits = compute_buyer_fit(specs)
    best_persona = max(fits, key=fits.get)

    return (
        f"Hello, welcome to CarQ vehicle intelligence. "
        f"Based on visual inspection, this {segment_val} is identified as a {brand} {model_name}, "
        f"with a confidence score of {confidence:.1f} percent. "
        f"Looking at the top three model probabilities, we have {top_summary}. "
        f"Under the hood, it features a {specs.get('Engine', 'not available')} engine "
        f"paired with {specs.get('Fuel', 'not available')} fuel and a "
        f"{specs.get('Transmission', 'not available')} gearbox. "
        f"Inside, it offers a {seating_val} seating setup, delivers an approximate mileage of "
        f"{specs.get('Mileage', 'not available')}, and sits in the estimated price range of "
        f"{specs.get('Price', 'not available')}. "
        f"Our buyer fit analysis shows its strongest match is {best_persona.split(' ', 1)[-1]} "
        f"at {fits[best_persona]} percent. "
        f"Drag the split slider in the Grad-CAM section to compare the original car image "
        f"against the model's visual attention heatmap."
    )


def build_short_summary_text(
    brand: str,
    model_name: str,
    confidence: float,
    specs: dict,
) -> str:
    return (
        f"CarQ detected a {brand} {model_name} with {confidence:.1f} percent confidence. "
        f"It has a {specs.get('Engine', 'N/A')} engine, {specs.get('Fuel', 'N/A')} fuel, "
        f"{specs.get('Mileage', 'N/A')} mileage, and an estimated price of {specs.get('Price', 'N/A')}."
    )


def answer_vehicle_question(
    question: str,
    brand: str,
    model_name: str,
    specs: dict,
    confidence: float,
    predictions: np.ndarray,
    class_names: list[str],
    brand_mapping: dict[str, str],
) -> str:
    q = (question or "").lower().strip()
    if not q:
        return build_detailed_briefing_text(
            brand, model_name, confidence, specs, predictions, class_names, brand_mapping
        )

    if any(k in q for k in ["price", "cost", "budget", "lakh", "₹", "rs", "afford"]):
        return (
            f"For the {brand} {model_name}, the estimated price bracket is "
            f"{specs.get('Price', 'not available')}. Paired with {specs.get('Fuel', 'standard fuel')} "
            f"and an estimated mileage of {specs.get('Mileage', 'N/A')}, total ownership cost "
            f"depends on your monthly running and variant selection."
        )

    if any(k in q for k in ["mileage", "average", "km/l", "fuel economy", "efficiency"]):
        return (
            f"The {brand} {model_name} delivers an approximate mileage of {specs.get('Mileage', 'not available')}. "
            f"With a {specs.get('Engine', 'N/A')} engine and {specs.get('Transmission', 'N/A')} gearbox, "
            f"actual fuel efficiency varies with city traffic, highway cruising, and AC usage."
        )

    if any(k in q for k in ["engine", "cc", "motor", "power", "performance"]):
        return (
            f"The {brand} {model_name} uses a {specs.get('Engine', 'not available')} engine "
            f"with {specs.get('Fuel', 'not available')} fuel and {specs.get('Transmission', 'not available')} "
            f"transmission, tailored for {specs.get('Segment', 'everyday driving')}."
        )

    if any(k in q for k in ["fuel", "petrol", "diesel", "hybrid", "cng", "ev"]):
        return (
            f"The {brand} {model_name} lists {specs.get('Fuel', 'not available')} as its fuel type, "
            f"with a {specs.get('Engine', 'N/A')} powertrain and around {specs.get('Mileage', 'N/A')} mileage."
        )

    if any(k in q for k in ["transmission", "gear", "manual", "automatic", "amt", "cvt", "dca"]):
        return (
            f"The {brand} {model_name} features {specs.get('Transmission', 'not available')} transmission. "
            f"Automatic or AMT options ease bumper-to-bumper traffic, while manual shifts suit enthusiasts."
        )

    if any(k in q for k in ["seat", "seating", "capacity", "passenger", "seater", "family", "space", "fit", "buyer"]):
        fits = compute_buyer_fit(specs)
        fit_spoken = ", ".join([f"{k.split(' ', 1)[-1]} {v} percent" for k, v in fits.items()])
        return (
            f"The {brand} {model_name} has a {specs.get('Seating', 'not available')} seating layout "
            f"in the {specs.get('Segment', 'passenger vehicle')} class. Its buyer fit scores are: {fit_spoken}."
        )

    if any(k in q for k in ["compare", "top", "other", "alternative", "second", "third", "similar"]):
        top_summary = format_top_predictions_spoken(predictions, class_names, brand_mapping)
        return (
            f"While {brand} {model_name} leads with {confidence:.1f} percent confidence, "
            f"the top three predictions are: {top_summary}."
        )

    return build_detailed_briefing_text(
        brand, model_name, confidence, specs, predictions, class_names, brand_mapping
    )


def text_to_male_mp3_bytes(text: str, lang: str = "en", tld: str = "co.uk", slow: bool = False) -> BytesIO:
    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# =========================================================
# LOAD MODEL & CACHE WRAPPERS
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_car_model.keras")


@st.cache_data(show_spinner=False)
def compute_gradcam_overlay(img_bytes: bytes, class_index: int, alpha: float):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    resized_rgb = np.array(img.resize((224, 224)))
    img_array = np.expand_dims(resized_rgb, axis=0)
    heatmap, _ = generate_gradcam_heatmap(model, img_array, class_index=class_index)
    overlay = overlay_gradcam(resized_rgb, heatmap, alpha=alpha)
    return resized_rgb, overlay


@st.cache_data(show_spinner=False)
def synthesize_male_speech(spoken_text: str, tld: str = "co.uk") -> bytes:
    return text_to_male_mp3_bytes(spoken_text, lang="en", tld=tld, slow=False).getvalue()


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
        "Segment": "Hatchback",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 24 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },
    "Dzire": {
        "Segment": "Compact Sedan",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 24 km/l",
        "Price": "₹6.5 - ₹10 Lakh"
    },
    "Baleno": {
        "Segment": "Premium Hatchback",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 22 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },
    "Brezza": {
        "Segment": "Compact SUV",
        "Engine": "1.5L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 19 km/l",
        "Price": "₹8.5 - ₹14 Lakh"
    },
    "Grand Vitara": {
        "Segment": "Mid-size SUV",
        "Engine": "1.5L Petrol",
        "Fuel": "Petrol / Hybrid",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 21 km/l",
        "Price": "₹11 - ₹20 Lakh"
    },
    "Creta": {
        "Segment": "Mid-size SUV",
        "Engine": "1.5L Petrol / Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 17-21 km/l",
        "Price": "₹11 - ₹20 Lakh"
    },
    "Venue": {
        "Segment": "Compact SUV",
        "Engine": "1.2L / 1.0L",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 18-20 km/l",
        "Price": "₹8 - ₹14 Lakh"
    },
    "Exter": {
        "Segment": "Micro SUV",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 19 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },
    "I20": {
        "Segment": "Premium Hatchback",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / CVT",
        "Seating": "5",
        "Mileage": "Approx. 20 km/l",
        "Price": "₹7 - ₹11 Lakh"
    },
    "Verna": {
        "Segment": "Mid-size Sedan",
        "Engine": "1.5L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 18-20 km/l",
        "Price": "₹11 - ₹18 Lakh"
    },
    "Altroz": {
        "Segment": "Premium Hatchback",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / DCA",
        "Seating": "5",
        "Mileage": "Approx. 19 km/l",
        "Price": "₹6.5 - ₹11 Lakh"
    },
    "Nexon": {
        "Segment": "Compact SUV",
        "Engine": "1.2L Turbo Petrol",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 17-24 km/l",
        "Price": "₹8 - ₹15 Lakh"
    },
    "Punch": {
        "Segment": "Micro SUV",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 18-20 km/l",
        "Price": "₹6 - ₹10 Lakh"
    },
    "Harrier": {
        "Segment": "Mid-size SUV",
        "Engine": "2.0L Diesel",
        "Fuel": "Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 16 km/l",
        "Price": "₹15 - ₹25 Lakh"
    },
    "Tiago": {
        "Segment": "Hatchback",
        "Engine": "1.2L Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / AMT",
        "Seating": "5",
        "Mileage": "Approx. 20 km/l",
        "Price": "₹5 - ₹8 Lakh"
    },
    "Thar": {
        "Segment": "Lifestyle Off-road SUV",
        "Engine": "2.0L Petrol / 2.2L Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "4",
        "Mileage": "Approx. 10-15 km/l",
        "Price": "₹11 - ₹18 Lakh"
    },
    "Scorpio-N": {
        "Segment": "3-Row SUV",
        "Engine": "2.0L Petrol / 2.2L Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "6 / 7",
        "Mileage": "Approx. 14-16 km/l",
        "Price": "₹14 - ₹25 Lakh"
    },
    "XUV 3XO": {
        "Segment": "Compact SUV",
        "Engine": "1.2L Turbo Petrol",
        "Fuel": "Petrol",
        "Transmission": "Manual / Automatic",
        "Seating": "5",
        "Mileage": "Approx. 18-21 km/l",
        "Price": "₹8 - ₹16 Lakh"
    },
    "XUV 700": {
        "Segment": "3-Row SUV",
        "Engine": "2.0L Petrol / 2.2L Diesel",
        "Fuel": "Petrol / Diesel",
        "Transmission": "Manual / Automatic",
        "Seating": "5 / 7",
        "Mileage": "Approx. 12-16 km/l",
        "Price": "₹14 - ₹27 Lakh"
    },
    "Bolero": {
        "Segment": "Utility SUV",
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
st.caption("AI-Powered Vehicle Recognition, Split Grad-CAM Explainability & Male Voice Intelligence")

st.divider()

# =========================================================
# INPUT: FILE UPLOAD OR LIVE CAMERA
# =========================================================

st.subheader("🔍 Identify Your Vehicle")

input_mode = st.radio(
    "Choose Image Input Source",
    options=["📁 Upload Image File", "📷 Live Camera Snap"],
    horizontal=True,
)

if input_mode == "📁 Upload Image File":
    active_media = st.file_uploader(
        "Upload a clear image of the car",
        type=["jpg", "jpeg", "png", "webp"],
        help="For best results, upload a clear front, side or 3/4 view."
    )
else:
    active_media = st.camera_input("Capture a vehicle photo")

# =========================================================
# PREDICTION
# =========================================================

if active_media is not None:
    raw_bytes = active_media.getvalue()
    image = Image.open(BytesIO(raw_bytes)).convert("RGB")

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
            caption="Input Vehicle",
            use_container_width=True
        )

    with result_col:
        st.markdown("### Identified Vehicle")
        st.markdown(f"## 🚘 {predicted_model}")
        st.write(f"**Manufacturer:** {predicted_brand}")
        st.write(f"**Body Segment:** {specs.get('Segment', 'N/A')}")

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
    # INTERACTIVE SPLIT GRAD-CAM EXPLANATION
    # =====================================================

    st.divider()
    st.subheader("🔥 Interactive Split Grad-CAM")
    st.caption(
        "Drag the ↔ handle left or right to compare the Original image and the Grad-CAM heatmap directly."
    )

    cam_col_1, cam_col_2 = st.columns([1, 1])
    with cam_col_1:
        selected_cam_class = st.selectbox(
            "Select Class for Heatmap",
            options=class_names,
            index=predicted_index,
        )
        selected_cam_index = class_names.index(selected_cam_class)

    with cam_col_2:
        overlay_alpha = st.slider(
            "Overlay Intensity",
            min_value=0.15,
            max_value=0.80,
            value=0.45,
            step=0.05,
        )

    with st.spinner("Generating Grad-CAM explanation..."):
        resized_rgb, cam_overlay = compute_gradcam_overlay(
            raw_bytes, selected_cam_index, overlay_alpha
        )

    render_split_comparison(resized_rgb, cam_overlay, height=380)

    # =====================================================
    # VEHICLE INFORMATION & BUYER PERSONA FIT
    # =====================================================

    st.divider()
    st.subheader("📋 Vehicle Information & Buyer Fit")

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

        st.markdown("#### 🎯 Buyer Use-Case Fit")
        persona_scores = compute_buyer_fit(specs)
        p_cols = st.columns(2)
        for idx, (persona_label, score) in enumerate(persona_scores.items()):
            with p_cols[idx % 2]:
                st.write(f"**{persona_label}** — {score}% fit")
                st.progress(score / 100.0)

    # =====================================================
    # DETAILED MALE VOICE ASSISTANT
    # =====================================================

    st.divider()
    st.subheader("🎙️ CarQ Male Voice Assistant")
    st.caption(
        "Listen to a detailed spoken briefing on specs, buyer fit, top-3 matches, or ask custom car questions."
    )

    if "voice_reply_text" not in st.session_state:
        st.session_state.voice_reply_text = ""

    voice_style = st.selectbox(
        "Voice Accent Profile",
        options=["co.uk (UK Male-style)", "com (US)", "com.au (Australian)", "co.in (Indian)"],
        index=0,
    )
    selected_tld = voice_style.split(" ")[0]

    voice_btn_1, voice_btn_2 = st.columns(2)
    with voice_btn_1:
        if st.button("🔊 Detailed Car Briefing"):
            st.session_state.voice_reply_text = build_detailed_briefing_text(
                predicted_brand,
                predicted_model,
                confidence,
                specs,
                predictions,
                class_names,
                brand_mapping,
            )
    with voice_btn_2:
        if st.button("⚡ Quick Spec Summary"):
            st.session_state.voice_reply_text = build_short_summary_text(
                predicted_brand,
                predicted_model,
                confidence,
                specs,
            )

    quick_cols = st.columns(5)
    quick_queries = [
        ("💰 Price", "What is the estimated price?"),
        ("🛣️ Mileage", "What is the mileage?"),
        ("⚙️ Engine", "What is the engine specification?"),
        ("🎯 Buyer Fit", "What is the buyer fit and seating capacity?"),
        ("📊 Top-3", "Compare the top three predictions"),
    ]
    for q_col, (q_label, q_text) in zip(quick_cols, quick_queries):
        with q_col:
            if st.button(q_label):
                st.session_state.voice_reply_text = answer_vehicle_question(
                    q_text,
                    predicted_brand,
                    predicted_model,
                    specs,
                    confidence,
                    predictions,
                    class_names,
                    brand_mapping,
                )

    custom_question = st.text_input(
        "Ask a detailed question about this vehicle",
        placeholder="e.g., Tell me about mileage, transmission, price, buyer fit, or top 3 cars",
    )
    if st.button("💬 Ask & Speak"):
        st.session_state.voice_reply_text = answer_vehicle_question(
            custom_question,
            predicted_brand,
            predicted_model,
            specs,
            confidence,
            predictions,
            class_names,
            brand_mapping,
        )

    if st.session_state.voice_reply_text:
        st.write(f"**Assistant:** {st.session_state.voice_reply_text}")
        try:
            audio_bytes = synthesize_male_speech(
                st.session_state.voice_reply_text, tld=selected_tld
            )
            st.audio(audio_bytes, format="audio/mp3", autoplay=False)
        except Exception as e:
            st.warning(f"Voice playback unavailable right now: {e}")

    # =====================================================
    # TOP 3 PREDICTIONS & COMPARISON TABLE
    # =====================================================

    st.divider()
    st.subheader("📊 AI Prediction Analysis & Top-3 Comparison")

    top_indices = np.argsort(predictions)[-3:][::-1]
    comparison_rows = []

    for rank, index in enumerate(top_indices, start=1):
        model_name = class_names[index]
        brand_name = brand_mapping[model_name]
        model_confidence = float(predictions[index] * 100)
        cand_specs = vehicle_specs.get(model_name, {})

        st.write(
            f"**{rank}. {brand_name} {model_name}** — "
            f"{model_confidence:.2f}%"
        )

        st.progress(
            min(float(predictions[index]), 1.0)
        )

        comparison_rows.append({
            "Rank": rank,
            "Vehicle": f"{brand_name} {model_name}",
            "Confidence": f"{model_confidence:.2f}%",
            "Segment": cand_specs.get("Segment", "N/A"),
            "Engine": cand_specs.get("Engine", "N/A"),
            "Mileage": cand_specs.get("Mileage", "N/A"),
            "Estimated Price": cand_specs.get("Price", "N/A"),
        })

    st.table(comparison_rows)

    # =====================================================
    # INSIGHT & EXPORT SUMMARY
    # =====================================================

    insight_msg = (
        f"💡 **CarQ Insight:** The model identified this vehicle "
        f"as **{predicted_brand} {predicted_model}** "
        f"(*{specs.get('Segment', 'Vehicle')}*) with "
        f"**{confidence:.2f}%** confidence. "
        f"Vehicle specifications are retrieved from the CarQ "
        f"vehicle information database."
    )
    st.info(insight_msg)

    report_payload = {
        "brand": predicted_brand,
        "model": predicted_model,
        "segment": specs.get("Segment", "N/A"),
        "confidence_percent": round(confidence, 2),
        "gradcam_target_class": f"{brand_mapping[selected_cam_class]} {selected_cam_class}",
        "gradcam_alpha": overlay_alpha,
        "specifications": specs,
        "buyer_fit": compute_buyer_fit(specs),
        "top_3_predictions": [
            {
                "rank": r,
                "brand": brand_mapping[class_names[idx]],
                "model": class_names[idx],
                "confidence_percent": round(float(predictions[idx] * 100), 2),
            }
            for r, idx in enumerate(top_indices, start=1)
        ],
    }

    exp_col_1, exp_col_2 = st.columns(2)
    with exp_col_1:
        st.download_button(
            "📄 Download Inspection Report (JSON)",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=f"CarQ_{predicted_model.replace(' ', '_')}_Report.json",
            mime="application/json",
        )
    with exp_col_2:
        if st.button("🔄 Analyze Another Vehicle"):
            st.session_state.voice_reply_text = ""
            st.rerun()

else:

    st.info(
        "🚘 Upload a vehicle image or snap one with your camera above to start AI-powered recognition."
    )

    st.markdown("""
    ### How CarQ Works

    **1. Upload or Snap Photo** → **2. AI Recognition & Split Grad-CAM** → **3. Brand, Segment & Buyer Fit**
    → **4. Detailed Male Voice Assistant** → **5. Top-3 Comparison & Report Export**
    """)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CarQ • Deep Learning • Computer Vision • Vehicle Intelligence"
)