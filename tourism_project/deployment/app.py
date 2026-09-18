
import json
import joblib
import pandas as pd
import streamlit as st

import base64

from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    layout="wide"
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

PRE_MODEL_PATH = BASE_DIR / "pre_engagement_model.joblib"
POST_MODEL_PATH = BASE_DIR / "post_engagement_model.joblib"
CONFIG_PATH = BASE_DIR / "model_config.json"
HEADER_IMAGE_PATH = BASE_DIR / "header.png"

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
pre_model = joblib.load(PRE_MODEL_PATH)
post_model = joblib.load(POST_MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    model_config = json.load(f)

pre_threshold = model_config["pre_engagement"]["threshold"]
post_threshold = model_config["post_engagement"]["threshold"]


# ---------------------------------------------------------
# PAGE STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
        color: #666;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .result-box {
        border: 2px solid #d7d7d7;
        border-radius: 12px;
        padding: 1.3rem;
        text-align: center;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }

    .result-heading {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .result-buy {
        font-size: 2rem;
        font-weight: 800;
        color: #17833d;
    }

    .result-no-buy {
        font-size: 2rem;
        font-weight: 800;
        color: #b3261e;
    }

    .probability {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .awaiting {
        font-size: 1.15rem;
        font-weight: 600;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
# ---------------------------------------------------------
# RESPONSIVE HEADER IMAGE - NO CROPPING
# ---------------------------------------------------------

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


if HEADER_IMAGE_PATH.exists():

    encoded_image = get_base64_image(HEADER_IMAGE_PATH)

    st.markdown(
        f"""
        <div style="
            width: 100%;
            margin-bottom: 1.5rem;
        ">
            <img
                src="data:image/png;base64,{encoded_image}"
                style="
                    width: 100%;
                    height: auto;
                    display: block;
                    border-radius: 12px;
                "
            >
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="main-title">Tourism Package Purchase Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predict customer purchase likelihood before and after the sales engagement.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# RESULT DISPLAY
# ---------------------------------------------------------
def show_result(title, result, probability):

    if result is None:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-heading">{title}</div>
                <div class="awaiting">Awaiting Prediction</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    css_class = "result-buy" if result == "BUYS" else "result-no-buy"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-heading">{title}</div>
            <div class="{css_class}">{result}</div>
            <div class="probability">
                Purchase Probability: {probability:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "pre_result" not in st.session_state:
    st.session_state.pre_result = None

if "pre_probability" not in st.session_state:
    st.session_state.pre_probability = None

if "post_result" not in st.session_state:
    st.session_state.post_result = None

if "post_probability" not in st.session_state:
    st.session_state.post_probability = None


# ---------------------------------------------------------
# INPUT OPTIONS
# ---------------------------------------------------------
contact_options = [
    "Self Enquiry",
    "Company Invited"
]

occupation_options = [
    "Salaried",
    "Small Business",
    "Large Business",
    "Free Lancer"
]

gender_options = [
    "Male",
    "Female"
]

marital_options = [
    "Single",
    "Married",
    "Divorced",
    "Unmarried"
]

designation_options = [
    "Executive",
    "Manager",
    "Senior Manager",
    "AVP",
    "VP"
]

product_options = [
    "Basic",
    "Standard",
    "Deluxe",
    "Super Deluxe",
    "King"
]


# =========================================================
# PRE-ENGAGEMENT SECTION
# =========================================================
st.markdown(
    '<div class="section-title">Pre-Engagement Prediction</div>',
    unsafe_allow_html=True
)

st.caption(
    "Uses customer information available before the sales interaction."
)

pre_left, pre_middle, pre_right = st.columns(
    [1.2, 1.2, 0.85],
    gap="large"
)


# ---------------------------------------------------------
# PRE-ENGAGEMENT INPUTS - LEFT COLUMN
# ---------------------------------------------------------
with pre_left:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    contact_type = st.selectbox(
        "Type of Contact",
        contact_options
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        occupation_options
    )

    gender = st.selectbox(
        "Gender",
        gender_options
    )

    persons_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    preferred_star = st.selectbox(
        "Preferred Property Star",
        [1, 2, 3, 4, 5]
    )


# ---------------------------------------------------------
# PRE-ENGAGEMENT INPUTS - MIDDLE COLUMN
# ---------------------------------------------------------
with pre_middle:

    marital_status = st.selectbox(
        "Marital Status",
        marital_options
    )

    number_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=50,
        value=2
    )

    passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

    designation = st.selectbox(
        "Designation",
        designation_options
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=30000,
        step=1000
    )


# ---------------------------------------------------------
# PRE-ENGAGEMENT RESULT + BUTTON
# ---------------------------------------------------------
with pre_right:

    st.write("")
    st.write("")

    show_result(
        "Pre-Engagement Decision",
        st.session_state.pre_result,
        st.session_state.pre_probability
    )

    if st.button(
        "Predict Pre-Engagement",
        use_container_width=True,
        type="primary"
    ):

        pre_input = pd.DataFrame([{
            "Age": age,
            "TypeofContact": contact_type,
            "CityTier": city_tier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons_visiting,
            "PreferredPropertyStar": preferred_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_trips,
            "Passport": passport,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income
        }])

        probability = pre_model.predict_proba(pre_input)[:, 1][0]

        prediction = int(
            probability >= pre_threshold
        )

        st.session_state.pre_probability = probability
        st.session_state.pre_result = (
            "BUYS" if prediction == 1 else "DOES NOT BUY"
        )

        st.rerun()


# =========================================================
# DIVIDER
# =========================================================
st.divider()


# =========================================================
# POST-ENGAGEMENT SECTION
# =========================================================
st.markdown(
    '<div class="section-title">Post-Engagement Prediction</div>',
    unsafe_allow_html=True
)

st.caption(
    "Reuses the customer information above and adds information "
    "captured during the sales interaction."
)

post_left, post_middle, post_right = st.columns(
    [1.2, 1.2, 0.85],
    gap="large"
)


# ---------------------------------------------------------
# POST-ENGAGEMENT ADDITIONAL INPUTS
# ---------------------------------------------------------
with post_left:

    pitch_score = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        product_options
    )


with post_middle:

    number_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=20,
        value=2
    )

    duration_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        max_value=120,
        value=20
    )


# ---------------------------------------------------------
# POST-ENGAGEMENT RESULT + BUTTON
# ---------------------------------------------------------
with post_right:

    show_result(
        "Post-Engagement Decision",
        st.session_state.post_result,
        st.session_state.post_probability
    )

    if st.button(
        "Predict Post-Engagement",
        use_container_width=True,
        type="primary"
    ):

        post_input = pd.DataFrame([{

            # Existing pre-engagement information
            "Age": age,
            "TypeofContact": contact_type,
            "CityTier": city_tier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons_visiting,
            "PreferredPropertyStar": preferred_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_trips,
            "Passport": passport,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,

            # Additional post-engagement information
            "PitchSatisfactionScore": pitch_score,
            "ProductPitched": product_pitched,
            "NumberOfFollowups": number_followups,
            "DurationOfPitch": duration_pitch
        }])

        probability = post_model.predict_proba(post_input)[:, 1][0]

        prediction = int(
            probability >= post_threshold
        )

        st.session_state.post_probability = probability
        st.session_state.post_result = (
            "BUYS" if prediction == 1 else "DOES NOT BUY"
        )

        st.rerun()
