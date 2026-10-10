import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="SmartMess AI",
    page_icon="🍽️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# Colourful UI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #202b49, #342653);
    color: white;
}
h1 { color: #63e6be !important; }
h2, h3 { color: #ffd166 !important; }
p, label { color: #f1f5f9 !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #243b55, #4568a0);
    border: 1px solid #63e6be;
    padding: 20px;
    border-radius: 15px;
}
.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #10b981);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🍽️ SmartMess AI")
st.subheader("🌱 Intelligent Food Consumption & Waste Prediction")
st.write("Predict food requirements and help reduce mess food waste.")

# Create a DEMO training dataset.
# These are simulated values, not real mess records.
@st.cache_resource
def train_demo_model():
    rng = np.random.default_rng(42)
    n = 1500

    students = rng.integers(20, 501, n)
    meals = rng.choice(
        ["breakfast", "lunch", "snacks", "dinner"], n
    )
    special = rng.choice([0, 1], n)

    rates = {
        "breakfast": 0.22,
        "lunch": 0.40,
        "snacks": 0.12,
        "dinner": 0.38
    }

    meal_rate = np.array([rates[m] for m in meals])

    consumption = (
        students * meal_rate
        + students * special * 0.025
        + rng.normal(0, 2.0, n)
    )
    consumption = np.maximum(consumption, 0)

    dataset = pd.DataFrame({
        "students": students,
        "meal": meals,
        "special_pakvan": special,
        "consumed_food_kg": consumption.round(2)
    })

    # Save the dataset in the project folder
    dataset.to_csv(BASE_DIR / "mess_food_dataset.csv", index=False)

    X = pd.DataFrame({
        "students": students,
        "meal_rate": meal_rate,
        "special_pakvan": special
    })

    model = LinearRegression()
    model.fit(X, consumption)

    # Save the newly trained demo model
    with open(BASE_DIR / "food_consumption_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, dataset, rates

model, dataset, rates = train_demo_model()

st.success("✅ Demo ML model trained successfully!")

st.markdown("### 📝 Enter Mess Details")

col1, col2 = st.columns(2)

with col1:
    students_present = st.number_input(
        "👨‍🎓 Students Present",
        min_value=1, max_value=1000, value=200
    )

    prepared_food = st.number_input(
        "🍚 Prepared Food (kg)",
        min_value=0.1, max_value=10000.0, value=100.0
    )

with col2:
    meal = st.selectbox(
        "🍱 Select Meal",
        ["breakfast", "lunch", "snacks", "dinner"]
    )

    special_pakvan = st.selectbox(
        "🥘 Special Pakvan Available?",
        ["yes", "no"]
    )

if st.button("🔮 Predict Food Consumption", use_container_width=True):

    input_data = pd.DataFrame([{
        "students": students_present,
        "meal_rate": rates[meal],
        "special_pakvan": int(special_pakvan == "yes")
    }])

    raw_prediction = float(model.predict(input_data)[0])

    # Consumption cannot exceed prepared food
    predicted_consumption = max(
        0.0, min(raw_prediction, prepared_food)
    )

    estimated_waste = prepared_food - predicted_consumption
    waste_percentage = estimated_waste / prepared_food * 100

    st.markdown("---")
    st.header("📊 Prediction Results")

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "🍽️ Predicted Consumption",
            f"{predicted_consumption:.2f} kg"
        )

    with b:
        st.metric(
            "♻️ Estimated Food Waste",
            f"{estimated_waste:.2f} kg"
        )

    with c:
        st.metric(
            "📉 Waste Percentage",
            f"{waste_percentage:.2f}%"
        )

    st.markdown("### 💡 Smart Recommendation")

    if waste_percentage >= 25:
        st.warning(
            "Consider preparing less food next time. "
            "Review actual consumption before changing quantities."
        )
    elif waste_percentage >= 10:
        st.info("Monitor food quantities to reduce avoidable waste.")
    else:
        st.success("Prepared food is close to the model's estimated requirement.")

    st.caption(
        "Demo prediction only. The dataset contains simulated values; "
        "real-world accuracy requires actual mess consumption records."
    )

st.markdown("---")
st.subheader("📁 Training Dataset")

st.write(
    f"Dataset size: {len(dataset)} simulated records. "
    "Use this CSV to inspect the training data."
)

st.download_button(
    "⬇️ Download Dataset CSV",
    data=dataset.to_csv(index=False).encode("utf-8"),
    file_name="mess_food_dataset.csv",
    mime="text/csv"
)