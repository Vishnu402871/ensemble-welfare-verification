import streamlit as st

st.title("Ensemble Welfare Verification System")

st.write(
    "Verify whether a citizen is eligible for a government welfare scheme."
)

# User Inputs
name = st.text_input("Citizen Name")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=25
)

income = st.number_input(
    "Annual Income (INR)",
    min_value=0,
    value=100000
)

state = st.text_input("State")

scheme = st.text_input("Scheme Name")

# Verification Button
if st.button("Verify Eligibility"):

    score = 0
    reasons = []

    # Age Rule
    if age >= 18:
        score += 1
        reasons.append("✓ Age requirement satisfied")
    else:
        reasons.append("✗ Applicant must be at least 18 years old")

    # Income Rule
    if income <= 200000:
        score += 1
        reasons.append("✓ Income within eligibility limit")
    else:
        reasons.append("✗ Income exceeds eligibility limit")

    # State Rule
    if state.strip() != "":
        score += 1
        reasons.append("✓ State information provided")
    else:
        reasons.append("✗ State information missing")

    confidence = (score / 3) * 100

    st.subheader("Eligibility Result")

    if score >= 2:
        st.success("Eligible")
    else:
        st.error("Not Eligible")

    st.write(f"Confidence Score: {confidence:.2f}%")

    st.subheader("Reasons")

    for r in reasons:
        st.write(r)