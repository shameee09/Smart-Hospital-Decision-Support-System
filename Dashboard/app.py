import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Hospital Decision Support System",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# MODEL PATHS - LOCAL + CLOUD DEPLOYMENT
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DIABETIC_MODEL_PATH = (
    BASE_DIR
    / "Final models"
    / "diabetic"
    / "diabetic_readmission_model.pkl"
)

PNEUMONIA_MODEL_PATH = (
    BASE_DIR
    / "Final models"
    / "Pneumoia"
    / "Pneumonia_model.pkl"
)

LOS_MODEL_PATH = (
    BASE_DIR
    / "Final models"
    / "Length of stay"
    / "lengthofstay_model.pkl"
)

HOSPITAL_MODEL_PATH = (
    BASE_DIR
    / "Final models"
    / "Hospital Analysis"
    / "hospital_model.pkl"
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    diabetic = joblib.load(DIABETIC_MODEL_PATH)
    pneumonia = joblib.load(PNEUMONIA_MODEL_PATH)
    los = joblib.load(LOS_MODEL_PATH)
    hospital = joblib.load(HOSPITAL_MODEL_PATH)

    return diabetic, pneumonia, los, hospital


try:

    (
        diabetic_model,
        pneumonia_model,
        los_model,
        hospital_model
    ) = load_models()

    system_ready = True

except Exception as e:

    system_ready = False
    st.error(f"System initialization error: {e}")


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []


if "patients" not in st.session_state:
    st.session_state.patients = []


if "patient" not in st.session_state:

    st.session_state.patient = {
        "Patient ID": "",
        "Patient Name": "",
        "Age": 25,
        "Gender": "Female"
    }


# =====================================================
# SAVE PREDICTION
# =====================================================

def save_prediction(module, result):

    st.session_state.history.append({

        "Patient ID":
            st.session_state.patient.get("Patient ID", ""),

        "Patient Name":
            st.session_state.patient.get("Patient Name", ""),

        "Assessment": module,

        "Result": result,

        "Date":
            datetime.now().strftime("%d-%m-%Y %H:%M")
    })


# =====================================================
# CHECK PATIENT
# =====================================================

def patient_registered():

    if not st.session_state.patient["Patient ID"]:

        st.warning(
            "Please register a patient before starting an assessment."
        )

        return False

    return True


# =====================================================
# GENERAL CLINICAL INPUT FUNCTION
# Used for Pneumonia
# =====================================================

def clinical_inputs(model, names, prefix):

    count = model.n_features_in_

    fields = names[:count]

    while len(fields) < count:

        fields.append(
            f"Clinical Measurement {len(fields) + 1}"
        )

    values = []

    col1, col2 = st.columns(2)

    for i, name in enumerate(fields):

        container = col1 if i % 2 == 0 else col2

        with container:

            value = st.number_input(
                name,
                value=0.0,
                step=0.1,
                key=f"{prefix}_{i}"
            )

            values.append(value)

    return np.array(values).reshape(1, -1)


# =====================================================
# PATIENT READMISSION INPUTS
# =====================================================

def diabetic_inputs(model):

    count = model.n_features_in_

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
            key="diabetic_gender"
        )

        gender_value = 0 if gender == "Female" else 1

        age_group = st.selectbox(
            "Age Group",
            [
                "0-10",
                "10-20",
                "20-30",
                "30-40",
                "40-50",
                "50-60",
                "60-70",
                "70-80",
                "80-90",
                "90-100"
            ],
            key="diabetic_age"
        )

        age_map = {
            "0-10": 0,
            "10-20": 1,
            "20-30": 2,
            "30-40": 3,
            "40-50": 4,
            "50-60": 5,
            "60-70": 6,
            "70-80": 7,
            "80-90": 8,
            "90-100": 9
        }

        age_value = age_map[age_group]

        time_hospital = st.number_input(
            "Time in Hospital (Days)",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            key="diabetic_time"
        )

    with col2:

        lab_procedures = st.number_input(
            "Number of Lab Procedures",
            min_value=0,
            value=10,
            step=1,
            key="diabetic_lab"
        )

        medications = st.number_input(
            "Number of Medications",
            min_value=0,
            value=5,
            step=1,
            key="diabetic_medication"
        )

        diagnoses = st.number_input(
            "Number of Diagnoses",
            min_value=0,
            value=1,
            step=1,
            key="diabetic_diagnosis"
        )

    possible_values = [
        gender_value,
        age_value,
        time_hospital,
        lab_procedures,
        medications,
        diagnoses
    ]

    values = possible_values[:count]

    while len(values) < count:

        extra = st.number_input(
            f"Additional Clinical Measurement {len(values) + 1}",
            value=0.0,
            key=f"diabetic_extra_{len(values)}"
        )

        values.append(extra)

    return np.array(values).reshape(1, -1)


# =====================================================
# LENGTH OF STAY INPUTS
# =====================================================

def los_inputs(model):

    count = model.n_features_in_

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
        key="los_gender"
    )

    gender_value = 0 if gender == "Female" else 1

    field_names = [
        "Dialysis / Renal Condition",
        "Asthma",
        "Iron Deficiency",
        "Pneumonia History",
        "Substance Use",
        "Psychological Condition",
        "Depression",
        "Psychotherapy History",
        "Fibrosis",
        "Malnutrition",
        "Hemoglobin",
        "Hematocrit",
        "Neutrophil Count",
        "Sodium Level",
        "Glucose Level",
        "Blood Urea",
        "Creatinine",
        "BMI",
        "Pulse Rate",
        "Respiration Rate",
        "Secondary Diagnosis Count",
        "Facility Code"
    ]

    values = [gender_value]

    col1, col2 = st.columns(2)

    needed = count - 1

    for i, name in enumerate(field_names[:needed]):

        container = col1 if i % 2 == 0 else col2

        with container:

            value = st.number_input(
                name,
                value=0.0,
                step=0.1,
                key=f"los_{i}"
            )

            values.append(value)

    while len(values) < count:

        extra = st.number_input(
            f"Additional Clinical Measurement {len(values) + 1}",
            value=0.0,
            key=f"los_extra_{len(values)}"
        )

        values.append(extra)

    return np.array(values).reshape(1, -1)


# =====================================================
# PNEUMONIA FIELDS
# =====================================================

PNEUMONIA_FIELDS = [
    "Fever Status",
    "Heart Rate / Tachycardia",
    "Crackles",
    "Oxygen Saturation (%)",
    "White Blood Cell Count"
]


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🏥 Smart Hospital Support")

st.sidebar.caption(
    "Clinical Decision Support Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "👤 Patient Registration",
        "🩺 Patient Readmission",
        "🛏️ Length of Stay",
        "🫁 Pneumonia Assessment",
        "🏥 Patient Outcome Analysis",
        "📋 Assessment History",
        "📊 Hospital Analytics"
    ]
)

st.sidebar.divider()

if system_ready:
    st.sidebar.success("● System Ready")
else:
    st.sidebar.error("● System Unavailable")


# =====================================================
# HOME
# =====================================================

if page == "🏠 Home":

    st.title(
        "Smart Hospital Decision Support System"
    )

    st.write(
        "Clinical decision-support platform for doctors, "
        "nurses and hospital administrators."
    )

    st.divider()

    st.subheader("Hospital Overview")

    total_patients = len(st.session_state.patients)
    total_assessments = len(st.session_state.history)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Registered Patients",
        total_patients
    )

    c2.metric(
        "Total Assessments",
        total_assessments
    )

    c3.metric(
        "Clinical Services",
        "4"
    )

    c4.metric(
        "System Status",
        "Live" if system_ready else "Unavailable"
    )

    st.divider()

    st.subheader(
        "🩺 Clinical Decision Support Services"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.info(
            """
            ### 🩺 Patient Readmission

            Assess the likelihood of hospital readmission.
            """
        )

    with c2:

        st.info(
            """
            ### 🛏️ Length of Stay

            Estimate the expected duration of hospitalization.
            """
        )

    with c3:

        st.info(
            """
            ### 🫁 Pneumonia Assessment

            Evaluate pneumonia risk using clinical indicators.
            """
        )

    with c4:

        st.info(
            """
            ### 🏥 Patient Outcome

            Assess the expected patient recovery status.
            """
        )


# =====================================================
# PATIENT REGISTRATION
# =====================================================

elif page == "👤 Patient Registration":

    st.title("👤 Patient Registration")

    st.write(
        "Register the patient before starting "
        "a clinical assessment."
    )

    with st.form("patient_registration"):

        c1, c2 = st.columns(2)

        with c1:

            patient_id = st.text_input(
                "Patient ID",
                placeholder="Example: P001"
            )

            patient_name = st.text_input(
                "Patient Name"
            )

        with c2:

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=25
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Other"]
            )

        submit = st.form_submit_button(
            "Save Patient Record",
            type="primary"
        )

        if submit:

            if not patient_id:

                st.error("Please enter Patient ID.")

            elif not patient_name:

                st.error("Please enter Patient Name.")

            else:

                patient_record = {
                    "Patient ID": patient_id,
                    "Patient Name": patient_name,
                    "Age": age,
                    "Gender": gender
                }

                st.session_state.patient = patient_record

                existing_ids = [
                    patient["Patient ID"]
                    for patient in st.session_state.patients
                ]

                if patient_id not in existing_ids:

                    st.session_state.patients.append(
                        patient_record
                    )

                    st.success(
                        "Patient registered successfully."
                    )

                else:

                    st.info(
                        "Existing patient selected successfully."
                    )

    if st.session_state.patients:

        st.subheader("Registered Patients")

        patients_df = pd.DataFrame(
            st.session_state.patients
        )

        st.dataframe(
            patients_df,
            use_container_width=True
        )


# =====================================================
# PATIENT READMISSION
# FIXED: SAME NAME AS SIDEBAR
# =====================================================

elif page == "🩺 Patient Readmission":

    st.title(
        "🩺 Patient Readmission Assessment"
    )

    st.write(
        "Enter the patient's clinical information "
        "to assess readmission risk."
    )

    if patient_registered() and system_ready:

        st.info(
            f"Current Patient: "
            f"{st.session_state.patient['Patient Name']} "
            f"({st.session_state.patient['Patient ID']})"
        )

        user_input = diabetic_inputs(
            diabetic_model
        )

        if st.button(
            "Assess Readmission Risk",
            type="primary"
        ):

            try:

                prediction = diabetic_model.predict(
                    user_input
                )[0]

                results = {
                    0: "No Readmission Expected",
                    1: "Readmission Within 30 Days",
                    2: "Readmission After 30 Days"
                }

                result = results.get(
                    int(prediction),
                    "Readmission Risk Identified"
                )

                if int(prediction) == 0:

                    st.success(
                        f"### ✅ {result}"
                    )

                else:

                    st.warning(
                        f"### ⚠️ {result}"
                    )

                save_prediction(
                    "Patient Readmission",
                    result
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =====================================================
# LENGTH OF STAY
# =====================================================

elif page == "🛏️ Length of Stay":

    st.title(
        "🛏️ Hospital Length of Stay Estimation"
    )

    st.write(
        "Enter clinical information to estimate "
        "the expected duration of hospitalization."
    )

    if patient_registered() and system_ready:

        st.info(
            f"Current Patient: "
            f"{st.session_state.patient['Patient Name']} "
            f"({st.session_state.patient['Patient ID']})"
        )

        user_input = los_inputs(
            los_model
        )

        if st.button(
            "Estimate Length of Stay",
            type="primary"
        ):

            try:

                prediction = los_model.predict(
                    user_input
                )[0]

                days = max(
                    0,
                    round(float(prediction), 1)
                )

                if days <= 3:
                    category = "Short Stay"

                elif days <= 7:
                    category = "Moderate Stay"

                else:
                    category = "Extended Stay"

                st.success(
                    f"### 🛏️ Estimated Hospital Stay: "
                    f"{days} Days"
                )

                st.info(
                    f"Care Planning Category: "
                    f"**{category}**"
                )

                save_prediction(
                    "Length of Stay",
                    f"{days} Days — {category}"
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =====================================================
# PNEUMONIA ASSESSMENT
# =====================================================

elif page == "🫁 Pneumonia Assessment":

    st.title(
        "🫁 Pneumonia Clinical Assessment"
    )

    st.write(
        "Enter respiratory and clinical indicators "
        "to evaluate pneumonia risk."
    )

    if patient_registered() and system_ready:

        st.info(
            f"Current Patient: "
            f"{st.session_state.patient['Patient Name']} "
            f"({st.session_state.patient['Patient ID']})"
        )

        user_input = clinical_inputs(
            pneumonia_model,
            PNEUMONIA_FIELDS.copy(),
            "pneumonia"
        )

        if st.button(
            "Assess Pneumonia Risk",
            type="primary"
        ):

            try:

                prediction = pneumonia_model.predict(
                    user_input
                )[0]
                
                
                results = {
                    0: "Normal",
                    1: "Pneumonia Detected",
                    2: "Clinical Review Required"
                }

                result = results.get(
                    int(prediction),
                    "Clinical Review Required"
                )

                if int(prediction) == 0:

                    st.success(
                        f"### ✅ Assessment Result: {result}"
                    )

                elif int(prediction) == 1:

                    st.error(
                        f"### ⚠️ Assessment Result: {result}"
                    )

                else:

                    st.warning(
                        f"### ⚠️ Assessment Result: {result}"
                    )

                save_prediction(
                    "Pneumonia Assessment",
                    result
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =====================================================
# PATIENT OUTCOME ANALYSIS
# FIXED CATEGORICAL INPUTS
# =====================================================

elif page == "🏥 Patient Outcome Analysis":

    st.title("🏥 Patient Outcome Analysis")

    st.write(
        "Enter patient and treatment information "
        "to assess the expected recovery outcome."
    )

    if patient_registered() and system_ready:

        st.info(
            f"Current Patient: "
            f"{st.session_state.patient['Patient Name']} "
            f"({st.session_state.patient['Patient ID']})"
        )

        col1, col2 = st.columns(2)

        # -----------------------------
        # COLUMN 1
        # -----------------------------

        with col1:

            age = st.number_input(
                "Patient Age",
                min_value=0,
                max_value=120,
                value=int(
                    st.session_state.patient.get(
                        "Age", 25
                    )
                ),
                key="hospital_age"
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                key="hospital_gender"
            )

            condition = st.selectbox(
                "Medical Condition",
                [
                    "Heart Disease",
                    "Diabetes",
                    "Fracture",
                    "Stroke",
                    "Cancer",
                    "Hypertension",
                    "Appendicitis",
                    "Heart Attack",
                    "Allergic Reaction",
                    "Respiratory Infection",
                    "Prostate Condition",
                    "Childbirth"
                ],
                key="hospital_condition"
            )

            procedure = st.selectbox(
                "Medical Procedure",
                [
                    "Angioplasty",
                    "Insulin Therapy",
                    "X-Ray and Casting",
                    "CT Scan and Treatment",
                    "Surgery and Chemotherapy",
                    "Medication",
                    "Appendectomy",
                    "Cardiac Care",
                    "Epinephrine",
                    "Antibiotics",
                    "Radiation Therapy",
                    "Delivery and Postnatal Care"
                ],
                key="hospital_procedure"
            )

        # -----------------------------
        # COLUMN 2
        # -----------------------------

        with col2:

            cost = st.number_input(
                "Treatment Cost",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                key="hospital_cost"
            )

            length_of_stay = st.number_input(
                "Length of Stay (Days)",
                min_value=1,
                max_value=365,
                value=3,
                step=1,
                key="hospital_stay"
            )

            readmission = st.selectbox(
                "Readmission",
                ["No", "Yes"],
                key="hospital_readmission"
            )

            satisfaction = st.slider(
                "Patient Satisfaction Score",
                min_value=1,
                max_value=5,
                value=3,
                key="hospital_satisfaction"
            )

        # =================================================
        # INTERNAL ENCODING
        # User sees words; model receives numbers
        # =================================================

        gender_options = [
            "Female",
            "Male"
        ]

        condition_options = [
            "Heart Disease",
            "Diabetes",
            "Fracture",
            "Stroke",
            "Cancer",
            "Hypertension",
            "Appendicitis",
            "Heart Attack",
            "Allergic Reaction",
            "Respiratory Infection",
            "Prostate Condition",
            "Childbirth"
        ]

        procedure_options = [
            "Angioplasty",
            "Insulin Therapy",
            "X-Ray and Casting",
            "CT Scan and Treatment",
            "Surgery and Chemotherapy",
            "Medication",
            "Appendectomy",
            "Cardiac Care",
            "Epinephrine",
            "Antibiotics",
            "Radiation Therapy",
            "Delivery and Postnatal Care"
        ]

        gender_value = gender_options.index(
            gender
        )

        condition_value = condition_options.index(
            condition
        )

        procedure_value = procedure_options.index(
            procedure
        )

        readmission_value = (
            0 if readmission == "No" else 1
        )

        # All dataset input columns except Outcome target
        all_hospital_values = [
            age,
            gender_value,
            condition_value,
            procedure_value,
            cost,
            length_of_stay,
            readmission_value,
            satisfaction
        ]

        required_count = hospital_model.n_features_in_

        hospital_values = all_hospital_values[
            :required_count
        ]

        while len(hospital_values) < required_count:

            hospital_values.append(0)

        user_input = np.array(
            hospital_values
        ).reshape(1, -1)

        if st.button(
            "Analyze Patient Outcome",
            type="primary"
        ):

            try:

                prediction = hospital_model.predict(
                    user_input
                )[0]

                outcomes = {
                    0: "Recovered",
                    1: "Stable",
                    2: "Requires Continued Monitoring"
                }

                result = outcomes.get(
                    int(prediction),
                    "Requires Clinical Review"
                )

                if result == "Recovered":

                    st.success(
                        f"### ✅ Expected Outcome: {result}"
                    )

                elif result == "Stable":

                    st.info(
                        f"### 🩺 Expected Outcome: {result}"
                    )

                else:

                    st.warning(
                        f"### ⚠️ Expected Outcome: {result}"
                    )

                save_prediction(
                    "Patient Outcome Analysis",
                    result
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =====================================================
# ASSESSMENT HISTORY
# =====================================================

elif page == "📋 Assessment History":

    st.title("📋 Patient Assessment History")

    if not st.session_state.history:

        st.info(
            "No clinical assessments have been recorded yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button(
            "Clear Assessment History"
        ):

            st.session_state.history = []

            st.rerun()


# =====================================================
# HOSPITAL ANALYTICS
# =====================================================

elif page == "📊 Hospital Analytics":

    st.title("📊 Hospital Analytics")

    total_patients = len(
        st.session_state.patients
    )

    total_assessments = len(
        st.session_state.history
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Registered Patients",
        total_patients
    )

    c2.metric(
        "Total Assessments",
        total_assessments
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        services_used = (
            history_df["Assessment"].nunique()
        )

    else:

        services_used = 0

    c3.metric(
        "Clinical Services Used",
        services_used
    )

    if not st.session_state.history:

        st.info(
            "Complete patient assessments "
            "to view detailed analytics."
        )

    else:

        st.subheader(
            "Clinical Assessments by Service"
        )

        st.bar_chart(
            history_df[
                "Assessment"
            ].value_counts()
        )

        st.subheader(
            "Recent Clinical Assessments"
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Smart Hospital Decision Support System | "
    "Assisted Healthcare Analytics"
)
