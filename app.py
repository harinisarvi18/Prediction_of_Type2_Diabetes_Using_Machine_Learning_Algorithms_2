from flask import Flask, render_template, request, send_file
import joblib
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import datetime

app = Flask(__name__)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join("outputs", "diabetes_model.pkl")

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def landing():
    return render_template("home.html")


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    probability = None
    risk_level = None
    form_data = {}

    if request.method == "POST":

        form_data = request.form

        # -------------------------
        # GET INPUT DATA
        # -------------------------

        age = float(form_data["age"])
        gender = int(form_data["gender"])
        pulse_rate = float(form_data["pulse_rate"])
        systolic_bp = float(form_data["systolic_bp"])
        diastolic_bp = float(form_data["diastolic_bp"])
        glucose = float(form_data["glucose"])
        height = float(form_data["height"])
        weight = float(form_data["weight"])
        bmi = float(form_data["bmi"])
        family_diabetes = int(form_data["family_diabetes"])
        hypertensive = int(form_data["hypertensive"])
        family_hypertension = int(form_data["family_hypertension"])
        cardiovascular_disease = int(form_data["cardiovascular_disease"])
        stroke = int(form_data["stroke"])

        # -------------------------
        # FEATURE ENGINEERING
        # -------------------------

        pulse_pressure = systolic_bp - diastolic_bp
        bmi_age_ratio = bmi / age
        glucose_bmi = glucose * bmi

        # -------------------------
        # FINAL FEATURE VECTOR
        # -------------------------

        features = [
            age,
            gender,
            pulse_rate,
            systolic_bp,
            diastolic_bp,
            glucose,
            height,
            weight,
            bmi,
            family_diabetes,
            hypertensive,
            family_hypertension,
            cardiovascular_disease,
            stroke,
            pulse_pressure,
            bmi_age_ratio,
            glucose_bmi
        ]

        # -------------------------
        # MODEL PREDICTION
        # -------------------------

        pred = model.predict([features])[0]

        prediction = "Diabetic" if pred == 1 else "Non-Diabetic"

        if hasattr(model, "predict_proba"):

            probability = round(model.predict_proba([features])[0][1] * 100, 2)

            # Risk classification

            if probability < 30:
                risk_level = "Low Risk"
            elif probability < 70:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"

        return render_template(
            "results.html",
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            bmi=bmi,
            bmi_category=bmi_category(bmi),
            form_data=form_data
        )

    return render_template("index.html", form_data=form_data)


# --------------------------------------------------
# BMI CATEGORY
# --------------------------------------------------

def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

@app.route("/about")
def about():
    return render_template("about.html")


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

@app.route("/model-performance")
def model_performance():
    return render_template("model_performance.html")


# --------------------------------------------------
# METHODOLOGY PAGE
# --------------------------------------------------

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


# --------------------------------------------------
# DOWNLOAD PDF REPORT
# --------------------------------------------------

@app.route("/download_report", methods=["POST"])
def download_report():

    data = request.form

    filename = "Diabetes_Risk_Report.pdf"

    file_path = os.path.join("outputs", filename)

    c = canvas.Canvas(file_path, pagesize=A4)

    width, height = A4

    y = height - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, "Type-2 Diabetes Risk Prediction Report")

    y -= 30

    c.setFont("Helvetica", 10)

    c.drawCentredString(
        width / 2,
        y,
        f"Generated on: {datetime.datetime.now()}"
    )

    y -= 40

    c.setFont("Helvetica-Bold", 12)

    c.drawString(50, y, "Prediction Result")

    y -= 20

    c.setFont("Helvetica", 11)

    c.drawString(50, y, f"Prediction: {data.get('prediction')}")

    c.drawString(50, y - 18, f"Risk Probability: {data.get('probability')}%")

    c.drawString(50, y - 36, f"BMI: {data.get('bmi')} ({data.get('bmi_category')})")

    c.showPage()

    c.save()

    return send_file(file_path, as_attachment=True)


# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)