from flask import Flask, render_template, request, jsonify
import sqlite3
from pathlib import Path
import joblib
import numpy as np
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "rural_health.db"
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

health_model = None
water_model = None

def load_models():
    global health_model, water_model
    h = MODEL_DIR / "health_model.joblib"
    w = MODEL_DIR / "water_model.joblib"
    if h.exists():
        health_model = joblib.load(h)
    if w.exists():
        water_model = joblib.load(w)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS health_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        temperature REAL,
        heart_rate REAL,
        systolic_bp REAL,
        diastolic_bp REAL,
        spo2 REAL,
        glucose REAL,
        risk TEXT,
        probability REAL,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS water_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT,
        ph REAL,
        turbidity REAL,
        tds REAL,
        temperature REAL,
        conductivity REAL,
        dissolved_oxygen REAL,
        quality TEXT,
        probability REAL,
        created_at TEXT
    );
    """)
    conn.commit()
    conn.close()

def ensure_models():
    if health_model is None or water_model is None:
        load_models()

def classify_health(payload):
    ensure_models()
    features = np.array([[
        payload["age"], payload["temperature"], payload["heart_rate"],
        payload["systolic_bp"], payload["diastolic_bp"],
        payload["spo2"], payload["glucose"]
    ]])
    pred = int(health_model.predict(features)[0])
    proba = float(max(health_model.predict_proba(features)[0]))
    labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return labels[pred], proba

def classify_water(payload):
    ensure_models()
    features = np.array([[
        payload["ph"], payload["turbidity"], payload["tds"],
        payload["temperature"], payload["conductivity"],
        payload["dissolved_oxygen"]
    ]])
    pred = int(water_model.predict(features)[0])
    proba = float(max(water_model.predict_proba(features)[0]))
    labels = {0: "SAFE", 1: "MODERATE", 2: "UNSAFE"}
    return labels[pred], proba

@app.route("/")
def dashboard():
    conn = get_db()
    health_count = conn.execute("SELECT COUNT(*) c FROM health_records").fetchone()["c"]
    high_health = conn.execute("SELECT COUNT(*) c FROM health_records WHERE risk='HIGH'").fetchone()["c"]
    water_count = conn.execute("SELECT COUNT(*) c FROM water_records").fetchone()["c"]
    unsafe_water = conn.execute("SELECT COUNT(*) c FROM water_records WHERE quality='UNSAFE'").fetchone()["c"]
    recent_health = conn.execute("SELECT * FROM health_records ORDER BY id DESC LIMIT 5").fetchall()
    recent_water = conn.execute("SELECT * FROM water_records ORDER BY id DESC LIMIT 5").fetchall()
    conn.close()
    return render_template("dashboard.html",
                           health_count=health_count, high_health=high_health,
                           water_count=water_count, unsafe_water=unsafe_water,
                           recent_health=recent_health, recent_water=recent_water)

@app.route("/health", methods=["GET", "POST"])
def health():
    result = None
    if request.method == "POST":
        try:
            p = {
                "name": request.form["name"],
                "age": int(request.form["age"]),
                "temperature": float(request.form["temperature"]),
                "heart_rate": float(request.form["heart_rate"]),
                "systolic_bp": float(request.form["systolic_bp"]),
                "diastolic_bp": float(request.form["diastolic_bp"]),
                "spo2": float(request.form["spo2"]),
                "glucose": float(request.form["glucose"])
            }
            risk, probability = classify_health(p)
            conn = get_db()
            conn.execute("""INSERT INTO health_records
                (name, age, temperature, heart_rate, systolic_bp, diastolic_bp, spo2, glucose, risk, probability, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p["name"], p["age"], p["temperature"], p["heart_rate"], p["systolic_bp"],
                 p["diastolic_bp"], p["spo2"], p["glucose"], risk, probability,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            result = {"risk": risk, "probability": round(probability * 100, 1)}
        except Exception as e:
            result = {"error": str(e)}
    return render_template("health.html", result=result)

@app.route("/water", methods=["GET", "POST"])
def water():
    result = None
    if request.method == "POST":
        try:
            p = {
                "location": request.form["location"],
                "ph": float(request.form["ph"]),
                "turbidity": float(request.form["turbidity"]),
                "tds": float(request.form["tds"]),
                "temperature": float(request.form["temperature"]),
                "conductivity": float(request.form["conductivity"]),
                "dissolved_oxygen": float(request.form["dissolved_oxygen"])
            }
            quality, probability = classify_water(p)
            conn = get_db()
            conn.execute("""INSERT INTO water_records
                (location, ph, turbidity, tds, temperature, conductivity, dissolved_oxygen, quality, probability, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (p["location"], p["ph"], p["turbidity"], p["tds"], p["temperature"],
                 p["conductivity"], p["dissolved_oxygen"], quality, probability,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            result = {"quality": quality, "probability": round(probability * 100, 1)}
        except Exception as e:
            result = {"error": str(e)}
    return render_template("water.html", result=result)

@app.route("/api/health", methods=["POST"])
def api_health():
    try:
        p = request.get_json(force=True)
        risk, probability = classify_health(p)
        return jsonify({"risk": risk, "confidence": round(probability, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/water", methods=["POST"])
def api_water():
    try:
        p = request.get_json(force=True)
        quality, probability = classify_water(p)
        return jsonify({"quality": quality, "confidence": round(probability, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/history")
def history():
    conn = get_db()
    health_records = conn.execute("SELECT * FROM health_records ORDER BY id DESC LIMIT 50").fetchall()
    water_records = conn.execute("SELECT * FROM water_records ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return render_template("history.html", health_records=health_records, water_records=water_records)

if __name__ == "__main__":
    init_db()
    load_models()
    app.run(debug=True)
