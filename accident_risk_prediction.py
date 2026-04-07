"""
accident_risk_prediction.py
============================
Zone-Based AI Accident Risk Prediction System

Predicts the probability of an accident occurring in a specific zone
based on real-world user inputs.  Uses a hybrid approach:

  1. Weighted rule-based baseline scoring  →  probability %
  2. Risk classification (Low / Medium / High / Critical)
  3. AI interpretation via Anthropic Claude  →  reasoning + recommendations

Designed for future integration with:
  - Real accident datasets (CSV / database)
  - ML models   (Random Forest, XGBoost, Logistic Regression)
  - GIS / map layers
  - Zone clustering  (DBSCAN / K-Means)
  - Ambulance placement optimisation  (LSCP / MCLP / p-median)

Usage:
  python accident_risk_prediction.py

Required packages:
  pip install anthropic python-dotenv

Environment variables (via .env or OS):
  ANTHROPIC_API_KEY   – Anthropic API key (required for AI layer)

Author : Thesis Project
Version: 2.0.0
"""

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import os
import sys
import time
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Environment variable loading ────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠  python-dotenv not installed (pip install python-dotenv).")
    print("   Falling back to OS environment variables only.\n")

# ── Gemini SDK ───────────────────────────────────────────────────
GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import errors
    GENAI_AVAILABLE = True
except ImportError:
    print("⚠  google-genai SDK not installed (pip install google-genai).")
    print("   AI interpretation layer will be unavailable.\n")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Gemini model fallback chain — tries each in order until one works
GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-pro-exp-02-05", # Fallback to 2.0 version
    "gemini-2.0-flash",
]

# Risk-level thresholds (probability %)
RISK_THRESHOLDS = {
    "Critical": 75,
    "High":     50,
    "Medium":   25,
    "Low":       0,
}

# ── Feature weight map ──────────────────────────────────────────────
WEIGHTS = {
    # Time & calendar
    "time_of_day":          8,
    "weekend":              4,
    "season_month":         3,
    "lighting_condition":   5,

    # Weather & environment
    "weather":             10,
    "visibility_level":     7,
    "rain_intensity":       6,
    "road_surface_condition": 8,

    # Traffic & road
    "traffic_density":      9,
    "road_type":            7,
    "speed_factor":         6,
    "traffic_control_presence": 4,
    "number_of_lanes":      4,
    "sharp_turn_or_blind_curve": 4,
    "road_construction_present": 4,

    # Historical
    "historical_accident_count": 6,
    "severity_trend":       4,
    "hotspot":              5,

    # Festival / event / crowd
    "is_festival_day":      4,
    "is_public_holiday":    3,
    "crowd_level":          5,
    "special_traffic_diversion": 3,
    "night_event":          4,

    # Emergency response
    "hospital_distance":    2,
    "ambulance_response_time_min": 3,
    "emergency_support":    1,
}

# Maximum score the baseline engine can produce (used for normalisation)
MAX_THEORETICAL_SCORE = float(sum(WEIGHTS.values()))


# ═══════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _hr():
    """Print a horizontal rule."""
    print("─" * 72)


def _section(title: str):
    """Print a section header."""
    print()
    _hr()
    print(f"  {title}")
    _hr()


def _prompt(label: str, default: str = "", choices: Optional[List[str]] = None) -> str:
    """Prompt for input with optional default and choice list."""
    suffix = ""
    if choices:
        suffix = f"  [{' / '.join(choices)}]"
    if default:
        suffix += f"  (default: {default})"
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        if not raw and default:
            return default.lower()
        if not raw:
            print("    ⚠ Input required.")
            continue
        val = raw.lower()
        if choices and val not in [c.lower() for c in choices]:
            print(f"    ⚠ Invalid choice. Pick from: {', '.join(choices)}")
            continue
        return val


def _prompt_int(label: str, lo: int, hi: int, default: Optional[int] = None) -> int:
    """Prompt for an integer within [lo, hi]."""
    d = f"  (default: {default})" if default is not None else ""
    while True:
        raw = input(f"  {label} [{lo}–{hi}]{d}: ").strip()
        if not raw and default is not None:
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"    ⚠ Must be between {lo} and {hi}.")
        except ValueError:
            print("    ⚠ Enter a valid integer.")


def _prompt_float(label: str, lo: float, hi: float,
                  default: Optional[float] = None) -> float:
    """Prompt for a float within [lo, hi]."""
    d = f"  (default: {default})" if default is not None else ""
    while True:
        raw = input(f"  {label} [{lo}–{hi}]{d}: ").strip()
        if not raw and default is not None:
            return default
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(f"    ⚠ Must be between {lo} and {hi}.")
        except ValueError:
            print("    ⚠ Enter a valid number.")


def _prompt_optional_float(label: str) -> Optional[float]:
    """Prompt for an optional float; returns None if skipped."""
    raw = input(f"  {label} (press Enter to skip): ").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("    ⚠ Invalid number — skipped.")
        return None


def _yn(label: str, default: str = "no") -> bool:
    """Yes/No prompt; returns True for 'yes'."""
    return _prompt(label, default=default, choices=["yes", "no"]) == "yes"


# ═══════════════════════════════════════════════════════════════════════
# AUTO-DERIVE HELPER
# ═══════════════════════════════════════════════════════════════════════

def _derive_time_of_day(hour: int) -> str:
    """Automatically derive time-of-day category from hour."""
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    elif 21 <= hour <= 23:
        return "night"
    else:  # 0–5
        return "late night"


def _derive_rain_from_weather(weather: str) -> str:
    """Auto-derive rain intensity from weather condition."""
    mapping = {
        "clear": "none", "cloudy": "none",
        "rain": "medium", "heavy rain": "heavy",
        "fog": "none", "storm": "heavy",
    }
    return mapping.get(weather, "none")





# ═══════════════════════════════════════════════════════════════════════
# 1. COLLECT USER INPUT  (STREAMLINED — 12 essential inputs)
# ═══════════════════════════════════════════════════════════════════════

def collect_user_input() -> Dict[str, Any]:
    """
    Collect only the essential inputs (~12 questions).
    Non-essential fields are auto-derived with smart defaults.
    """
    data: Dict[str, Any] = {}

    # ── Zone & Location (3 inputs) ──────────────────────────────────
    _section("ZONE & LOCATION")
    data["zone_name"] = input("  Zone name or ID (press Enter to skip): ").strip() or None
    data["latitude"]  = _prompt_float("Latitude",  -90.0,  90.0)
    data["longitude"] = _prompt_float("Longitude", -180.0, 180.0)

    # ── Time (2 inputs — everything else auto-derived) ──────────────
    _section("TIME")
    data["hour"] = _prompt_int("Current hour (0–23)", 0, 23,
                               default=datetime.now().hour)
    data["day_of_week"] = _prompt(
        "Day of week", default=datetime.now().strftime("%A").lower(),
        choices=["monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday"])

    # Auto-derive
    data["time_of_day"] = _derive_time_of_day(data["hour"])
    data["is_weekend"]  = data["day_of_week"] in ("saturday", "sunday")
    data["month"]       = datetime.now().month

    # ── Weather (1 input — rain, visibility, road surface derived) ──
    _section("WEATHER & ENVIRONMENT")
    data["weather"] = _prompt(
        "Weather condition", default="clear",
        choices=["clear", "cloudy", "rain", "heavy rain", "fog", "storm"])
    data["rain_intensity"] = _derive_rain_from_weather(data["weather"])
    data["lighting_condition"] = _prompt(
        "Lighting condition", default="daylight",
        choices=["daylight", "twilight", "dark_streetlights", "dark_no_streetlights"])
    data["visibility_level"] = _prompt(
        "Visibility level", default="good",
        choices=["good", "moderate", "poor"])
    data["road_surface_condition"] = _prompt(
        "Road surface condition", default="dry",
        choices=["dry", "wet", "muddy", "icy", "damaged"])

    # ── Traffic & Road (3 inputs) ───────────────────────────────────
    _section("TRAFFIC & ROAD")
    data["traffic_density"] = _prompt(
        "Traffic density", default="medium",
        choices=["low", "medium", "high", "very high"])
    data["road_type"] = _prompt(
        "Road type", default="urban road",
        choices=["highway", "urban road", "rural road",
                 "intersection", "market area", "residential road"])
    data["speed_limit"] = _prompt_int("Speed limit (km/h)", 10, 200, default=60)
    data["traffic_control_presence"] = _prompt(
        "Traffic control presence", default="none",
        choices=["signal", "stop_sign", "pedestrian_crossing", "none"])
    data["number_of_lanes"] = _prompt_int("Number of lanes", 1, 10, default=2)
    data["sharp_turn_or_blind_curve"] = _yn("Sharp turn or blind curve present?")
    data["road_construction_present"] = _yn("Road construction present?")
    data["avg_vehicle_speed"] = None        # uses speed_limit

    # ── Risk & History (2 inputs) ───────────────────────────────────
    _section("RISK PROFILE")
    data["severity_trend"] = _prompt(
        "Accident severity trend in this zone", default="medium",
        choices=["low", "medium", "high"])
    data["is_hotspot"] = _yn("Known accident hotspot zone?")
    
    historical_count = _prompt_int("Historical accident count (enter -1 to skip)", -1, 1000, default=-1)
    data["historical_accident_count"] = None if historical_count == -1 else historical_count

    # ── Festival / Crowd (1 input — sub-fields auto-derived) ────────
    _section("SPECIAL CONDITIONS")
    data["is_festival_day"] = _yn("Is it a festival day?")
    data["is_public_holiday"] = _yn("Is it a public holiday?")
    data["festival_name"] = None
    if data["is_festival_day"]:
        data["festival_name"] = input("  Event name (press Enter to skip): ").strip() or None

    data["crowd_level"] = _prompt(
        "Crowd level", default="low",
        choices=["low", "medium", "high", "very high"])

    data["special_traffic_diversion"] = _yn("Is there a special traffic diversion?")
    
    # Auto-derive from context
    data["night_event"] = (data["is_festival_day"]
                           and data["time_of_day"] in ("night", "late night"))

    # ── Emergency (optional — 1 input) ──────────────────────────────
    _section("EMERGENCY CONTEXT (optional)")
    data["ambulance_response_time_min"] = _prompt_optional_float(
        "Estimated ambulance response time (min)")
    data["hospital_distance_km"] = None
    data["emergency_support"]    = "available"

    return data


# ═══════════════════════════════════════════════════════════════════════
# 2. VALIDATE INPUTS
# ═══════════════════════════════════════════════════════════════════════

def validate_inputs(data: Dict[str, Any]) -> List[str]:
    """Run sanity checks on collected data. Returns warning messages."""
    warnings: List[str] = []

    if data["rain_intensity"] == "heavy" and data["weather"] == "clear":
        warnings.append("Heavy rain reported but weather is clear — "
                        "adjusting weather to 'heavy rain'.")
        data["weather"] = "heavy rain"

    if data["rain_intensity"] != "none" and data["road_surface_condition"] == "dry":
        warnings.append("Rain reported but road surface marked dry — "
                        "adjusting to 'wet'.")
        data["road_surface_condition"] = "wet"

    if (data.get("avg_vehicle_speed") is not None
            and data["avg_vehicle_speed"] > data["speed_limit"] * 1.5):
        warnings.append("Average speed significantly above speed limit.")

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# 3. PRE-PROCESS FEATURES
# ═══════════════════════════════════════════════════════════════════════

def preprocess_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Convert inputs into normalised feature scores in [0.0, 1.0]."""
    f: Dict[str, float] = {}

    # Time of day
    tod_map = {"morning": 0.3, "afternoon": 0.4, "evening": 0.6,
               "night": 0.85, "late night": 1.0}
    f["time_of_day"] = tod_map.get(data["time_of_day"], 0.5)

    # Weekend & Calendar
    f["weekend"] = 1.0 if data["is_weekend"] else 0.0
    high_risk_months = {6, 7, 8, 9, 11, 12, 1}
    f["season_month"] = 0.8 if data["month"] in high_risk_months else 0.3
    
    # Lighting
    light_map = {"daylight": 0.0, "twilight": 0.4, "dark_streetlights": 0.6, "dark_no_streetlights": 1.0}
    f["lighting_condition"] = light_map.get(data.get("lighting_condition", "daylight"), 0.3)

    # Weather & Visibility
    weather_map = {"clear": 0.0, "cloudy": 0.2, "rain": 0.5,
                   "heavy rain": 0.85, "fog": 0.9, "storm": 1.0}
    f["weather"] = weather_map.get(data["weather"], 0.3)

    vis_map = {"good": 0.0, "moderate": 0.5, "poor": 1.0}
    f["visibility_level"] = vis_map.get(data["visibility_level"], 0.3)

    rain_map = {"none": 0.0, "light": 0.3, "medium": 0.6, "heavy": 1.0}
    f["rain_intensity"] = rain_map.get(data["rain_intensity"], 0.0)

    # Road surface
    surf_map = {"dry": 0.0, "wet": 0.4, "muddy": 0.6, "icy": 1.0, "damaged": 0.8}
    f["road_surface_condition"] = surf_map.get(data["road_surface_condition"], 0.2)

    # Traffic density & Type
    td_map = {"low": 0.1, "medium": 0.4, "high": 0.75, "very high": 1.0}
    f["traffic_density"] = td_map.get(data["traffic_density"], 0.4)

    rt_map = {"residential road": 0.2, "rural road": 0.4, "urban road": 0.5,
              "highway": 0.75, "market area": 0.8, "intersection": 0.9}
    f["road_type"] = rt_map.get(data["road_type"], 0.5)

    # Speed factor
    speed = data.get("avg_vehicle_speed") or data["speed_limit"]
    if speed >= 120:   f["speed_factor"] = 1.0
    elif speed >= 80:  f["speed_factor"] = 0.7
    elif speed >= 60:  f["speed_factor"] = 0.45
    elif speed >= 40:  f["speed_factor"] = 0.25
    else:              f["speed_factor"] = 0.1

    # Traffic control & Lanes
    tc_map = {"signal": 0.0, "stop_sign": 0.3, "pedestrian_crossing": 0.4, "none": 1.0}
    f["traffic_control_presence"] = tc_map.get(data.get("traffic_control_presence", "none"), 1.0)
    
    lanes = data.get("number_of_lanes", 2)
    if lanes == 1: f["number_of_lanes"] = 0.8
    elif lanes == 2: f["number_of_lanes"] = 0.4
    else: f["number_of_lanes"] = 0.2

    # Road features
    f["sharp_turn_or_blind_curve"] = 1.0 if data.get("sharp_turn_or_blind_curve") else 0.0
    f["road_construction_present"] = 1.0 if data.get("road_construction_present") else 0.0

    # Historical frequency
    hf = data.get("historical_accident_count")
    f["historical_accident_count"] = min(hf / 50.0, 1.0) if hf else 0.3

    st_map = {"low": 0.2, "medium": 0.5, "high": 1.0}
    f["severity_trend"] = st_map.get(data["severity_trend"], 0.5)
    f["hotspot"] = 1.0 if data["is_hotspot"] else 0.0

    # Festival / event / crowd
    f["is_festival_day"]   = 1.0 if data["is_festival_day"] else 0.0
    f["is_public_holiday"] = 1.0 if data.get("is_public_holiday") else 0.0
    
    cl_map = {"low": 0.1, "medium": 0.4, "high": 0.75, "very high": 1.0}
    f["crowd_level"]       = cl_map.get(data["crowd_level"], 0.3)
    f["special_traffic_diversion"] = 1.0 if data.get("special_traffic_diversion") else 0.0
    f["night_event"]       = 1.0 if data.get("night_event") else 0.0

    # Emergency response
    hd = data.get("hospital_distance_km")
    f["hospital_distance"] = min(hd / 20.0, 1.0) if hd else 0.3
    
    art = data.get("ambulance_response_time_min")
    f["ambulance_response_time_min"] = min(art / 30.0, 1.0) if art else 0.3
    
    es_map = {"available": 0.0, "limited": 0.5, "unavailable": 1.0}
    f["emergency_support"] = es_map.get(data.get("emergency_support", "available"), 0.3)

    return f


# ═══════════════════════════════════════════════════════════════════════
# 4. CALCULATE BASELINE RISK
# ═══════════════════════════════════════════════════════════════════════

def calculate_baseline_risk(features: Dict[str, float]) -> Tuple[float, float]:
    """Compute raw risk score and probability percentage."""
    raw_score = sum(features.get(k, 0.0) * w for k, w in WEIGHTS.items())
    raw_score = max(0.0, min(raw_score, MAX_THEORETICAL_SCORE))
    probability = (raw_score / MAX_THEORETICAL_SCORE) * 100.0
    return round(raw_score, 2), round(probability, 2)


# ═══════════════════════════════════════════════════════════════════════
# 5. CLASSIFY RISK LEVEL
# ═══════════════════════════════════════════════════════════════════════

def classify_risk_level(probability: float) -> str:
    """Map probability % → risk category."""
    if probability >= RISK_THRESHOLDS["Critical"]:
        return "Critical"
    elif probability >= RISK_THRESHOLDS["High"]:
        return "High"
    elif probability >= RISK_THRESHOLDS["Medium"]:
        return "Medium"
    else:
        return "Low"


# ═══════════════════════════════════════════════════════════════════════
# 6. BUILD GEMINI PROMPT
# ═══════════════════════════════════════════════════════════════════════

def build_gemini_prompt(data: Dict[str, Any],
                        features: Dict[str, float],
                        raw_score: float,
                        probability: float,
                        risk_level: str) -> str:
    """Construct a structured prompt for Gemini."""
    zone_id = data.get("zone_name") or "Unnamed Zone"
    lat, lon = data["latitude"], data["longitude"]

    input_summary = textwrap.dedent(f"""\
    ═══  ZONE RISK ASSESSMENT DATA  ═══

    Zone:        {zone_id}
    Coordinates: ({lat}, {lon})

    ── Time & Calendar ──
    Time of Day:  {data['time_of_day']}   Hour: {data['hour']}
    Day of Week:  {data['day_of_week']}   Weekend: {'Yes' if data['is_weekend'] else 'No'}
    Month:        {data['month']}

    ── Environment & Conditions ──
    Lighting Condition: {data['lighting_condition']}
    Weather:           {data['weather']}
    Visibility Level:  {data['visibility_level']}
    Rain Intensity:    {data['rain_intensity']}
    Road Surface:      {data['road_surface_condition']}

    ── Traffic & Road ──
    Traffic Density:     {data['traffic_density']}
    Road Type:           {data['road_type']}
    Speed Limit:         {data['speed_limit']} km/h
    Traffic Control:     {data['traffic_control_presence']}
    Number of Lanes:     {data['number_of_lanes']}
    Sharp/Blind Curve:   {'Yes' if data['sharp_turn_or_blind_curve'] else 'No'}
    Road Construction:   {'Yes' if data['road_construction_present'] else 'No'}

    ── Historical ──
    Severity Trend:         {data['severity_trend']}
    Hotspot Zone:           {'Yes' if data['is_hotspot'] else 'No'}
    Historical Accidents:   {data.get('historical_accident_count') or 'N/A'}

    ── Special Conditions & Context ──
    Festival Day:       {'Yes' if data['is_festival_day'] else 'No'}
    Public Holiday:     {'Yes' if data['is_public_holiday'] else 'No'}
    Festival Name:      {data.get('festival_name') or 'N/A'}
    Crowd Level:        {data['crowd_level']}
    Traffic Diversion:  {'Yes' if data['special_traffic_diversion'] else 'No'}

    ── Emergency Response ──
    Ambulance Response:    {data.get('ambulance_response_time_min', 'N/A')} min

    ═══  BASELINE ENGINE OUTPUT  ═══
    Raw Risk Score:              {raw_score} / {MAX_THEORETICAL_SCORE}
    Predicted Probability:       {probability}%
    Risk Classification:         {risk_level}
    """)

    # Top contributing features
    contribs = sorted(
        ((k, features[k] * WEIGHTS[k]) for k in features),
        key=lambda x: x[1], reverse=True
    )[:8]
    top_factors = "\n".join(
        f"    {i+1}. {k.replace('_', ' ').title():30s}  "
        f"score contribution: {v:.2f}"
        for i, (k, v) in enumerate(contribs)
    )

    prompt = textwrap.dedent(f"""\
    You are a senior traffic-safety and emergency-response analyst AI.

    Below is a zone-based accident-risk assessment processed through
    a weighted rule-based baseline prediction engine.

    {input_summary}

    Top Contributing Risk Factors (baseline):
    {top_factors}

    ── YOUR TASK ──

    Based on all the data above, provide:

    1. **AI Risk Reasoning** (3–5 sentences):
       Explain *why* this zone has the predicted risk level.

    2. **Key Risk Factors** (bullet list of 4–6 factors):
       Most significant factors driving the risk.

    3. **Recommended Preventive & Emergency Actions** (bullet list):
       4–8 concrete, actionable recommendations.

    4. **Ambulance Deployment Insight** (2–3 sentences):
       Advise on ambulance pre-positioning for this zone.

    Format with the four numbered headings.  Be specific and actionable.
    """)

    return prompt


# ═══════════════════════════════════════════════════════════════════════
# 7. GET AI REASONING FROM GEMINI  (with model fallback chain)
# ═══════════════════════════════════════════════════════════════════════

def get_ai_reasoning_from_gemini(prompt: str) -> Optional[str]:
    """
    Send prompt to Google Gemini with automatic model fallback.
    Tries each model in GEMINI_MODELS until one succeeds.
    Returns None on total failure.
    """
    if not GENAI_AVAILABLE:
        print("  ⚠ Gemini SDK unavailable — skipping AI layer.")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ⚠ GEMINI_API_KEY not set in environment.")
        print("    Set it in a .env file or export it as an env variable.")
        return None

    client = genai.Client(api_key=api_key)

    for model in GEMINI_MODELS:
        try:
            print(f"  ⏳ Trying model: {model} …")
            t0 = time.time()

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            elapsed = time.time() - t0
            print(f"  ✅ Response from {model} in {elapsed:.1f}s")

            if response.text:
                return response.text.strip()
            return None

        except errors.APIError as e:
            if "API key not valid" in str(e) or e.code == 403: # Authentication failure
                 print(f"  ❌ Authentication failed for {model} - {e}")
                 return None
            if e.code == 404:
                 print(f"  ⚠ Model {model} not available — trying next …")
                 continue
            if e.code == 429:
                 print(f"  ❌ Rate limit reached on {model} — trying next …")
                 continue
             
            print(f"  ❌ API error on {model} (status {e.code}): {e.message}")
            continue
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return None

    print("  ❌ All models failed — using baseline fallback.")
    return None


# ═══════════════════════════════════════════════════════════════════════
# 8. GENERATE RECOMMENDATIONS (RULE-BASED)
# ═══════════════════════════════════════════════════════════════════════

def generate_recommendations(data: Dict[str, Any],
                             features: Dict[str, float],
                             probability: float,
                             risk_level: str) -> List[str]:
    """Produce context-aware recommended actions."""
    recs: List[str] = []

    # Critical-level
    if risk_level == "Critical":
        recs.append("🚨  FLAG zone as TEMPORARY HIGH-RISK immediately")
        recs.append("🚑  Deploy ambulance on standby within the zone")
        recs.append("👮  Increase police patrol and monitoring")

    # High-level
    if risk_level in ("Critical", "High"):
        recs.append("⚠️   Add temporary warning signage / LED boards")
        recs.append("📡  Activate real-time traffic monitoring")

    # Weather-specific
    if data["weather"] in ("heavy rain", "storm", "fog"):
        recs.append("🌧️   Enforce reduced speed limits due to weather")
    if data["visibility_level"] == "poor":
        recs.append("🌫️   Deploy fog/visibility warning markers")

    # Road-specific
    if data["road_surface_condition"] in ("icy", "damaged"):
        recs.append("🛣️   Urgent: Road maintenance / pothole repair")

    # Speed
    speed = data.get("avg_vehicle_speed") or data["speed_limit"]
    if speed >= 80:
        recs.append("🏎️   Deploy speed radar / enforce speed control")

    # Highway + rain
    if data["road_type"] == "highway" and data["rain_intensity"] in ("medium", "heavy"):
        recs.append("🛑  Highway + rain: stationary patrol + speed monitoring")

    # Festival / Crowd
    if data["is_festival_day"]:
        recs.append("🎉  Festival/holiday: deploy crowd & traffic control team")
    if data["crowd_level"] in ("high", "very high"):
        recs.append("👥  High crowd: set up pedestrian-safe barriers")
    if data.get("night_event"):
        recs.append("🌙  Night event: additional lighting + DUI checkpoints")

    # Market + holiday + crowd
    if (data["road_type"] == "market area" and data["is_festival_day"]
            and data["crowd_level"] in ("high", "very high")):
        recs.append("🏪  Market + holiday + crowd: restrict vehicular traffic")

    # Hotspot + poor visibility
    if data["is_hotspot"] and data["visibility_level"] == "poor":
        recs.append("📍  Hotspot + poor visibility: mandatory enforcement zone")

    # Ambulance response
    art = data.get("ambulance_response_time_min")
    if art is not None and art > 15:
        recs.append("🚑  Ambulance response > 15 min: pre-position closer unit")

    # Night + festival + crowd
    if (data["time_of_day"] in ("night", "late night")
            and data["is_festival_day"]
            and data["crowd_level"] in ("high", "very high")):
        recs.append("🚑  CRITICAL: Night + festival + high crowd → "
                     "temporary ambulance deployment ESSENTIAL")

    if not recs:
        recs.append("✅  Continue routine monitoring — no elevated actions needed")

    return recs


# ═══════════════════════════════════════════════════════════════════════
# 9. IDENTIFY KEY RISK FACTORS
# ═══════════════════════════════════════════════════════════════════════

def identify_key_risk_factors(features: Dict[str, float],
                              top_n: int = 6) -> List[Tuple[str, float]]:
    """Return top-N contributing features by weighted score."""
    contribs = [
        (k.replace("_", " ").title(), features[k] * WEIGHTS[k])
        for k in features if features[k] * WEIGHTS[k] > 0
    ]
    contribs.sort(key=lambda x: x[1], reverse=True)
    return contribs[:top_n]


# ═══════════════════════════════════════════════════════════════════════
# 10. FALLBACK EXPLANATION
# ═══════════════════════════════════════════════════════════════════════

def generate_fallback_explanation(data: Dict[str, Any],
                                  key_factors: List[Tuple[str, float]],
                                  probability: float,
                                  risk_level: str) -> str:
    """Rule-based textual explanation when Claude is unavailable."""
    zone_id = data.get("zone_name") or "the specified zone"
    factors_text = ", ".join(f[0] for f in key_factors[:4])

    explanation = (
        f"The {risk_level.lower()} risk level ({probability}%) for {zone_id} "
        f"is primarily driven by: {factors_text}. "
    )
    level_text = {
        "Critical": ("Multiple high-impact factors are active simultaneously, "
                     "creating a compounding effect. Immediate preventive "
                     "measures and emergency pre-positioning are strongly recommended."),
        "High":     ("Several risk factors are elevated. Proactive intervention "
                     "such as patrols, speed enforcement, and ambulance "
                     "pre-positioning should be considered."),
        "Medium":   ("The combination of moderate risks warrants heightened "
                     "awareness and routine monitoring of the zone."),
        "Low":      ("Current conditions are relatively safe. Standard monitoring "
                     "is sufficient, but re-evaluate if conditions change."),
    }
    explanation += level_text.get(risk_level, "")
    return explanation


# ═══════════════════════════════════════════════════════════════════════
# 11. DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════

def display_results(data: Dict[str, Any],
                    raw_score: float,
                    probability: float,
                    risk_level: str,
                    key_factors: List[Tuple[str, float]],
                    recommendations: List[str],
                    ai_response: Optional[str],
                    fallback_explanation: str):
    """Print professional CLI output."""
    emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    zone_id = data.get("zone_name") or "Unnamed Zone"

    print()
    print("═" * 72)
    print("  🚦  ACCIDENT RISK PREDICTION — RESULTS")
    print("═" * 72)

    print(f"""
  Zone:                      {zone_id}
  Coordinates:               ({data['latitude']}, {data['longitude']})
  Time:                      {data['time_of_day'].title()} (Hour {data['hour']})
  Day:                       {data['day_of_week'].title()} {'(Weekend)' if data['is_weekend'] else '(Weekday)'}
  Weather:                   {data['weather'].title()} | Visibility: {data['visibility_level'].title()}
""")
    _hr()

    print(f"""
  📊 RISK ASSESSMENT
  ─────────────────────────────────────────
  Baseline Risk Score:       {raw_score} / {MAX_THEORETICAL_SCORE}
  Predicted Probability:     {probability}%
  Risk Level:                {emoji.get(risk_level, '⚪')}  {risk_level}
""")
    _hr()

    # Key risk factors with visual bars
    print("\n  🔍 KEY RISK FACTORS (by contribution)")
    max_contrib = max((s for _, s in key_factors), default=1)
    for i, (name, score) in enumerate(key_factors, 1):
        bar_len = int(score / max(max_contrib, 0.01) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"   {i}. {name:30s}  {bar}  {score:.2f}")
    _hr()

    # AI or fallback explanation
    if ai_response:
        print("\n  🤖 AI EXPLANATION (Gemini)")
        print("  " + "─" * 40)
        for line in ai_response.split("\n"):
            print(f"  {line}")
    else:
        print("\n  📝 RISK EXPLANATION (baseline engine)")
        print("  " + "─" * 40)
        for line in textwrap.fill(fallback_explanation, width=66).split("\n"):
            print(f"  {line}")
    _hr()

    # Recommendations
    print("\n  📋 RECOMMENDED ACTIONS")
    for rec in recommendations:
        print(f"   • {rec}")
    _hr()

    # Ambulance insight (when no AI response)
    if not ai_response:
        print("\n  🚑 AMBULANCE DEPLOYMENT INSIGHT")
        art = data.get("ambulance_response_time_min")
        if risk_level in ("Critical", "High"):
            insight = (f"  Given the {risk_level.lower()} risk, pre-positioning "
                       f"an ambulance within the zone is recommended.")
            if art and art > 15:
                insight += (f"\n  Current response time ({art} min) exceeds "
                            f"target — stage a unit closer.")
            if data["crowd_level"] in ("high", "very high"):
                insight += ("\n  High crowd may impede movement — "
                            "clear a dedicated emergency corridor.")
        elif risk_level == "Medium":
            insight = ("  Ensure an ambulance is within 15-minute range. "
                       "Monitor conditions for escalation.")
        else:
            insight = ("  Standard ambulance coverage is sufficient. "
                       "No special pre-positioning needed.")
        print(insight)
        _hr()

    # Footer
    ai_label = "Gemini AI" if ai_response else "Baseline (fallback)"
    print(f"""
  ══════════════════════════════════════════════════════════════════
  Analysis generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Engine: Baseline v2.0 + {ai_label}
  ══════════════════════════════════════════════════════════════════
""")


# ═══════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Orchestrate the full prediction pipeline."""
    print()
    print("═" * 72)
    print("  🚦  ZONE-BASED AI ACCIDENT RISK PREDICTION SYSTEM  v2.0")
    print("  ──  Hybrid: Weighted Baseline + Gemini AI Layer")
    print("═" * 72)

    # Step 1: Collect inputs (~12 questions)
    try:
        data = collect_user_input()
    except (KeyboardInterrupt, EOFError):
        print("\n\n  ⚠ Input cancelled. Exiting.")
        sys.exit(0)

    # Step 2: Validate
    _section("PROCESSING")
    warnings = validate_inputs(data)
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  ✅ Inputs validated.")

    # Step 3: Pre-process
    features = preprocess_features(data)

    # Step 4: Baseline risk
    raw_score, probability = calculate_baseline_risk(features)
    print(f"  📊 Baseline risk: {raw_score} → {probability}%")

    # Step 5: Classify
    risk_level = classify_risk_level(probability)
    print(f"  🏷️  Risk level: {risk_level}")

    # Step 6: Key factors
    key_factors = identify_key_risk_factors(features)

    # Step 7: AI reasoning (with model fallback)
    _section("AI REASONING LAYER")
    prompt = build_gemini_prompt(data, features, raw_score, probability, risk_level)
    ai_response = get_ai_reasoning_from_gemini(prompt)

    # Step 8: Recommendations
    recommendations = generate_recommendations(data, features, probability, risk_level)

    # Step 9: Fallback explanation
    fallback = generate_fallback_explanation(data, key_factors, probability, risk_level)

    # Step 10: Display
    display_results(data, raw_score, probability, risk_level,
                    key_factors, recommendations, ai_response, fallback)


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⚠ Process interrupted. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌ Fatal error: {e}")
        sys.exit(1)
