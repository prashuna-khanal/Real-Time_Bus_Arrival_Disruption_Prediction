import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Transit Pulse — Bus Delay Prediction",
    page_icon="🚏",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root{
    --bg:#0A1628;
    --panel:#101C33;
    --panel-alt:#0D1A2E;
    --border:#1E2C47;
    --text:#E8ECF1;
    --text-muted:#7C8AA3;
    --amber:#F5A623;
    --green:#3ADB76;
    --red:#FF5C5C;
    --blue:#4FA3F7;
}

.stApp{ background:var(--bg); }
.block-container{ padding-top:1.5rem; padding-bottom:2rem; max-width:1200px; }

h1,h2,h3{
    font-family:'Space Grotesk', sans-serif !important;
    color:var(--text) !important;
    letter-spacing:-0.01em;
}
h1{ font-weight:700 !important; }
h2,h3{ font-weight:600 !important; }

p, span, label, li, div{ color:var(--text); }

/* eyebrow / caption text */
.stCaption, [data-testid="stCaptionContainer"]{
    color:var(--text-muted) !important;
    letter-spacing:0.03em;
}

/* sidebar */
[data-testid="stSidebar"]{
    background:var(--panel-alt);
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] h1{
    font-size:1.3rem !important;
    border-bottom:1px solid var(--border);
    padding-bottom:0.9rem;
    margin-bottom:0.6rem;
}
[data-testid="stSidebar"] label{
    color:var(--text-muted) !important;
    font-size:0.72rem !important;
    text-transform:uppercase;
    letter-spacing:0.08em;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:6px;
    padding:8px 12px;
    margin-bottom:6px;
    text-transform:none;
    letter-spacing:0;
    font-size:0.92rem !important;
    color:var(--text) !important;
}

/* departure-board style metric tiles */
div[data-testid="stMetric"]{
    background:var(--panel);
    border:1px solid var(--border);
    border-left:3px solid var(--amber);
    border-radius:6px;
    padding:14px 18px;
}
div[data-testid="stMetricLabel"]{
    color:var(--text-muted) !important;
    text-transform:uppercase;
    letter-spacing:0.09em;
    font-size:0.7rem !important;
}
div[data-testid="stMetricValue"]{
    font-family:'IBM Plex Mono', monospace !important;
    color:var(--amber) !important;
    font-size:1.7rem !important;
    font-weight:600 !important;
}

/* buttons */
.stButton button{
    background:var(--amber);
    color:#211404;
    border:none;
    border-radius:6px;
    font-family:'Space Grotesk', sans-serif;
    font-weight:600;
    letter-spacing:0.02em;
    padding:0.6rem 1rem;
}
.stButton button:hover{
    background:#FFB84D;
    color:#211404;
}

/* selects / sliders */
[data-baseweb="select"] > div{
    background:var(--panel) !important;
    border-color:var(--border) !important;
}
.stSlider [data-baseweb="slider"]{ color:var(--amber); }

/* dataframes */
[data-testid="stDataFrame"]{
    border:1px solid var(--border);
    border-radius:6px;
    overflow:hidden;
}

/* alert boxes */
div[data-testid="stAlert"]{
    border-radius:6px;
    font-family:'Space Grotesk', sans-serif;
}

/* divider */
hr{ border-color:var(--border) !important; }

/* eyebrow tag used on page headers */
.tp-eyebrow{
    font-family:'IBM Plex Mono', monospace;
    font-size:0.72rem;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:var(--amber);
    margin-bottom:0.2rem;
}

/* risk pill */
.tp-pill{
    display:inline-block;
    font-family:'Space Grotesk', sans-serif;
    font-weight:600;
    font-size:0.95rem;
    padding:6px 16px;
    border-radius:999px;
}
.tp-pill-low{ background:rgba(58,219,118,0.15); color:var(--green); border:1px solid var(--green); }
.tp-pill-mod{ background:rgba(245,166,35,0.15); color:var(--amber); border:1px solid var(--amber); }
.tp-pill-high{ background:rgba(255,92,92,0.15); color:var(--red); border:1px solid var(--red); }

/* panel wrapper for map / sections */
.tp-panel{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:8px;
    padding:1rem;
    margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

DATA = Path("Data")
PROCESSED = DATA / "Processed"
MODELS = Path("Models")


@st.cache_data
def load_data():
    vehicle = pd.read_csv(PROCESSED / "vehicle_clean.csv")
    disruption = pd.read_csv(PROCESSED / "disruption_clean.csv")
    schedule = pd.read_parquet(PROCESSED / "bus_schedule")
    return vehicle, disruption, schedule


@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "random_forest_model.pkl")
    route_encoder = joblib.load(MODELS / "route_encoder.pkl")
    agency_encoder = joblib.load(MODELS / "agency_encoder.pkl")
    return model, route_encoder, agency_encoder


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


vehicle, disruption, schedule = load_data()
rf, route_encoder, agency_encoder = load_model()

# The model's encoders were only fit on routes/agencies present in the
# labelled training sample (nearest_stop), which is a small subset of the
# full bus_schedule. To keep predictions meaningful, we only let the user
# pick routes/agencies the model actually saw during training.
known_route_ids = set(route_encoder.classes_)
known_agency_ids = set(agency_encoder.classes_)

schedule["route_id_str"] = schedule["route_id"].astype(str)
schedule["agency_id_str"] = schedule["agency_id"].astype(str)

schedule_known = schedule[
    schedule["route_id_str"].isin(known_route_ids) &
    schedule["agency_id_str"].isin(known_agency_ids)
].copy()

st.sidebar.markdown(
    "<div style='font-family:IBM Plex Mono, monospace; font-size:0.7rem; "
    "letter-spacing:0.14em; color:#F5A623; margin-bottom:2px;'>TRANSIT PULSE</div>",
    unsafe_allow_html=True
)
st.sidebar.title("Network Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Route Explorer", "Predict Delay", "Analytics", "Model Performance"]
)

# ---------------------------------------------------------------- HOME
if page == "Home":
    st.markdown("<div class='tp-eyebrow'>LIVE NETWORK STATUS</div>", unsafe_allow_html=True)
    st.title("Bus delay prediction dashboard")
    st.caption("Real-time route analytics and delay prediction, built on GTFS, GPS and disruption data")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles", f"{len(vehicle):,}")
    c2.metric("Bus stops", f"{schedule['stop_id'].nunique():,}")
    c3.metric("Routes", f"{schedule['route_short_name'].nunique():,}")
    c4.metric("Disruptions", f"{len(disruption):,}")

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Project overview")
        st.write("""
            This dashboard predicts bus delays using machine learning.

            It combines:
            - Vehicle GPS locations
            - GTFS bus schedule
            - Route information
            - Disruption reports

            The prediction model is a Random Forest Regressor trained on engineered transport features.
        """)

    with right:
        st.subheader("Best model")
        st.metric("Random Forest R²", "0.332")
        st.metric("RMSE", "8.63")
        st.metric("MAE", "4.79")

    st.write("")
    st.subheader("Dataset summary")
    summary = pd.DataFrame({
        "Dataset": ["Vehicle", "Bus Schedule", "Disruption"],
        "Rows": [len(vehicle), len(schedule), len(disruption)],
        "Columns": [vehicle.shape[1], schedule.shape[1], disruption.shape[1]]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.success("Dashboard Ready")

# ---------------------------------------------------------------- ROUTE EXPLORER
elif page == "Route Explorer":
    st.markdown("<div class='tp-eyebrow'>NETWORK MAP</div>", unsafe_allow_html=True)
    st.title("Route explorer")
    st.write("Explore bus routes, stops and current vehicle locations.")

    st.caption(
        f"Showing {schedule['route_short_name'].nunique():,} total routes. "
        f"Only {schedule_known['route_short_name'].nunique():,} have enough "
        f"training data to support delay prediction."
    )

    col1, col2 = st.columns([3, 1])
    routes = schedule["route_short_name"].dropna().sort_values().unique()

    with col1:
        selected_route = st.selectbox("Select Route", routes, key="route_selector")
    with col2:
        direction = st.radio("Direction", ["Outbound", "Inbound"], horizontal=True, key="direction_selector")

    direction_value = 0 if direction == "Outbound" else 1

    # === FIXED ROUTE DATA PREPARATION ===
    route_df = schedule[
        (schedule["route_short_name"] == selected_route) &
        (schedule["direction_id"] == direction_value)
    ].copy()

    if route_df.empty:
        st.warning("No stops available for this route and direction.")
    else:
        # Fix 1: Use only the first trip (dramatic performance improvement)
        if not route_df.empty:
            first_trip = route_df["trip_id"].iloc[0]
            route_df = route_df[route_df["trip_id"] == first_trip]

        # Fix 2: Remove duplicate stops
        route_df = route_df.drop_duplicates(subset="stop_sequence")

        # Fix 3: Drop invalid coordinates
        route_df = route_df.dropna(subset=["stop_lat", "stop_lon"])

        # Sort properly
        route_df = route_df.sort_values("stop_sequence")

        # Safety check
        st.write(f"**Stops loaded:** {len(route_df)}")  # Should be ~25-60 now

        if len(route_df) == 0:
            st.error("No valid stops after cleaning.")
        else:
            center_lat = route_df["stop_lat"].mean()
            center_lon = route_df["stop_lon"].mean()

            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=13, 
                tiles="CartoDB Positron"
            )

            points = []

            for _, row in route_df.iterrows():
                points.append([row["stop_lat"], row["stop_lon"]])
                
                folium.CircleMarker(
                    location=[row["stop_lat"], row["stop_lon"]],
                    radius=5,
                    color="#4FA3F7",
                    fill=True,
                    fill_opacity=0.9,
                    popup=f"<b>{row['stop_name']}</b><br>Stop #{row['stop_sequence']}",
                    tooltip=row["stop_name"]
                ).add_to(m)

            # Route line
            folium.PolyLine(
                points, 
                color="#1E88E5", 
                weight=6, 
                opacity=0.85
            ).add_to(m)

            # Current vehicles on this route
            vehicle_route = vehicle[vehicle["LineRef"].astype(str) == str(selected_route)]
            if not vehicle_route.empty:
                for _, veh in vehicle_route.iterrows():
                    folium.Marker(
                        [veh["Latitude"], veh["Longitude"]],
                        tooltip=f"Vehicle • Delay: {veh.get('Delay', 'N/A')}",
                        icon=folium.Icon(color="green", icon="bus", prefix="fa")
                    ).add_to(m)

            # Better rendering
            st_folium(m, height=600, width=None, use_container_width=True)

            # Metrics
            st.write("")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stops", route_df["stop_id"].nunique())
            c2.metric("Trips (sampled)", route_df["trip_id"].nunique())
            c3.metric("Active Vehicles", len(vehicle_route))
            c4.metric("Direction", direction)

            st.subheader("Route stops")
            stops_display = route_df[["stop_sequence", "stop_name", "arrival_time", "departure_time"]].copy()
            st.dataframe(stops_display, use_container_width=True, hide_index=True)
# ---------------------------------------------------------------- PREDICT DELAY
elif page == "Predict Delay":
    st.markdown("<div class='tp-eyebrow'>DELAY FORECAST</div>", unsafe_allow_html=True)
    st.title("Predict delay")
    st.write("Estimate delay for a journey along a selected route.")
    st.info(
        "Only routes present in the model's training sample are shown here, "
        "so predictions are based on patterns the model actually learned."
    )

    routes = schedule_known["route_short_name"].dropna().sort_values().unique()

    if len(routes) == 0:
        st.error("No routes available — check that the saved encoders match the schedule data.")
        st.stop()

    selected_route = st.selectbox("Select Route", routes)

    route_df = schedule_known[schedule_known["route_short_name"] == selected_route].sort_values("stop_sequence")

    if route_df.empty:
        st.warning("No stops available for this route.")
    else:
        stop_names = route_df["stop_name"].tolist()

        c1, c2 = st.columns(2)
        with c1:
            start_stop = st.selectbox("Starting Stop", stop_names, key="start_stop")
        with c2:
            destination_stop = st.selectbox(
                "Destination Stop", stop_names, index=len(stop_names) - 1, key="destination_stop"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            travel_hour = st.slider("Hour", 0, 23, 12)
        with c2:
            travel_minute = st.slider("Minute", 0, 59, 0)
        with c3:
            traffic_density = st.slider("Traffic (proxy)", 0, 100, 50)

        start_row = route_df[route_df["stop_name"] == start_stop].iloc[0]
        end_row = route_df[route_df["stop_name"] == destination_stop].iloc[0]

        stops_remaining = abs(int(end_row["stop_sequence"]) - int(start_row["stop_sequence"]))
        distance = haversine(start_row["stop_lat"], start_row["stop_lon"],
                              end_row["stop_lat"], end_row["stop_lon"])

        st.write("")
        if st.button("Predict Delay", use_container_width=True):

            route_id = str(route_df.iloc[0]["route_id"])
            agency_id = str(route_df.iloc[0]["agency_id"])

            # These should always pass now, since schedule_known was already
            # filtered to routes/agencies the encoders were trained on. This
            # check stays as a safety net in case the underlying data changes.
            if route_id not in route_encoder.classes_ or agency_id not in agency_encoder.classes_:
                st.error(
                    "This route/agency was not part of the model's training data — "
                    "prediction unavailable. Please pick a different route."
                )
                st.stop()

            route_id_enc = route_encoder.transform([route_id])[0]
            agency_id_enc = agency_encoder.transform([agency_id])[0]

            if travel_hour < 6:
                time_of_day = 0
            elif travel_hour < 10:
                time_of_day = 1
            elif travel_hour < 16:
                time_of_day = 2
            elif travel_hour < 20:
                time_of_day = 3
            else:
                time_of_day = 4

            X = pd.DataFrame({
                "route_id_enc": [route_id_enc],
                "agency_id_enc": [agency_id_enc],
                "hour": [travel_hour],
                "minute": [travel_minute],
                "distance_km": [distance],
                "traffic_density": [traffic_density],
                "distance_per_stop": [distance / max(stops_remaining, 1)],
                "hour_distance": [travel_hour * distance],
                "stop_sequence": [stops_remaining],
                "pickup_type": [0],
                "drop_off_type": [0],
                "is_peak_hour": [int(travel_hour in [7, 8, 9, 16, 17, 18])],
                "day_of_week": [2],
                "is_weekend": [0],
                "time_of_day": [time_of_day],
            })

            # Ensure column order matches exactly what the model was trained on
            X = X[rf.feature_names_in_] if hasattr(rf, "feature_names_in_") else X

            prediction = rf.predict(X)[0]
            if prediction < 5:
                risk_label, risk_class = "Low risk", "tp-pill-low"
            elif prediction < 10:
                risk_label, risk_class = "Moderate risk", "tp-pill-mod"
            else:
                risk_label, risk_class = "High risk", "tp-pill-high"

            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted delay", f"{prediction:.1f} min")
            c2.metric("Distance", f"{distance:.2f} km")
            c3.metric("Stops remaining", stops_remaining)

            st.write("")
            st.markdown(f"<span class='tp-pill {risk_class}'>{risk_label}</span>", unsafe_allow_html=True)

            st.write("")
            st.subheader("Current disruption notices")
            open_disruptions = disruption[disruption["Progress"].str.lower() == "open"][
                ["Summary", "MiscellaneousReason", "Planned"]
            ].head(5)
            st.dataframe(open_disruptions, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- ANALYTICS
elif page == "Analytics":
    st.markdown("<div class='tp-eyebrow'>NETWORK INSIGHTS</div>", unsafe_allow_html=True)
    st.title("Analytics")

    st.subheader("Top 10 routes by scheduled trips")
    top_routes = (
        schedule.groupby("route_short_name")["trip_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(top_routes, color="#F5A623")

    st.subheader("Disruption causes")
    cause_counts = disruption["MiscellaneousReason"].value_counts()
    st.bar_chart(cause_counts, color="#4FA3F7")

    st.subheader("Planned vs unplanned disruptions")
    planned_counts = disruption["Planned"].value_counts()
    st.bar_chart(planned_counts, color="#3ADB76")

# ---------------------------------------------------------------- MODEL PERFORMANCE
elif page == "Model Performance":
    st.markdown("<div class='tp-eyebrow'>MODEL EVALUATION</div>", unsafe_allow_html=True)
    st.title("Model performance")

    results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosted Trees"],
        "RMSE": [9.330, 8.632, 8.918],
        "MAE": [5.786, 4.792, 4.957],
        "R2": [0.220, 0.332, 0.287]
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.subheader("Best model: Random Forest")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", "0.332")
    c2.metric("RMSE", "8.63")
    c3.metric("MAE", "4.79")

    st.info(
        "Random Forest outperformed Linear Regression and Gradient Boosted Trees, "
        "capturing non-linear relationships between time-of-day, distance and delay."
    )