"""
Streamlit Dashboard for Diabetes Digital Twin.

Provides:
- Real-time CGM visualization
- AI chat interface
- Prediction and explanation displays
- What-if simulation tools
- Patient statistics
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import httpx

# Page configuration
st.set_page_config(
    page_title="Diabetes Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ok { background: #d4edda; color: #155724; }
    .status-warn { background: #fff3cd; color: #856404; }
    .status-danger { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8080/api/v1")


# ==================== Helper Functions ====================


def api_request(method: str, endpoint: str, **kwargs):
    """Make API request with error handling."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.request(method, f"{API_BASE_URL}{endpoint}", **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        st.error(f"API Error: {e}")
        return None


def get_cgm_data(patient_id: int, hours: int = 24) -> pd.DataFrame:
    """Fetch real CGM data from the API or fallback to demo."""
    data = api_request("GET", f"/patients/{patient_id}/cgm", params={"period_hours": hours})
    
    if data and data.get("time") and data.get("glucose"):
        df = pd.DataFrame({
            "time": pd.to_datetime(data["time"]),
            "glucose": data["glucose"]
        })
        # Validate that we have proper data length; optionally fallback if it's too short
        if not df.empty:
            return df
            
    # Fallback if API fails or returns no data
    return generate_demo_data(hours)

def generate_demo_data(hours: int = 24) -> pd.DataFrame:
    """Generate demo CGM data for visualization."""
    np.random.seed(42)

    now = datetime.now()
    timestamps = [now - timedelta(hours=hours) + timedelta(minutes=5*i)
                  for i in range(hours * 12)]

    # Generate realistic glucose pattern
    base_glucose = 120
    glucose_values = []

    for i, ts in enumerate(timestamps):
        hour = ts.hour

        # Dawn phenomenon
        dawn_effect = 20 * np.exp(-((hour - 6) ** 2) / 8) if 4 <= hour <= 9 else 0

        # Meal effects
        breakfast_effect = 40 * np.exp(-((hour - 8) ** 2) / 2) if 7 <= hour <= 11 else 0
        lunch_effect = 35 * np.exp(-((hour - 13) ** 2) / 2) if 12 <= hour <= 16 else 0
        dinner_effect = 45 * np.exp(-((hour - 19) ** 2) / 2) if 18 <= hour <= 22 else 0

        # Random variation
        noise = np.random.normal(0, 8)

        glucose = base_glucose + dawn_effect + breakfast_effect + lunch_effect + dinner_effect + noise
        glucose = np.clip(glucose, 50, 300)
        glucose_values.append(glucose)

    return pd.DataFrame({
        "time": timestamps,
        "glucose": glucose_values,
    })


def create_glucose_chart(df: pd.DataFrame) -> go.Figure:
    """Create interactive glucose chart with target ranges."""
    fig = go.Figure()

    # Target range bands
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1,
                  annotation_text="Target Range", annotation_position="top left")
    fig.add_hrect(y0=0, y1=70, fillcolor="red", opacity=0.1,
                  annotation_text="Low", annotation_position="bottom left")
    fig.add_hrect(y0=180, y1=400, fillcolor="yellow", opacity=0.1,
                  annotation_text="High", annotation_position="top left")

    # Glucose line
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["glucose"],
        mode="lines",
        name="Glucose",
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.1)",
    ))

    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=180, line_dash="dash", line_color="orange", opacity=0.5)

    fig.update_layout(
        title="Continuous Glucose Monitor",
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        yaxis=dict(range=[40, 350]),
        height=400,
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def create_tir_chart(in_range: float, below: float, above: float) -> go.Figure:
    """Create Time in Range donut chart."""
    fig = go.Figure(data=[go.Pie(
        values=[in_range, below, above],
        labels=["In Range (70-180)", "Below Range (<70)", "Above Range (>180)"],
        hole=0.6,
        marker_colors=["#2ecc71", "#e74c3c", "#f39c12"],
        textinfo="percent+label",
    )])

    fig.update_layout(
        title="Time in Range",
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    return fig


def create_prediction_chart(current: float, predictions: dict) -> go.Figure:
    """Create glucose prediction visualization."""
    times = [0] + [int(k.replace("min", "")) for k in predictions.keys()]
    values = [current] + list(predictions.values())

    fig = go.Figure()

    # Prediction line
    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#9b59b6", width=3, dash="dash"),
        marker=dict(size=10),
    ))

    # Target range
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1)

    fig.update_layout(
        title="Glucose Prediction",
        xaxis_title="Minutes from now",
        yaxis_title="Glucose (mg/dL)",
        height=300,
        yaxis=dict(range=[40, 300]),
    )

    return fig


# ==================== Sidebar ====================


def render_sidebar():
    """Render sidebar with patient selection and settings."""
    st.sidebar.title("🩺 Diabetes Digital Twin")
    st.sidebar.markdown("---")

    # Patient selection
    st.sidebar.subheader("Patient")
    patient_id = st.sidebar.selectbox(
        "Select Patient",
        options=[1, 2, 3],
        format_func=lambda x: f"Patient {x}",
    )
    st.session_state["patient_id"] = patient_id

    # Time range
    st.sidebar.subheader("Display Settings")
    time_range = st.sidebar.selectbox(
        "Time Range",
        options=["6 hours", "12 hours", "24 hours", "3 days", "7 days"],
        index=2,
    )
    st.session_state["time_range"] = time_range

    def get_hours(tr):
        if "day" in tr:
            return int(tr.split()[0]) * 24
        return int(tr.split()[0])

    hours = get_hours(st.session_state.get("time_range", "24 hours"))
    
    # Quick stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Stats")

    # Fetch real stats from API
    stats = api_request("GET", f"/patients/{patient_id}/stats", params={"period_hours": hours})
    df = get_cgm_data(patient_id, hours)

    current_glucose = df["glucose"].iloc[-1] if not df.empty else 120
    avg_glucose = stats.get("average_glucose", df["glucose"].mean() if not df.empty else 120) if stats else df["glucose"].mean()

    # Status indicator
    if current_glucose < 70:
        status = "🔴 LOW"
        status_color = "red"
    elif current_glucose > 180:
        status = "🟡 HIGH"
        status_color = "orange"
    else:
        status = "🟢 IN RANGE"
        status_color = "green"

    st.sidebar.metric("Current Glucose", f"{current_glucose:.0f} mg/dL", status)
    st.sidebar.metric(f"{hours}h Average", f"{avg_glucose:.0f} mg/dL")

    # Time in range
    in_range = stats.get("time_in_range", np.mean((df["glucose"] >= 70) & (df["glucose"] <= 180)) * 100 if not df.empty else 100) if stats else np.mean((df["glucose"] >= 70) & (df["glucose"] <= 180)) * 100
    st.sidebar.metric("Time in Range", f"{in_range:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.info("💡 Ask the AI assistant any questions about your glucose management!")


# ==================== Main Content ====================


def render_overview_tab():
    """Render main overview dashboard."""
    patient_id = st.session_state.get("patient_id", 1)
    
    def get_hours(tr):
        if "day" in tr:
            return int(tr.split()[0]) * 24
        return int(tr.split()[0])

    hours = get_hours(st.session_state.get("time_range", "24 hours"))
    
    col1, col2 = st.columns([2, 1])

    with col1:
        # CGM Chart
        df = get_cgm_data(patient_id, hours)
        fig = create_glucose_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Time in Range
        stats = api_request("GET", f"/patients/{patient_id}/stats", params={"period_hours": hours})
        if stats:
            in_range = stats.get("time_in_range", 100)
            below = stats.get("time_below_range", 0)
            above = stats.get("time_above_range", 0)
        else:
            in_range = np.mean((df["glucose"] >= 70) & (df["glucose"] <= 180)) * 100 if not df.empty else 100
            below = np.mean(df["glucose"] < 70) * 100 if not df.empty else 0
            above = np.mean(df["glucose"] > 180) * 100 if not df.empty else 0

        tir_fig = create_tir_chart(in_range, below, above)
        st.plotly_chart(tir_fig, use_container_width=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not df.empty and len(df) > 12:
            change_1h = df['glucose'].iloc[-1] - df['glucose'].iloc[-12]
        else:
            change_1h = 0
        avg = stats.get("average_glucose", df['glucose'].mean() if not df.empty else 0) if stats else df['glucose'].mean()
        st.metric(
            "Average Glucose",
            f"{avg:.0f} mg/dL",
            f"{change_1h:.0f} from 1h ago"
        )

    with col2:
        cv = stats.get("coefficient_of_variation", (df['glucose'].std() / df['glucose'].mean() * 100) if not df.empty else 0) if stats else (df['glucose'].std() / df['glucose'].mean() * 100)
        st.metric("Glucose Variability (CV)", f"{cv:.1f}%")

    with col3:
        hypo_events = stats.get("hypo_events", np.sum(df["glucose"] < 70) if not df.empty else 0) if stats else np.sum(df["glucose"] < 70)
        st.metric(f"Hypo Events ({hours}h)", hypo_events)

    with col4:
        hyper_events = stats.get("hyper_events", np.sum(df["glucose"] > 250) if not df.empty else 0) if stats else np.sum(df["glucose"] > 250)
        st.metric(f"Hyper Events ({hours}h)", hyper_events)


def render_prediction_tab():
    """Render prediction and explanation tab."""
    patient_id = st.session_state.get("patient_id", 1)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Glucose Predictions")

        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Analyzing glucose patterns..."):
                pred_data = api_request("POST", "/predict", json={"patient_id": patient_id, "horizon_minutes": 120})
                
                if pred_data and "predictions" in pred_data:
                    current = pred_data.get("current_glucose", 145)
                    predictions = pred_data["predictions"]
                else:
                    # Demo prediction fallback
                    current = 145
                    predictions = {
                        "30min": 152,
                        "60min": 165,
                        "90min": 158,
                        "120min": 148,
                    }

                fig = create_prediction_chart(current, predictions)
                st.plotly_chart(fig, use_container_width=True)

                # Show predictions
                st.markdown("### Predicted Values")
                for time, value in predictions.items():
                    emoji = "🟢" if 70 <= value <= 180 else "🟡" if value > 180 else "🔴"
                    st.write(f"{emoji} **{time}**: {value:.0f} mg/dL")
                
                if pred_data and "risk_level" in pred_data:
                    st.markdown(f"**Risk Level**: {pred_data['risk_level'].upper()}")

                st.session_state["last_prediction"] = {"current": current, "predictions": predictions}

    with col2:
        st.subheader("🔍 Explanation")
        
        if st.button("Explain Latest Prediction"):
            if "last_prediction" not in st.session_state:
                st.warning("Please generate a prediction first.")
            else:
                with st.spinner("Generating explanation..."):
                    exp_data = api_request("POST", "/explain", json={"patient_id": patient_id, "horizon_minutes": 60})
                    
                    if exp_data and "explanation_text" in exp_data:
                        st.markdown(exp_data["explanation_text"])
                    else:
                        st.markdown("""
                        ### Why is glucose predicted to rise?
                
                        **Top Contributing Factors:**
                
                        1. **Recent Carbohydrate Intake** (weight: 45%)
                           - 55g carbs consumed 20 minutes ago
                           - Peak glucose effect expected in ~45 minutes
                
                        2. **Current Glucose Trend** (weight: 30%)
                           - Rising at 2.3 mg/dL per minute
                           - Indicates active meal absorption
                
                        3. **Insulin on Board** (weight: 15%)
                           - 3.2 units active
                           - Will start lowering glucose in ~30 minutes
                
                        4. **Time of Day** (weight: 10%)
                           - Post-breakfast window
                           - Higher insulin resistance
                
                        ---
                
                        **Recommendation:**
                        Your prediction shows glucose will peak around 165 mg/dL in 60 minutes, then return to target. No immediate action needed.
                        """)


def render_simulation_tab():
    """Render what-if simulation tab."""
    st.subheader("🧪 What-If Simulation")
    patient_id = st.session_state.get("patient_id", 1)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Simulate Scenario")

        carbs = st.slider("Carbohydrates (g)", 0, 150, 50)
        insulin = st.slider("Insulin (units)", 0.0, 20.0, 5.0, 0.5)
        exercise = st.slider("Exercise (minutes)", 0, 120, 0)
        intensity = st.selectbox("Exercise Intensity", ["light", "moderate", "vigorous"])

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                sim_data = api_request("POST", "/simulate", json={
                    "patient_id": patient_id,
                    "carbs_grams": float(carbs),
                    "insulin_units": float(insulin),
                    "exercise_minutes": exercise,
                    "exercise_intensity": intensity
                })
                st.session_state["simulation_result"] = sim_data
            st.session_state["simulation_run"] = True

    with col2:
        if st.session_state.get("simulation_run"):
            sim_data = st.session_state.get("simulation_result")
            
            if sim_data and "simulated_trajectory" in sim_data:
                # Use real simulation data
                trajectory = sim_data["simulated_trajectory"]
                times = [p["time"] for p in trajectory]
                glucose = [p["glucose"] for p in trajectory]
                
                peak = sim_data.get("peak_glucose", max(glucose))
                nadir = min(glucose)
                time_to_baseline = sim_data.get("time_to_baseline_minutes", times[-1])
                recommendations = sim_data.get("recommendations", [])
            else:
                # Fallback to demo simulation
                current = 120
                times = list(range(0, 181, 15))
                glucose = []
    
                for t in times:
                    g = current
    
                    # Meal effect
                    if carbs > 0:
                        meal_effect = carbs * 2.5 * np.exp(-((t - 60) ** 2) / (2 * 30 ** 2))
                        g += meal_effect
    
                    # Insulin effect
                    if insulin > 0:
                        insulin_effect = insulin * 40 * (1 - np.exp(-t / 30)) * np.exp(-(t - 90) / 120)
                        g -= insulin_effect
    
                    # Exercise effect
                    if exercise > 0 and t <= exercise + 60:
                        ex_factor = {"light": 0.3, "moderate": 0.5, "vigorous": 0.8}[intensity]
                        g -= ex_factor * min(t, exercise)
    
                    glucose.append(max(40, min(350, g)))
                
                peak = max(glucose)
                nadir = min(glucose)
                time_to_baseline = times[-1]
                recommendations = []

            fig = go.Figure()

            fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1)
            fig.add_trace(go.Scatter(
                x=times, y=glucose,
                mode="lines+markers",
                name="Simulated",
                line=dict(color="#e74c3c", width=3),
            ))

            fig.update_layout(
                title=f"Simulation: {carbs}g carbs, {insulin}u insulin, {exercise}min exercise",
                xaxis_title="Minutes",
                yaxis_title="Glucose (mg/dL)",
                yaxis=dict(range=[40, 300]),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            ### Simulation Results

            - **Peak Glucose:** {peak:.0f} mg/dL
            - **Lowest Glucose:** {nadir:.0f} mg/dL
            - **Time to return to baseline:** ~{time_to_baseline} min

            {"⚠️ Warning: Peak glucose exceeds target range. Consider adjusting insulin timing." if peak > 180 else "✅ Glucose stays within target range."}
            {"🔴 Caution: Risk of hypoglycemia. Consider reducing insulin or adding carbs." if nadir < 70 else ""}
            """)
            
            if recommendations:
                st.markdown("### Recommendations")
                for rec in recommendations:
                    st.markdown(f"- {rec}")


def render_chat_tab():
    """Render AI chat interface."""
    st.subheader("💬 AI Assistant")
    patient_id = st.session_state.get("patient_id", 1)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your diabetes management assistant. I can help you understand your glucose patterns, predict future levels, and provide personalized recommendations. What would you like to know?"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your glucose, meals, insulin, or anything else..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_data = api_request("POST", "/chat", json={
                    "patient_id": patient_id,
                    "message": prompt,
                    "include_context": True
                })
                
                if chat_data and "response" in chat_data:
                    response = chat_data["response"]
                    if chat_data.get("suggested_actions"):
                        response += "\n\n**Suggested Actions:**\n"
                        for action in chat_data["suggested_actions"]:
                            response += f"- {action}\n"
                else:
                    # Demo responses fallback based on keywords
                    response = generate_demo_response(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


def generate_demo_response(prompt: str) -> str:
    """Generate demo response for chat."""
    prompt_lower = prompt.lower()

    if "prediction" in prompt_lower or "predict" in prompt_lower or "will" in prompt_lower:
        return """Based on your current patterns, here's my prediction:

**Current Glucose:** 145 mg/dL

**Predictions:**
- In 30 minutes: ~152 mg/dL 🟢
- In 60 minutes: ~165 mg/dL 🟢
- In 90 minutes: ~158 mg/dL 🟢
- In 120 minutes: ~148 mg/dL 🟢

Your glucose is trending slightly upward, likely due to your recent meal. It should peak around 165 mg/dL in about an hour and then return to baseline. This is a normal post-meal pattern.

**Key Factors:**
1. Recent carb intake (55g)
2. Active insulin working to lower glucose
3. Morning time period

Would you like me to explain any of these factors in more detail?"""

    elif "meal" in prompt_lower or "eat" in prompt_lower or "carb" in prompt_lower:
        return """I can help you understand meal impacts! Let me simulate what would happen:

**If you eat 50g of carbohydrates now:**

With your current insulin-to-carb ratio of 1:10, you would need approximately **5 units** of rapid-acting insulin.

**Predicted response:**
- Peak glucose: ~185 mg/dL (at 60 minutes)
- Return to baseline: ~150 minutes

**Recommendation:**
- Pre-bolus 15-20 minutes before eating to reduce the spike
- If it's a high-fat meal, consider splitting the bolus

Would you like me to simulate a specific meal scenario?"""

    elif "exercise" in prompt_lower or "workout" in prompt_lower:
        return """Exercise can significantly affect your glucose! Here's what to know:

**Before Exercise:**
- If glucose < 90 mg/dL: Have 15-20g carbs first
- If glucose > 250 mg/dL: Check ketones before exercising

**During Exercise:**
- Monitor glucose every 30 minutes
- Keep fast-acting carbs available

**For a 30-minute moderate workout, I recommend:**
- Reduce bolus by 30-50% for meals within 2 hours
- Consider reducing basal rate during/after (if on pump)
- Have 15g carbs available in case of lows

Your current glucose is 145 mg/dL - this is a good level to start exercising! Just monitor for drops during and after.

Shall I set up monitoring reminders for your workout?"""

    elif "low" in prompt_lower or "hypo" in prompt_lower:
        return """⚠️ **Hypoglycemia Management**

If your glucose drops below 70 mg/dL, follow the **Rule of 15:**

1. **Treat:** Consume 15-20g of fast-acting carbs:
   - 4 glucose tablets
   - 4 oz juice or regular soda
   - 1 tablespoon honey

2. **Wait:** 15 minutes for glucose to rise

3. **Recheck:** Test glucose again

4. **Repeat** if still below 70 mg/dL

**Warning Signs to Watch For:**
- Shakiness, sweating, hunger
- Confusion, irritability
- Fast heartbeat

If you can't treat yourself or symptoms are severe, this is an emergency - someone should administer glucagon and call for help.

Are you currently experiencing low glucose symptoms?"""

    else:
        return f"""I understand you're asking about: "{prompt}"

Based on your recent data, here's what I can share:

- **Current Glucose:** 145 mg/dL (in target range ✅)
- **24-hour Average:** 152 mg/dL
- **Time in Range:** 72% (good!)

I can help you with:
- 📈 Glucose predictions ("What will my glucose be in 1 hour?")
- 🍽️ Meal planning ("What if I eat 60g carbs?")
- 💪 Exercise guidance ("How should I prepare for a workout?")
- 📊 Pattern analysis ("Why is my glucose high in the morning?")

What specific aspect would you like to explore?"""


# ==================== Main App ====================


def main():
    """Main application entry point."""
    render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Predictions", "🧪 Simulation", "💬 Chat"])

    with tab1:
        render_overview_tab()

    with tab2:
        render_prediction_tab()

    with tab3:
        render_simulation_tab()

    with tab4:
        render_chat_tab()


if __name__ == "__main__":
    main()
