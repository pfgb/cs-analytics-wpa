import sys
import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objs as go
import awpy
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from features import extract_round_economy_features
from tick_features import extract_in_round_events
from wpa_engine import calculate_match_wpa, aggregate_player_wpa

# Set page configuration
st.set_page_config(
    page_title = "CS2 Dynamic WPA Engine",
    layout="wide"
)

@st.cache_resource
def train_model():
    dataset_path = Path("data/processed_demos/tick_dataset.parquet")
    econ_dataset_path = Path("data/processed_demos/economy_dataset.parquet")
    if not dataset_path.exists():
        st.error("Processed dataset not found!")
        st.stop()

    if not econ_dataset_path.exists():
        st.error("Economy dataset not found!")
        st.stop()

    train_df = pl.read_parquet(dataset_path)
    features = ["ct_alive", "t_alive", "ct_equip_value", "t_equip_value", "equip_value_ratio", "seconds_elapsed", "is_bomb_planted", "seconds_since_plant"]
    X_train = train_df[features]
    Y_train = train_df["ct_win"]

    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)

    xgb = XGBClassifier(n_estimators=100, max_depth = 5, learning_rate = 0.1, objective="binary:logistic")
    xgb.fit(X_train, Y_train)

    return xgb

@st.cache_resource
def train_econ_model():
    econ_dataset_path = Path("data/processed_demos/economy_dataset.parquet")

    if not econ_dataset_path.exists():
        st.error("Economy dataset not found!")
        st.stop()

    econ_df = pl.read_parquet(econ_dataset_path)
    features = ["ct_equip_value", "t_equip_value", "equip_value_ratio"]
    X_econ = econ_df[features]
    Y_econ = econ_df["ct_win"]

    #econ_scaler = StandardScaler()
    #X_econ_scaled = econ_scaler.fit_transform(X_econ)

    econ_lr = LogisticRegression(random_state=42)
    econ_lr.fit(X_econ, Y_econ)

    return econ_lr

st.title("CS2 Live Round Win Probability")
st.markdown("""
Interactive page that evaluates the WPA: Win Probability Added for each player on a CS2 match with a training set of professional matches.
It also tracks the CT round-win probability as kills and bomb plants happen.
""")

st.sidebar.header("Match Selection")
raw_demos_dir = Path("data/my_demos")
demo_files = list(raw_demos_dir.glob("*.dem"))

if not demo_files:
    st.sidebar.error(f"No .dem files found in {raw_demos_dir}")

selected_demo = st.sidebar.selectbox(
    "Select Match Demo",
    options=demo_files,
    format_func = lambda p: p.name
)

if st.sidebar.button("Retrain Model"):
    st.cache_resource.clear()
    st.rerun()

model = train_model()
econ_model = train_econ_model()

with st.spinner("Processing match events and calculating WPA..."):
    dem = awpy.Demo(str(selected_demo))
    wpa_events = calculate_match_wpa(dem, model, econ_model)
    leaderboard = aggregate_player_wpa(wpa_events)

st.subheader("Player Impact: WPA, K/D")

df_leaderboard = leaderboard.to_pandas()

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(
        df_leaderboard[[
            "player_name", "average_wpa"
        ]].style.format({
            "average_wpa": "{:+.2%}"
        }).background_gradient(subset=["average_wpa"], cmap="RdYlGn"),
        width="stretch"
    )

with col2:
    st.info("""
    **Understanding Average WPA:**
    Average_WPA = (Positive_WPA + Negative_WPA) / N: Average (by rounds played) round win probability added during a match
    Positive WPA is acquired by killing enemies or planting the bomb, Negative by getting killed.
    """)

st.subheader("Round-by-Round Win Probability Trajectory")

rounds_available = sorted(wpa_events["round_num"].unique().to_list())
selected_round = st.selectbox("Select Round to Inspect:", options = rounds_available)

round_data = wpa_events.filter(pl.col("round_num") == selected_round).sort("tick").to_pandas()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=round_data["seconds_elapsed"],
    y=round_data["ct_win_prob"],
    mode="lines+markers",
    name="Ct Win Probability on Selected Round",
    line=dict(color="#1f77b4", width=3),
    marker=dict(size=8),
    hovertemplate=(
        "<b>Time:<b> %{x:.1f}s<br>" +
        "<b>CT P(Win):</b> %{y:.2%}<br>" +
        "<b>Event:</b> %{customdata[0]}<br>" +
        "<b>Actor:</b> %{customdata[1]}<br>" +
        "<extra></extra>"
    ),
    customdata=round_data[["event_type", "actor_name"]].values
))

fig.add_hline(y = 0.50, line_dash="dash", line_color="gray", annotation_text="50%")

fig.update_layout(
    title=f"Round {selected_round} Win Probability",
    xaxis_title="Round Seconds",
    yaxis_title="P(CT_win)",
    yaxis=dict(range=[0.0, 1.0]),
    template="plotly_white",
    height=450
)

st.plotly_chart(fig, width="stretch")

st.subheader("Top 5 Events (by WPA):")

top_plays = wpa_events.filter(
    pl.col("actor_name").is_not_null()
).with_columns(
    pl.col("actor_wpa").alias("wpa")
).sort("wpa", descending=True).head(5).to_pandas()

st.table(top_plays[[
    "round_num", "seconds_elapsed", "event_type",
    "actor_name", "actor_side", "wpa"
]].rename(columns={
    "round_num": "Round",
    "seconds_elapsed": "Time (s)",
    "event_type": "Event",
    "actor_name": "Player",
    "actor_side": "Side",
    "actor_wpa": "WPA"
}))


