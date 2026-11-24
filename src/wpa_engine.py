import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import polars as pl
import numpy as np
import awpy
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.tick_features import extract_in_round_events

def calculate_match_wpa(
    dem: awpy.Demo, model: XGBClassifier, scaler: StandardScaler = None
) -> pl.DataFrame:
    """
    Computes Win Probability Added (WPA) for every kill and every bomb plant in a match (defusals not implemented)
    """

    # Extracts the match demo
    df = extract_in_round_events(dem)
    # Relevant features
    features = ["ct_alive", "t_alive", "ct_equip_value", "t_equip_value", "equip_value_ratio", "seconds_elapsed", "is_bomb_planted", "seconds_since_plant"]
    # Standardizes the features if it is needed
    X = df.select(features).to_numpy()
    if scaler is not None:
        X = scaler.transform(X)

    # Predicted Win Probability
    ct_probs = model.predict_proba(X)[:, 1]
    df = df.with_columns(pl.Series("ct_win_prob", ct_probs))

    df = df.sort("tick")

    # Previous win probability (round start is 0.5 by default (not realistic change this later))
    df = df.with_columns(
        pl.col("ct_win_prob")
        .shift(1)
        .over("round_num")
        .fill_null(0.50)
        .alias("prev_ct_win_prob")
    ).with_columns(
        (pl.col("ct_win_prob") - pl.col("prev_ct_win_prob")).alias("delta_p")
    )

    # Get kill events
    kills_df = dem.kills
    rounds_df = dem.rounds
    kills_df = kills_df.sort("tick")
    kills_df = kills_df.join_asof(
        rounds_df.select("start_tick", "end_tick", "round_num", "freeze_end_tick"),
        left_on="tick",
        right_on="start_tick",
        strategy="backward"
    )
    kills_df.drop("start_tick", "end_tick", "freeze_end_tick")

    kills_df = kills_df.select([
        pl.col("round_num"),
        pl.col("tick"),
        pl.col("attacker_name").alias("actor_name"),
        pl.col("attacker_side").alias("actor_side"),
        pl.col("victim_name"),
        pl.col("victim_side")
    ])

    # Get plant events (no defusals)
    bomb_df = dem.bomb
    bomb_df = bomb_df.sort("tick")
    bomb_df = bomb_df.join_asof(
        rounds_df.select("start_tick", "round_num"),
        left_on="tick",
        right_on="start_tick",
        strategy="backward"
    )
    bomb_df.drop("start_tick")

    plant_events = pl.DataFrame()
    if hasattr(dem, "bomb") and dem.bomb is not None:
        plant_events = bomb_df.filter(
            pl.col("event") == "finish_plant"
        ).select([
            pl.col("round_num"),
            pl.col("tick"),
            pl.col("name").alias("actor_name"),
            pl.lit("terrorist").alias("actor_side"),
            pl.lit(None).alias("victim_name"),
            pl.lit(None).alias("victim_side")
        ])


    # Adding attacker and victim handles and bomb plants
    wpa_events = df.join(kills_df, on=["round_num", "tick"], how="left")
    wpa_events = wpa_events.join(plant_events, on=["round_num", "tick"], how="left")
    wpa_events = wpa_events.sort("tick")

    wpa_events = wpa_events.with_columns([
        # Actor WPA (positive change)
        pl.when(pl.col("actor_side") == "counter-terrorist")
        .then(pl.col("delta_p"))
        .otherwise(- 1.0 * pl.col("delta_p"))
        .alias("actor_wpa"),

        #Victim WPA
        pl.when(pl.col("victim_side") == "counter-terrorist")
        .then(pl.col("delta_p"))
        .when(pl.col("victim_side") == "terrorist")
        .then(-1.0 * pl.col("delta_p"))
        .otherwise(0.0)
        .alias("victim_wpa")
    ])

    return wpa_events

def aggregate_player_wpa(wpa_events: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregates the tables returned from the previous functions
    """

    n_rounds = wpa_events["round_num"].max()

    actor_stats = (
        wpa_events.filter(pl.col("actor_name").is_not_null())
        .group_by("actor_name")
        .agg([
            pl.col("actor_wpa").sum().alias("positive_wpa"),
            pl.len().alias("actions")
        ])
        .rename({"actor_name": "player_name"})
    )

    victim_stats = (
        wpa_events.filter(pl.col("victim_name").is_not_null())
        .group_by("victim_name")
        .agg([
            pl.col("victim_wpa").sum().alias("negative_wpa"),
            pl.len().alias("actions")
        ])
        .rename({"victim_name": "player_name"})
    )    

    # Combine both WPA
    leaderboard = (
        actor_stats.join(victim_stats, on="player_name", how="full")
        .fill_null(0.0)
        .with_columns(
            ((pl.col("positive_wpa") + pl.col("negative_wpa")) / n_rounds).alias("average_wpa")
        )
        .sort("average_wpa", descending=True)
    )

    return leaderboard


