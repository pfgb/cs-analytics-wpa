import awpy
import polars as pl
from pathlib import Path

def extract_in_round_events(dem: awpy.Demo) -> pl.DataFrame:
    """
    Extracts snapshots at every key event during a rounds (kills and bomb plants)
    """

    if dem.kills is None or dem.rounds is None:
        raise ValueError("Demo missing kills and rounds")

    
    kills_df = dem.kills
    rounds_df = dem.rounds
    econ_df = dem.round_economy
    bomb_df = dem.bomb
    tick_rate = dem.tick_rate if dem.tick_rate else 64.0

    kills_df = kills_df.sort("tick")
    rounds_df = rounds_df.sort("start_tick")

    round_meta = rounds_df.select([
        pl.col("round_num"),
        pl.col("freeze_end_tick"),
        (pl.col("winner_side") == "counter-terrorist").cast(pl.Int32).alias("ct_win")
    ])


    # There is no round_num indication on the kill columns, so we will do a left join with <= with rounds_df according to the tick of the kills
    kills_df = kills_df.join_asof(
        rounds_df.select("start_tick", "end_tick", "round_num", "freeze_end_tick"),
        left_on="tick",
        right_on="start_tick",
        strategy="backward"
    )
    
    kill_events = kills_df.select([
        pl.col("round_num"),
        pl.col("tick"),
        pl.lit("kill").alias("event_type"),
        (pl.col("victim_side") == "counter-terrorist").cast(pl.Int32).alias("ct_death"),
        (pl.col("victim_side") == "terrorist").cast(pl.Int32).alias("t_death")
    ])

    # adding bomb status
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
            pl.lit("bomb_planted").alias("event_type"),
            pl.lit(0).alias("ct_death"),
            pl.lit(0).alias("t_death")
        ])

    if not plant_events.is_empty():
        event_states = pl.concat([kill_events, plant_events])
    else:
        event_states = kill_events

    # event_states is the more complete end dataset from which we will select revelant columns
    # currently has the kill register, round number and ct win
    event_states = event_states.join(round_meta, on="round_num",how="inner")
    event_states = event_states.sort("tick")

    ct_econ = econ_df.filter(pl.col("side") == "counter-terrorist").select([
        pl.col("round_num"),
        pl.col("equipment_value").alias("ct_equip_value")
    ])

    t_econ = econ_df.filter(pl.col("side") == "terrorist").select([
        pl.col("round_num"),
        pl.col("equipment_value").alias("t_equip_value")
    ])

    econ_features = ct_econ.join(t_econ, on="round_num", how="inner")
    # adds economic features to event_states
    event_states = event_states.join(econ_features, on="round_num", how="inner")
    event_states = event_states.with_columns(
        (pl.col("ct_equip_value")/(pl.col("t_equip_value") + 1)).alias("equip_value_ratio")
    )


    # add number of players alive and seconds elapsed
    event_states = event_states.sort("tick")
    event_states = event_states.with_columns([
        ((pl.col("tick") - pl.col("freeze_end_tick")) / tick_rate).alias("seconds_elapsed"),
        (5 - pl.col("ct_death").cum_sum().over("round_num")).alias("ct_alive"),
        (5 - pl.col("t_death").cum_sum().over("round_num")).alias("t_alive")       
    ])



    event_states = event_states.sort("tick")

    event_states = event_states.with_columns([
        (pl.col("event_type") == "bomb_planted").cast(pl.Int32).alias("is_plant_event"),
    ])

    event_states = event_states.with_columns([
        (pl.col("is_plant_event")).cum_sum().over("round_num").alias("is_bomb_planted")
    ])

    event_states = event_states.with_columns([
        ((pl.col("tick") - pl.col("freeze_end_tick")) / tick_rate).alias("seconds_elapsed"),
    ])

    plant_ticks_df = event_states.filter(pl.col("event_type") == "bomb_planted").select([
        pl.col("round_num"),
        pl.col("tick").alias("plant_tick")
    ])

    if not plant_ticks_df.is_empty():
        event_states = event_states.join(plant_ticks_df, on="round_num", how="left")
        event_states = event_states.with_columns(
            pl.when((pl.col("is_bomb_planted") == 1) & pl.col("plant_tick").is_not_null())
            .then((pl.col("tick") - pl.col("plant_tick")) / tick_rate)
            .otherwise(0.0).alias("seconds_since_plant")
        )
    else:
        event_states =  event_states.with_columns(pl.lit(0.0).alias("seconds_since_plant"))

    dataset = event_states.select(
        pl.col("round_num"),
        pl.col("event_type"),
        pl.col("ct_alive"),
        pl.col("t_alive"),
        pl.col("ct_equip_value"),
        pl.col("t_equip_value"),
        pl.col("equip_value_ratio"),
        pl.col("tick"),
        pl.col("seconds_elapsed"),
        pl.col("is_bomb_planted"),
        pl.col("seconds_since_plant"),
        pl.col("ct_win")
    )

    return(event_states)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEMO_FILE = PROJECT_ROOT / "data" / "raw_demos" / "aurora-vs-vitality-m1-anubis.dem"

    dem = awpy.Demo(str(DEMO_FILE))
    df = extract_in_round_events(dem)

    print("-- Dynamic Ticks: Rounds 1 & 2 --")
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=-1):
        print(df.filter(pl.col("round_num") <= 4))

    