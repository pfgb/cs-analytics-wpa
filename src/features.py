import polars as pl
import awpy

def extract_round_economy_features(demo: awpy.Demo) -> pl.DataFrame:
    """
    Extracts round-level economy features and merges them with the round outcome from a .dem file.
    Returns a Polars DataFrame ready for baseline modeling.
    """

    if demo.round_economy is None or demo.rounds is None:
        raise ValueError("Demo file missing round_economy or rounds table.")

    economy_df = demo.round_economy
    rounds_df = demo.rounds

    # Pivot economy data so each round has a row with Ct and T equipment values
    ct_econ = economy_df.filter(pl.col("side") == "counter-terrorist").select([
        pl.col("round_num"),
        pl.col("equipment_value").alias("ct_equip_value"),
        pl.col("buy_type").alias("ct_buy_type")
    ])

    t_econ = economy_df.filter(pl.col("side") == "terrorist").select([
        pl.col("round_num"),
        pl.col("equipment_value").alias("t_equip_value"),
        pl.col("buy_type").alias("t_buy_type")
    ])

    # Joins the 2 dataframes above into 1 dataframe
    econ_features = ct_econ.join(t_econ, on="round_num", how="inner")

    # Select round outcomes (CT win = 1, T win = 0)
    outcomes = rounds_df.select([
        pl.col("round_num"),
        (pl.col("winner_side") == "counter-terrorist").cast(pl.Int32).alias("ct_win")
    ])
    # Join the economy features with the round result
    full_dataset = econ_features.join(outcomes, on="round_num", how="inner")

    # Feature Engineering: Equipment Value Differential and Ratio
    full_dataset = full_dataset.with_columns([
        (pl.col("ct_equip_value") - pl.col("t_equip_value")).alias("equip_value_diff"),
        ((pl.col("ct_equip_value") + 1) / (pl.col("t_equip_value") + 1)).alias("equip_value_ratio")
    ])

    return full_dataset

if __name__ == "__main__":
    from pathlib import Path

        # Path(__file__) is the location of THIS file
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Path to the .dem file
    DEMO_PATH = PROJECT_ROOT / "data" / "raw_demos" /  "vitality-vs-mouz-m1-cache.dem"

    demo = awpy.Demo(str(DEMO_PATH))
    df = extract_round_economy_features(demo)

    print("--- Extracting Features Dataset ---")
    print(df.head(5))


