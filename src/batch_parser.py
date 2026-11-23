import awpy
import polars as pl
from pathlib import Path
from src.features import extract_round_economy_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DEMOS_DIR = PROJECT_ROOT / "data" / "raw_demos"
PROCESSED_DEMOS_DIR = PROJECT_ROOT / "data" / "processed_demos"

def process_all_demos(raw_dir: Path, output_dir: Path) -> pl.DataFrame:
    # Finds all .dem files on the folder
    demo_files = list(raw_dir.glob("*.dem"))

    if not demo_files:
        raise FileNotFoundError(f"No .dem files found in {raw_dir.resolve()}")

    print(f"Found {len(demo_files)} .dem files in  {raw_dir.resolve()}")

    all_round_data = []

    for demo_path in demo_files:
        print(f"Parsing demo: {demo_path.resolve()}")
        try:
            dem = awpy.Demo(str(demo_path))
            df = extract_round_economy_features(dem)

            # Creates a column with the match_id
            df = df.with_columns(
                pl.lit(demo_path.stem).alias("match_id")
            )

            all_round_data.append(df)

        except Exception as e:
            print(f" [Skipped] Failed to parse {demo_path.name}: {e}")

    # Transforms the list of data frames in a single data frame
    combined_df = pl.concat(all_round_data)

    # Save output to .parquet file
    output_dir.mkdir(parents=True, exist_ok = True)
    save_path = output_dir / "economy_dataset.parquet"
    combined_df.write_parquet(save_path)
    print(f"\nSuccessfully processed {len(all_round_data)} out of {len(demo_files)} matches.")
    print(f"Total dataset shape: {combined_df.shape}")
    print(f"Saved dataset on: {save_path.resolve()}")

    return combined_df

if __name__ == "__main__":
    process_all_demos(RAW_DEMOS_DIR, PROCESSED_DEMOS_DIR)

    



