from pathlib import Path
from awpy import Demo

# Path(__file__) is the location of THIS file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the .dem file
DEMO_PATH = PROJECT_ROOT / "data" / "raw_demos" /  "vitality-vs-mouz-m1-cache.dem"


def inspect_demo(demo_path: Path): # : Path is a type hint for demo_path
    if not demo_path.exists():
        print(f"Error: Could not find demo at {demo_path.resolve()}")
        return

    print(f"Loading and parsing demo: {demo_path.name} ...")

    # Initialize and parse the CS2 demo
    demo = Demo(str(demo_path))

    # awpy provides event DataFrame via Polars
    print("\n--- Map & Header Info ---")
    print(f"Map: {demo.header.get('map_name')}")
    print(f"Tickrate: {demo.header.get('tick_rate')}")

    print("\n--- Available Datasets ---")

    # Kills
    if hasattr(demo, "kills") and demo.kills is not None:
        print(f"Total Kill Events: {len(demo.kills)}")
        print("\nSample Kills:")
        print(demo.kills.head(3))

    # Rounds
    if hasattr(demo, "rounds") and demo.rounds is not None:
        print(f"\nTotal Rounds Played: {len(demo.rounds)}")
        print("\nSample Rounds:")
        print(demo.rounds.head(3))

    if hasattr(demo, "round_economy") and demo.round_economy is not None:
        print(f"\nSample Round Economy Table: ")
        print(demo.round_economy.head(3))

if __name__ ==  "__main__":
    inspect_demo(DEMO_PATH)