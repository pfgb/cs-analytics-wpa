# Counter-Strike 2 Live-Round Win Probability and Player Performance Tracker

An esports data analytics project focused on Counter-Strike 2. The main focus of it is as a learning, hands-on project with Python to apply technique learned on undergraduate and graduate studies.

It has two main features:
* A live round-win probability tracker;
* Player performance using the WPA: Win Probability Added, or how much round win probability a player added during rounds.

The main round-win probability model was trained on a small dataset (available on the tick_dataset.parquet) of grand finals found on HLTV.

Tested models included logistical regression, small neural networks and XGBoost, which proved to be the best model on most settings.


## Project Structure:
* notebooks:
  * baseline_model: a simple model that calculates the probability based on the round-start equipment from both teams;
  * in_round_model: calculates the P(CT round win) as kills and events change its pobability;
  * player_wpa_analysis: calculates aggregated player's WPA through a match.
* src (most important files):
  * app.py: streamlit app that calculates aggregated WPA for every player, shows every round win probability and most impactful plays (use in your matches!)
  * tick_features.py: extracts features and events from demo files using awpy library;
  * wpa_engine.py: calculates the WPA for every player

## Known Limitations:

This is primarily a learning project, so there are glaring simplifications and limitations:
* Bomb plants are tracked but bomb defusals are not;
* Every round starts with 50% CT Win probability for WPA purposes;
* The training dataset contains 12 maps, or 3 recent Grand Finals: FURIA vs BetBoom; Vitality vs MOUZ and Vitality vs Aurora. A bigger dataset would return a more accurate model (if this surprisingly becomes a useful model, I will improve this).

Though simplistic, lots of useful things were learned during this project from data processing (including the polars library which I had never used before and awpy) to a hands-on Machine Learning application (my past adventures here were on research problems with Deep Learning models like N-BEATS). Even Git is a newer thing for me, so the overall objective of being educational was achieved.

# How to Use:
1) Clone the repository and install the requirements;
2) Launch app.py using Streamlit;
3) Save your matches on a folder called: data/my_demos (or change the location at app.py, line 48)
