# Intelligent Traffic Route Planner

A Streamlit dashboard that demonstrates classic path‑finding algorithms (BFS, Uniform Cost Search, A*) on a simulated road network of Chennai.  The network, traffic levels and road closures are synthetic and intended for educational purposes only.

## Features
- Interactive map visualisation with traffic colour‑coding.
- Choose start / destination and optimisation criterion (fastest or shortest).
- Compare three algorithms side‑by‑side, see execution time, explored nodes and alternative routes.
- Dynamically adjust traffic levels and close/open roads.
- Fuel consumption estimate based on vehicle mileage.
- Ready to run locally or deploy to **Streamlit Community Cloud**.

## Repository structure
```
├── app.py               # Streamlit entry‑point (runs `streamlit run app.py`)
├── traffic_planner.py   # Core implementation of the road network and algorithms
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Setup & local development
1. **Clone the repository** (or copy the files into a new folder):
   ```bash
   git clone https://github.com/ragulraj-k/Traffic_Route_Planner
   cd <repo‑folder>
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```
   The app will open in your default browser (usually at `http://localhost:8501`).

## Deploy to Streamlit Community Cloud
1. Push the repository to a GitHub (or GitLab/Bitbucket) account.
2. In the Streamlit Community Cloud UI click **New app**, select the repo and the `app.py` file as the entry‑point.
3. The service automatically installs the packages from `requirements.txt` and launches the app.
4. You can share the generated URL with anyone.

## Usage tips
- Use the **Sidebar** to select start/end locations, algorithm and optimisation criterion.
- Click **Find optimal route** to compute a path.
- Adjust traffic levels or close roads in the **Traffic Management** tab to see the route recompute in real‑time.
- The **Algorithm Comparison** tab shows aggregated statistics for 200 runs of each algorithm.

## Contributing
Feel free to open issues or submit pull requests.  Typical contributions might include:
- Adding more realistic traffic data or a different city.
- Extending the UI with more visualisations.
- Refactoring for performance or testability.

