# Emergency Response Optimization with Time-Based Zone Division

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Optimization](https://img.shields.io/badge/optimization-LSCP-success)
![Clustering](https://img.shields.io/badge/clustering-DBSCAN-orange)

## 📖 Overview
This project focuses on **Improving Post-Accident Survival Rates with Intelligent Emergency Response**. By analyzing historical geographic accident data, the system divides geographical areas into dynamic risk zones (Red, Orange, Green) based on different times of the day. It then calculates the minimum number of ambulances required and determines their optimal physical placement to ensure that emergency responders can reach any accident hotspots within a strict maximum threshold time (e.g., 15 minutes), maximizing adherence to the crucial medical "Golden Hour."

## ✨ Key Features
- **Time-Based Hotspot Clustering**: Uses auto-tuned DBSCAN machine learning to group accident coordinates into dynamic clusters based on the time of day (Morning, Afternoon, Evening, Night).
- **Dynamic Risk Zoning**: Categorizes the severity of clustered zones (🔴 **Red**, 🟠 **Orange**, 🟢 **Green**) by weighting accident counts and injury severities.
- **Coverage Point Generation**: Systematically maps 8 directional demand boundaries (N, S, E, W, NE, NW, SE, SW) around each generated zone to ensure edge-to-edge coverage.
- **LSCP Ambulance Optimization**: Uses Integer Linear Programming constraints (via the `PuLP` library) to solve the exact *Location Set Covering Problem*, outputting the absolute minimum ambulances needed.
- **Travel Time Estimation**: Utilizes the **Google Maps Distance Matrix API** for accurate driving times, complete with an automatic fallback mechanism that uses Haversine distance and localized estimated speeds.
- **Reverse Geocoding**: Leverages `geopy` to resolve the final calculated deployment coordinates back into human-readable real-world street addresses.
- **Interactive Visualization**: Generates rich, time-layered HTML maps (using `folium`) showcasing zones, accident clusters, and ambulance deployment positions.

## ⚙️ Architecture & Workflow

1. **Load Data**: The system ingest standard CSVs containing historical collision reports (Requires `LATITUDE`, `LONGITUDE`, `ACCIDENT TIME`, and injury data).
2. **Zone Division (`zone_division.py`)**: Standardizes time periods, scores the threat level dynamically, sets hyperparameters, and extracts the core hot-zones.
3. **Ambulance Placement (`ambulance_placement.py`)**: The primary execution pipeline. Generates candidate deployment coordinates, queries travel time matrices, runs the LSCP solver to prioritize high-risk (Red) zones first, and formats exact address directives.

## 🛠 Prerequisites

Ensure you have Python 3.8+ installed. You also need the following dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib folium pulp requests geopy
```

## 🚀 Setup & Usage

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/Emergency-Response-Optimization-with-Time-Based-Zone-Division.git
cd Emergency-Response-Optimization-with-Time-Based-Zone-Division
```

**2. Supply your API Key (Optional but Recommended):**
To calculate real traffic and travel times, set the Google Maps API Key via environment variable. If none is found, the system will seamlessly fallback to Haversine speed estimates.
```bash
# Windows (Command Prompt)
set GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"

# Windows (PowerShell)
$env:GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"

# Linux / macOS
export GOOGLE_MAPS_API_KEY="YOUR_API_KEY_HERE"
```

**3. Run the Placement Script:**
Execute the primary optimization script against your CSV dataset:
```bash
python ambulance_placement.py synthetic.csv
```

## 📊 Expected Output
Upon successful execution, the console will output:
1. Shift-by-shift summaries (Morning, Afternoon, Evening, Night).
2. Risk breakdown for recognized collision clusters.
3. **Minimum Ambulances Needed** per shift.
4. Exact geographic deployment latitude/longitudes with mapped **street addresses**.
5. Generated file: `ambulance_placement_map.html` which can be opened in any web browser.

---
*Created as part of an Advanced Thesis Study on Emergency Medical Services (EMS) route dispatch policies and proactive deployment algorithms.*
