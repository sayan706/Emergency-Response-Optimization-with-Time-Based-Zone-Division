"""
ambulance_placement.py
======================
Ambulance Placement Optimization using LSCP (Location Set Covering Problem)
on top of Time-Based Zone Division.

Pipeline:
  1. Load CSV → time-based zone division (reuses zone_division.py logic)
  2. Generate 8 directional boundary points per zone
  3. Build travel-time matrix (Google Maps API or Haversine fallback)
  4. Solve LSCP → minimum ambulances & optimal placements
  5. Output results per time period (console + interactive map)

Usage:
  python ambulance_placement.py                  # uses synthetic.csv
  python ambulance_placement.py mydata.csv       # uses custom CSV

Required packages:
  pip install pandas numpy scikit-learn matplotlib folium pulp requests
"""

import os
import sys
import concurrent.futures
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")

# ── Check required packages ─────────────────────────────────────────────
try:
    import pulp
except ImportError:
    print("ERROR: 'pulp' package required.  pip install pulp")
    sys.exit(1)

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("WARNING: 'folium' not installed – map will be skipped.  pip install folium")

try:
    import requests as req_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("WARNING: 'geopy' not installed – location names will be skipped.  pip install geopy")

# =========================================================================
# CONFIGURATION
# =========================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Google Maps Distance Matrix API key (set env var or edit here)
MAP_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "AIzaSyBLkK2TU_wmKk-cIhLKsiKVd2Qe_cnBIjg")
# TomTom API Key (fallback)
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "BHyiRkYeg52YC5WLBFNClJnbRF7PBdoX")

# Coverage threshold in minutes
COVERAGE_THRESHOLD_MINUTES = 15

# Fallback average road speed (km/h) when API is unavailable
FALLBACK_SPEED_KMH = 30

# Time period definitions (same as zone_division.py)
TIME_PERIOD_ORDER = ['Morning', 'Afternoon', 'Evening', 'Night']
TIME_RANGES = {
    'Morning':   '06:00 – 11:59',
    'Afternoon': '12:00 – 17:59',
    'Evening':   '18:00 – 21:59',
    'Night':     '22:00 – 05:59',
}

ZONE_PRIORITY = {'Red': 0, 'Orange': 1, 'Green': 2}


# =========================================================================
# 1. ZONE DIVISION FUNCTIONS  (logic copied from zone_division.py)
# =========================================================================

def _extract_hour(t):
    """Extract hour from a time string like '8:25:00' or '13:00:00'."""
    try:
        return pd.to_datetime(str(t), format='%H:%M:%S').hour
    except Exception:
        try:
            return pd.to_datetime(str(t)).hour
        except Exception:
            return np.nan


def _assign_time_period(hour):
    """Map hour → Morning / Afternoon / Evening / Night."""
    if pd.isna(hour):
        return 'Unknown'
    hour = int(hour)
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'


def load_and_clean_data(file_path):
    """Load CSV, validate columns, clean data, compute severity & time period."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # --- required columns ---
    for col in ['LATITUDE', 'LONGITUDE']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- detect severity columns ---
    fatal_col = grievous_col = minor_col = None
    for target, keyword in [('fatal', 'FATAL'), ('grievous', 'GRIEV'), ('minor', 'MINOR')]:
        exact = {'fatal': 'NO OF FATALITIES',
                 'grievous': 'GRIEVOUSLY INJURED',
                 'minor': 'MINOR INJURED'}[target]
        if exact in df.columns:
            if target == 'fatal':    fatal_col = exact
            elif target == 'grievous': grievous_col = exact
            else:                      minor_col = exact
        else:
            matches = [c for c in df.columns if keyword in c.upper()]
            if matches:
                if target == 'fatal':    fatal_col = matches[0]
                elif target == 'grievous': grievous_col = matches[0]
                else:                      minor_col = matches[0]

    # --- convert to numeric ---
    num_cols = ['LATITUDE', 'LONGITUDE']
    for c in [fatal_col, grievous_col, minor_col]:
        if c:
            num_cols.append(c)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- fill / create severity columns ---
    for col_ref, temp_name in [(fatal_col, 'TEMP_FATAL'),
                                (grievous_col, 'TEMP_GRIEVOUS'),
                                (minor_col, 'TEMP_MINOR')]:
        if col_ref:
            df[col_ref] = df[col_ref].fillna(0)
        else:
            df[temp_name] = 0
    if not fatal_col:    fatal_col = 'TEMP_FATAL'
    if not grievous_col: grievous_col = 'TEMP_GRIEVOUS'
    if not minor_col:    minor_col = 'TEMP_MINOR'

    df = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
    if len(df) < 3:
        raise ValueError("Not enough valid accident points for clustering.")

    # --- time column ---
    time_col = 'ACCIDENT TIME' if 'ACCIDENT TIME' in df.columns else None
    if time_col is None:
        possible = [c for c in df.columns if 'TIME' in c.upper()]
        if possible:
            time_col = possible[0]
    if time_col is None:
        raise ValueError("No time column found. Need 'ACCIDENT TIME' or similar.")

    df['HOUR'] = df[time_col].apply(_extract_hour)
    df['TIME_PERIOD'] = df['HOUR'].apply(_assign_time_period)
    df = df[df['TIME_PERIOD'] != 'Unknown'].copy()

    # --- severity score ---
    df['SEVERITY_SCORE'] = 4 * df[fatal_col] + 2 * df[grievous_col] + 1 * df[minor_col]
    if df['SEVERITY_SCORE'].sum() == 0:
        df['SEVERITY_SCORE'] = 1

    return df


def auto_tune_dbscan(X_scaled, n_rows):
    """Auto-tune DBSCAN and return best labels, params, score."""
    best_score = -9999
    best_params = None
    best_labels = None

    for eps in np.arange(0.2, 1.6, 0.1):
        for ms in [2, 3, 4, 5]:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
            uq = set(labels)
            nc = len(uq - {-1})
            nn = list(labels).count(-1)
            nr = nn / len(labels)
            if nc < 1:
                continue

            if nc >= 2:
                mask = labels != -1
                if len(set(labels[mask])) >= 2 and np.sum(mask) > 2:
                    try:
                        sil = silhouette_score(X_scaled[mask], labels[mask])
                    except Exception:
                        sil = -1
                else:
                    sil = -1
            else:
                sil = 0.1

            cs = pd.Series(labels[labels != -1]).value_counts()
            avg_cs = cs.mean() if len(cs) > 0 else 0
            cp = 1.5 if nc > max(6, n_rows // 5) else 0
            np_ = 1.5 if nr > 0.5 else (0.7 if nr > 0.3 else 0)
            sr = min(avg_cs / 5, 2.0)
            fs = 2.0 * sil + 1.2 * sr - np_ - cp

            if fs > best_score:
                best_score = fs
                best_params = (eps, ms)
                best_labels = labels

    if best_params is None:
        raise ValueError("No valid DBSCAN parameters found.")
    return best_labels, best_params, best_score


def assign_zones_for_period(period_df, period_name, cluster_centers):
    """Compute per-cluster risk & assign Red/Orange/Green for one time period."""
    if len(period_df) == 0:
        return pd.DataFrame()
    pc = period_df[period_df['CLUSTER'] != -1].copy()
    if len(pc) == 0:
        return pd.DataFrame()

    s = pc.groupby('CLUSTER').agg(
        AVG_SEVERITY=('SEVERITY_SCORE', 'mean'),
        TOTAL_SEVERITY=('SEVERITY_SCORE', 'sum'),
        ACCIDENT_COUNT=('CLUSTER', 'count'),
    ).reset_index()

    s['RISK_SCORE'] = 0.4 * s['ACCIDENT_COUNT'] + 0.3 * s['AVG_SEVERITY'] + 0.3 * s['TOTAL_SEVERITY']

    if len(s) == 1:
        s['ZONE'] = 'Red'
    elif len(s) == 2:
        s = s.sort_values('RISK_SCORE', ascending=False).reset_index(drop=True)
        s['ZONE'] = ['Red', 'Orange']
    else:
        rt = s['RISK_SCORE'].quantile(0.66)
        ot = s['RISK_SCORE'].quantile(0.33)
        s['ZONE'] = s['RISK_SCORE'].apply(
            lambda x: 'Red' if x >= rt else ('Orange' if x >= ot else 'Green'))

    if len(s) > 1:
        mn, mx = s['RISK_SCORE'].min(), s['RISK_SCORE'].max()
        if mx > 0 and mn / mx >= 0.50:
            s.loc[s['ZONE'] == 'Green', 'ZONE'] = 'Orange'

    s['TIME_PERIOD'] = period_name
    s = s.merge(cluster_centers, on='CLUSTER', how='left')
    return s


def run_zone_division(file_path):
    """Full zone-division pipeline. Returns df, cluster_centers, summary, clustered_df."""
    print("=" * 70)
    print("  STEP 1: TIME-BASED ZONE DIVISION")
    print("=" * 70)

    df = load_and_clean_data(file_path)

    print("\n  Accident Time Distribution:")
    for tp in TIME_PERIOD_ORDER:
        print(f"    {tp}: {(df['TIME_PERIOD'] == tp).sum()} accidents")

    X = df[['LATITUDE', 'LONGITUDE']].copy()
    X_scaled = StandardScaler().fit_transform(X)
    best_labels, best_params, _ = auto_tune_dbscan(X_scaled, len(df))
    print(f"\n  DBSCAN: eps={best_params[0]:.1f}, min_samples={best_params[1]}")

    df['CLUSTER'] = best_labels
    clustered_df = df[df['CLUSTER'] != -1].copy()
    if len(clustered_df) == 0:
        raise ValueError("No hotspot clusters found.")
    print(f"  Clusters: {clustered_df['CLUSTER'].nunique()}, Noise: {(df['CLUSTER'] == -1).sum()}")

    cc = clustered_df.groupby('CLUSTER').agg(
        CENTER_LATITUDE=('LATITUDE', 'mean'),
        CENTER_LONGITUDE=('LONGITUDE', 'mean'),
    ).reset_index()

    summaries = []
    for p in TIME_PERIOD_ORDER:
        ps = assign_zones_for_period(df[df['TIME_PERIOD'] == p], p, cc)
        if len(ps) > 0:
            summaries.append(ps)
    if not summaries:
        raise ValueError("No valid time-period zone assignments.")

    tzs = pd.concat(summaries, ignore_index=True)
    zo = {'Red': 0, 'Orange': 1, 'Green': 2}
    po = {p: i for i, p in enumerate(TIME_PERIOD_ORDER)}
    tzs['ZONE_ORDER'] = tzs['ZONE'].map(zo)
    tzs['PERIOD_ORDER'] = tzs['TIME_PERIOD'].map(po)
    tzs = tzs.sort_values(['PERIOD_ORDER', 'ZONE_ORDER', 'RISK_SCORE'],
                           ascending=[True, True, False]).reset_index(drop=True)
    return df, cc, tzs, clustered_df


# =========================================================================
# 2. BOUNDARY POINT GENERATION
# =========================================================================

def generate_boundary_points(clustered_df, cluster_id, zone_color):
    """Generate 8 directional boundary points (N/S/E/W/NE/NW/SE/SW) for a cluster."""
    pts = clustered_df[clustered_df['CLUSTER'] == cluster_id]
    clat, clon = pts['LATITUDE'].mean(), pts['LONGITUDE'].mean()

    lat_off = max(pts['LATITUDE'].std() * 2,
                  (pts['LATITUDE'].max() - pts['LATITUDE'].min()) / 2,
                  0.002)
    lon_off = max(pts['LONGITUDE'].std() * 2,
                  (pts['LONGITUDE'].max() - pts['LONGITUDE'].min()) / 2,
                  0.002)

    diag = 0.707  # cos(45°)
    dirs = {
        'N':  (clat + lat_off,        clon),
        'S':  (clat - lat_off,        clon),
        'E':  (clat,                  clon + lon_off),
        'W':  (clat,                  clon - lon_off),
        'NE': (clat + lat_off * diag, clon + lon_off * diag),
        'NW': (clat + lat_off * diag, clon - lon_off * diag),
        'SE': (clat - lat_off * diag, clon + lon_off * diag),
        'SW': (clat - lat_off * diag, clon - lon_off * diag),
    }
    return [{'CLUSTER': cluster_id, 'ZONE': zone_color, 'DIRECTION': d,
             'LATITUDE': la, 'LONGITUDE': lo, 'CENTER_LAT': clat, 'CENTER_LON': clon}
            for d, (la, lo) in dirs.items()]


# =========================================================================
# 3. TRAVEL-TIME ESTIMATION
# =========================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometres."""
    R = 6371
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def travel_time_haversine(lat1, lon1, lat2, lon2, speed=FALLBACK_SPEED_KMH):
    """Estimated travel time (minutes) via Haversine + road detour factor 1.4×."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1.4 / speed * 60


def _fetch_matrix_chunk(i0, j0, origins_chunk, dests_chunk, key):
    o_str = "|".join(f"{la},{lo}" for la, lo in origins_chunk)
    d_str = "|".join(f"{la},{lo}" for la, lo in dests_chunk)
    url = (f"https://maps.googleapis.com/maps/api/distancematrix/json"
           f"?origins={o_str}&destinations={d_str}&key={key}&mode=driving")
    try:
        data = req_lib.get(url, timeout=30).json()
        if data['status'] != 'OK':
            return False, f"API Error: {data.get('error_message', data['status'])}"
        
        chunk_data = []
        for row in data['rows']:
            row_data = []
            for el in row['elements']:
                val = (el['duration']['value'] / 60) if el['status'] == 'OK' else float('inf')
                row_data.append(val)
            chunk_data.append(row_data)
        return True, (i0, j0, chunk_data)
    except Exception as e:
        return False, f"API request failed: {e}"

def _fetch_tomtom_matrix_chunk(i0, j0, origins_chunk, dests_chunk, key, retries=3):
    import time
    
    url = f"https://api.tomtom.com/routing/matrix/2?key={key}"
    headers = {
        'Content-Type': 'application/json'
    }
    
    origins_payload = [{"point": {"latitude": la, "longitude": lo}} for la, lo in origins_chunk]
    dests_payload = [{"point": {"latitude": la, "longitude": lo}} for la, lo in dests_chunk]
    
    payload = {
        "origins": origins_payload,
        "destinations": dests_payload
    }
    
    for attempt in range(retries):
        try:
            # Stay inside TomTom's 5 Transactions Per Second (cells/sec) free limit
            time.sleep(1.1)
            resp = req_lib.post(url, json=payload, headers=headers, timeout=30)
            data = resp.json()
            
            if resp.status_code == 429 or 'error' in data or 'detailedError' in data:
                err_msg = str(data.get('detailedError', data.get('error', resp.status_code)))

                # If this error trips specifically on OVER_TRANSACTION_LIMIT even after math controls,
                # it definitively means the DAILY 2,500 transaction cap is exceeded (not just QPS).
                if "OVER_TRANSACTION_LIMIT" in err_msg or "FORBIDDEN" in err_msg:
                    return False, f"TomTom API Daily Quota Exceeded."
                
                if attempt < retries - 1:
                    print(f"    [TomTom] Rate limit warning. Retrying {attempt+1}/{retries} in 5s...")
                    time.sleep(5)
                    continue
                return False, f"TomTom API Error: {err_msg}"
            
            # TomTom v2 returns a list inside 'data'
            matrix_data = data.get('data', [])
            if not matrix_data:
                return False, "TomTom API returned no matrix data."
                
            # Initialize with inf
            chunk_data = [[float('inf')] * len(dests_chunk) for _ in range(len(origins_chunk))]
            
            for item in matrix_data:
                o_idx = item.get('originIndex')
                d_idx = item.get('destinationIndex')
                
                # Verify indices are within bounds just in case
                if o_idx is not None and d_idx is not None and o_idx < len(origins_chunk) and d_idx < len(dests_chunk):
                    summary = item.get('routeSummary')
                    if summary and 'travelTimeInSeconds' in summary:
                        chunk_data[o_idx][d_idx] = summary['travelTimeInSeconds'] / 60.0
                        
            return True, (i0, j0, chunk_data)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return False, f"TomTom API request failed: {e}"
            
    return False, "TomTom API failed after max retries."


def _api_travel_matrix(origins, dests, key, api_type='google'):
    """Query Distance Matrix API concurrently. Returns matrix (minutes) or None."""
    if not REQUESTS_AVAILABLE:
        return None
    mat = np.zeros((len(origins), len(dests)))
    
    # Google comfortably handles large block requests (BS=10 -> 100 cells)
    # TomTom Freemium charges 1 Transaction per cell. It permits max 5 TPS. 
    # Therefore max TomTom Matrix dimension is 2x2 (4 cells per sec) to avoid OVER_TRANSACTION_LIMIT.
    BS = 10 if api_type == 'google' else 2
    
    tasks = []
    workers = 10 if api_type == 'google' else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i0 in range(0, len(origins), BS):
            o_chunk = origins[i0:i0 + BS]
            for j0 in range(0, len(dests), BS):
                d_chunk = dests[j0:j0 + BS]
                if api_type == 'google':
                    tasks.append(executor.submit(_fetch_matrix_chunk, i0, j0, o_chunk, d_chunk, key))
                else:
                    tasks.append(executor.submit(_fetch_tomtom_matrix_chunk, i0, j0, o_chunk, d_chunk, key))
                
        for future in concurrent.futures.as_completed(tasks):
            success, result = future.result()
            if not success:
                print(f"    {result}")
                return None
            
            i0, j0, chunk_data = result
            for i, row_data in enumerate(chunk_data):
                for j, val in enumerate(row_data):
                    mat[i0 + i][j0 + j] = val

    return mat


def build_travel_matrix(candidates, demands):
    """Build travel-time matrix (candidates × demands). Returns (matrix, method_name)."""
    # 1. Try Google Maps API
    use_google = (MAP_API_KEY not in ("YOUR_API_KEY_HERE", "") and REQUESTS_AVAILABLE)
    if use_google:
        print("    Using Google Maps Distance Matrix API …")
        m = _api_travel_matrix(candidates, demands, MAP_API_KEY, 'google')
        if m is not None:
            return m, "Google Maps API"
        print("    Google API failed → Trying TomTom API fallback")

    # 2. Try TomTom API
    use_tomtom = (TOMTOM_API_KEY not in ("", None) and REQUESTS_AVAILABLE)
    if use_tomtom:
        print("    Using TomTom Matrix Routing API …")
        m = _api_travel_matrix(candidates, demands, TOMTOM_API_KEY, 'tomtom')
        if m is not None:
            return m, "TomTom API"
        print("    TomTom API failed → Haversine fallback")

    # 3. Fallback to Haversine
    print(f"    Using Haversine estimation ({FALLBACK_SPEED_KMH} km/h) …")
    mat = np.zeros((len(candidates), len(demands)))
    for i, (cla, clo) in enumerate(candidates):
        for j, (dla, dlo) in enumerate(demands):
            mat[i][j] = travel_time_haversine(cla, clo, dla, dlo)
    return mat, "Haversine Estimation"


# =========================================================================
# 4. CANDIDATE LOCATION GENERATION
# =========================================================================

def generate_candidates(cc_df, boundary_pts):
    """
    Candidate ambulance locations =
      cluster centres + boundary points + midpoints between centres.
    """
    locs, labels = [], []
    for _, r in cc_df.iterrows():
        locs.append((r['CENTER_LATITUDE'], r['CENTER_LONGITUDE']))
        labels.append(f"Center-C{int(r['CLUSTER'])}")
    for p in boundary_pts:
        locs.append((p['LATITUDE'], p['LONGITUDE']))
        labels.append(f"Bnd-C{int(p['CLUSTER'])}-{p['DIRECTION']}")
    centres = cc_df[['CENTER_LATITUDE', 'CENTER_LONGITUDE']].values
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            locs.append(((centres[i][0] + centres[j][0]) / 2,
                         (centres[i][1] + centres[j][1]) / 2))
            labels.append(f"Mid-C{int(cc_df.iloc[i]['CLUSTER'])}-C{int(cc_df.iloc[j]['CLUSTER'])}")
    return locs, labels


# =========================================================================
# 5. LSCP SOLVER
# =========================================================================

def solve_lscp(demand_pts, demand_zones, cand_locs, cand_labels, tt_matrix, threshold):
    """
    Solve Location Set Covering Problem.
    Priority: Red → Orange → Green (solved inclusively; relaxes lower priorities
    if infeasible).
    Returns dict with selected ambulance info + coverage mapping.
    """
    nc, nd = len(cand_locs), len(demand_pts)

    # coverage[j][i] = 1 if candidate j reaches demand i within threshold
    cov = (tt_matrix <= threshold).astype(int)

    uncoverable = [i for i in range(nd) if cov[:, i].sum() == 0]
    if uncoverable:
        print(f"    ⚠ {len(uncoverable)} point(s) unreachable within {threshold} min")

    red_i   = [i for i in range(nd) if demand_zones[i] == 'Red'    and i not in uncoverable]
    orange_i = [i for i in range(nd) if demand_zones[i] == 'Orange' and i not in uncoverable]
    green_i  = [i for i in range(nd) if demand_zones[i] == 'Green'  and i not in uncoverable]

    # Try covering all → relax green → relax orange if infeasible
    for label, indices in [("All", red_i + orange_i + green_i),
                           ("Red+Orange", red_i + orange_i),
                           ("Red only", red_i)]:
        if not indices:
            continue
        prob = pulp.LpProblem("LSCP", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x{j}", cat='Binary') for j in range(nc)]
        prob += pulp.lpSum(x)
        for i in indices:
            covers = [j for j in range(nc) if cov[j][i]]
            if covers:
                prob += pulp.lpSum(x[j] for j in covers) >= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] == 'Optimal':
            sel = [j for j in range(nc) if x[j].varValue and x[j].varValue > 0.5]
            cov_map = {i: [j for j in sel if cov[j][i]] for i in range(nd)}
            return {'selected_indices': sel,
                    'selected_locations': [cand_locs[j] for j in sel],
                    'selected_labels': [cand_labels[j] for j in sel],
                    'ambulance_count': len(sel),
                    'coverage_map': cov_map,
                    'uncoverable': uncoverable,
                    'status': f"Optimal ({label})"}

    return {'selected_indices': [], 'selected_locations': [],
            'selected_labels': [], 'ambulance_count': 0,
            'coverage_map': {}, 'uncoverable': uncoverable,
            'status': 'Infeasible'}


# =========================================================================
# 6. CONSOLE OUTPUT
# =========================================================================

def get_location_name(lat, lon):
    """Retrieve a concise location name using reverse geocoding via Nominatim."""
    if not GEOPY_AVAILABLE:
        return ""
    try:
        geolocator = Nominatim(user_agent="ambulance_optimization_thesis", timeout=5)
        location = geolocator.reverse(f"{lat}, {lon}")
        if location:
            addr = location.raw.get('address', {})
            parts = []
            if 'road' in addr: parts.append(addr['road'])
            if 'suburb' in addr: parts.append(addr['suburb'])
            elif 'neighbourhood' in addr: parts.append(addr['neighbourhood'])
            
            city = addr.get('city', addr.get('town', addr.get('village', '')))
            if city: parts.append(city)
            
            if parts:
                return "📍 " + ", ".join(parts)
            # Fallback to the first part of the address
            return "📍 " + location.address.split(',')[0]
    except Exception:
        pass
    return ""


def print_period(period, zs, bpts, lscp, method):
    """Print full results for one time period."""
    print(f"\n{'─' * 70}")
    print(f"  ⏰  {period.upper()} ({TIME_RANGES[period]})")
    print(f"{'─' * 70}")

    if len(zs) == 0:
        print("    No clustered accidents.")
        return

    # ── zones ──
    emoji = {'Red': '🔴', 'Orange': '🟠', 'Green': '🟢'}
    print("\n  📊 ZONE SUMMARY")
    for _, r in zs.iterrows():
        print(f"    {emoji.get(r['ZONE'],'⚪')} Cluster {int(r['CLUSTER'])} → {r['ZONE']}  "
              f"| Accidents: {int(r['ACCIDENT_COUNT'])}  Avg Sev: {r['AVG_SEVERITY']:.2f}  "
              f"Risk: {r['RISK_SCORE']:.2f}  "
              f"Center: ({r['CENTER_LATITUDE']:.6f}, {r['CENTER_LONGITUDE']:.6f})")

    # ── boundary points ──
    print("\n  📍 BOUNDARY POINTS (8 per zone)")
    cur = None
    for p in bpts:
        if p['CLUSTER'] != cur:
            cur = p['CLUSTER']
            print(f"    {emoji.get(p['ZONE'],'⚪')} Cluster {int(cur)} ({p['ZONE']}):")
        print(f"       {p['DIRECTION']:3s} → ({p['LATITUDE']:.6f}, {p['LONGITUDE']:.6f})")

    # ── ambulance placements ──
    print(f"\n  🚑 LSCP RESULT   [Solver: {lscp['status']}]  [Method: {method}]")
    print(f"    Threshold: {COVERAGE_THRESHOLD_MINUTES} min")
    print(f"    ✅ MINIMUM AMBULANCES NEEDED: {lscp['ambulance_count']}")
    for idx, (loc, lab) in enumerate(zip(lscp['selected_locations'], lscp['selected_labels'])):
        loc_name = get_location_name(loc[0], loc[1])
        print(f"      🚑 #{idx+1}  ({loc[0]:.6f}, {loc[1]:.6f})  ← {lab}")
        if loc_name:
            print(f"         {loc_name}")

    # ── coverage mapping ──
    if lscp['ambulance_count'] > 0:
        print(f"\n  📋 COVERAGE MAPPING")
        for idx, j in enumerate(lscp['selected_indices']):
            covered = [f"C{int(bpts[i]['CLUSTER'])}-{bpts[i]['DIRECTION']}({bpts[i]['ZONE']})"
                       for i, ambs in lscp['coverage_map'].items()
                       if j in ambs and i < len(bpts)]
            print(f"    🚑 #{idx+1} ({lscp['selected_labels'][idx]}):")
            for k in range(0, len(covered), 4):
                print(f"       {', '.join(covered[k:k+4])}")

    if lscp['uncoverable']:
        print(f"\n  ⚠️  {len(lscp['uncoverable'])} point(s) beyond threshold reach")


# =========================================================================
# 7. MAP VISUALIZATION
# =========================================================================

def generate_map(all_results, out_path):
    """Create interactive Folium HTML map with layer per time period."""
    if not FOLIUM_AVAILABLE:
        print("\n  Folium not available – skipping map.")
        return

    lats, lons = [], []
    for r in all_results.values():
        for p in r['bpts']:
            lats.append(p['LATITUDE']); lons.append(p['LONGITUDE'])
        for loc in r['lscp']['selected_locations']:
            lats.append(loc[0]); lons.append(loc[1])
    if not lats:
        return

    m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=13)
    zcol = {'Red': 'red', 'Orange': 'orange', 'Green': 'green'}

    for period, res in all_results.items():
        fg = folium.FeatureGroup(name=f"{period} ({TIME_RANGES[period]})",
                                 show=(period == 'Morning'))
        
        # 1. DRAW ZONES (POLYGONS) SO OVERLAPS ARE VERY CLEAR
        # Draw Green -> Orange -> Red so Red is on top visually
        for z_color in ['Green', 'Orange', 'Red']:
            subset = res['zs'][res['zs']['ZONE'] == z_color]
            for _, row in subset.iterrows():
                cluster_id = row['CLUSTER']
                c_bpts = [p for p in res['bpts'] if p['CLUSTER'] == cluster_id]
                dir_map = {p['DIRECTION']: (p['LATITUDE'], p['LONGITUDE']) for p in c_bpts}
                # cyclic order for polygon
                cyclic_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                poly_points = [dir_map[d] for d in cyclic_dirs if d in dir_map]
                if poly_points:
                    folium.Polygon(
                        locations=poly_points,
                        color=zcol.get(z_color, 'gray'),
                        weight=2,
                        fill=True,
                        fill_color=zcol.get(z_color, 'gray'),
                        fill_opacity=0.3,
                        tooltip=f"Zone {z_color} (Cluster {int(cluster_id)})"
                    ).add_to(fg)

        # 2. demand points (boundary points), ordered Green -> Orange -> Red
        for z_color in ['Green', 'Orange', 'Red']:
            for p in [x for x in res['bpts'] if x['ZONE'] == z_color]:
                folium.CircleMarker(
                    [p['LATITUDE'], p['LONGITUDE']], radius=5,
                    color=zcol.get(p['ZONE'], 'gray'), fill=True, fill_opacity=0.7,
                    popup=f"{p['ZONE']} C{int(p['CLUSTER'])} {p['DIRECTION']}",
                    tooltip=f"{p['ZONE']}-{p['DIRECTION']}"
                ).add_to(fg)

        # 3. cluster centres, offset so Red markers sit above Green
        for z_color in ['Green', 'Orange', 'Red']:
            for _, row in res['zs'][res['zs']['ZONE'] == z_color].iterrows():
                z_offset = {'Red': 1000, 'Orange': 500, 'Green': 0}.get(z_color, 0)
                folium.Marker(
                    [row['CENTER_LATITUDE'], row['CENTER_LONGITUDE']],
                    tooltip=f"C{int(row['CLUSTER'])} {row['ZONE']}",
                    icon=folium.Icon(color=zcol.get(row['ZONE'], 'gray'),
                                     icon='warning-sign', prefix='glyphicon'),
                    z_index_offset=z_offset
                ).add_to(fg)

        # 4. ambulances
        for idx, loc in enumerate(res['lscp']['selected_locations']):
            folium.Marker(
                [loc[0], loc[1]],
                popup=f"🚑 #{idx+1} {res['lscp']['selected_labels'][idx]}",
                tooltip=f"🚑 #{idx+1}",
                icon=folium.Icon(color='blue', icon='plus-sign', prefix='glyphicon'),
                z_index_offset=2000
            ).add_to(fg)
            folium.Circle([loc[0], loc[1]], radius=5360, color='blue',
                          fill=False, opacity=0.25).add_to(fg)
            
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_path)
    print(f"\n  ✅ Map saved → {out_path}")


# =========================================================================
# 8. MAIN PIPELINE
# =========================================================================

def main(csv_path):
    """Run complete pipeline: zone division → boundary pts → LSCP → output."""
    print("\n" + "=" * 70)
    print("  🚑  AMBULANCE PLACEMENT OPTIMIZATION (LSCP)")
    print("=" * 70)

    # Step 1 – zone division
    df, cc, tzs, cdf = run_zone_division(csv_path)

    # Step 2-5 – per time-period optimisation
    print("\n" + "=" * 70)
    print("  STEP 2: AMBULANCE PLACEMENT PER TIME PERIOD")
    print("=" * 70)

    all_results = {}
    for period in TIME_PERIOD_ORDER:
        pz = tzs[tzs['TIME_PERIOD'] == period]
        if len(pz) == 0:
            print(f"\n  ⏭  {period}: no zones – skipped")
            continue

        bpts = []
        for _, r in pz.iterrows():
            bpts.extend(generate_boundary_points(cdf, int(r['CLUSTER']), r['ZONE']))
        bpts.sort(key=lambda p: ZONE_PRIORITY.get(p['ZONE'], 9))

        dem = [(p['LATITUDE'], p['LONGITUDE']) for p in bpts]
        dzones = [p['ZONE'] for p in bpts]
        cands, clabels = generate_candidates(
            pz[['CLUSTER', 'CENTER_LATITUDE', 'CENTER_LONGITUDE']], bpts)

        tt, method = build_travel_matrix(cands, dem)
        lscp = solve_lscp(dem, dzones, cands, clabels, tt, COVERAGE_THRESHOLD_MINUTES)

        all_results[period] = {'zs': pz, 'bpts': bpts, 'lscp': lscp, 'method': method}
        print_period(period, pz, bpts, lscp, method)

    # ── overall summary ──
    print("\n" + "=" * 70)
    print("  📊 OVERALL SUMMARY")
    print("=" * 70)
    total = 0
    for period, res in all_results.items():
        c = res['lscp']['ambulance_count']
        total += c
        zs = res['zs']
        print(f"  {period:12s}  {c} ambulance(s)  |  "
              f"🔴{len(zs[zs['ZONE']=='Red'])} "
              f"🟠{len(zs[zs['ZONE']=='Orange'])} "
              f"🟢{len(zs[zs['ZONE']=='Green'])}")
    print(f"\n  Total across all periods : {total}")
    peak = max((r['lscp']['ambulance_count'] for r in all_results.values()), default=0)
    print(f"  Peak (max in any period) : {peak}")
    print(f"  Coverage threshold       : {COVERAGE_THRESHOLD_MINUTES} min")

    # ── map ──
    if FOLIUM_AVAILABLE and all_results:
        mp = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                          "ambulance_placement_map.html")
        generate_map(all_results, mp)

    print("\n" + "=" * 70)
    print("  ✅ ANALYSIS COMPLETE")
    print("=" * 70)
    return all_results


# =========================================================================
# ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "synthetic.csv"
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found → {csv_file}")
        print("Usage: python ambulance_placement.py [path_to_csv]")
        sys.exit(1)
    main(csv_file)
