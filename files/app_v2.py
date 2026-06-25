"""
AeroSight v2 — Flight Delay Decision Support System
=====================================================
Weather-Aware · LightGBM · SHAP · Prescriptive Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import requests
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AeroSight — Flight Delay DSS",
    page_icon="✈️", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2.4rem; font-weight: 900;
        background: linear-gradient(90deg, #1a365d, #3182ce);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding-top: 0.5rem; letter-spacing: -1px;
    }
    .sub-title { font-size: 1rem; color: #718096; text-align: center; margin-bottom: 1.5rem; }
    .card { background: #1a2035; border: 1px solid #2d3748; border-radius: 16px; padding: 24px; margin: 8px 0; }
    .card-label { font-size: 10px; font-weight: 700; color: #718096; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
    .card-value { font-size: 22px; font-weight: 800; color: #e2e8f0; }
    .card-sub { font-size: 12px; color: #718096; margin-top: 2px; }
    .risk-header { background: #1a2035; border: 1px solid #2d3748; border-radius: 20px; padding: 32px 36px; margin-bottom: 20px; }
    .risk-title { font-size: 2.8rem; font-weight: 900; color: #e2e8f0; margin: 0; line-height: 1.1; }
    .risk-subtitle { font-size: 1rem; color: #718096; margin-top: 8px; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin: 4px 4px 4px 0; }
    .badge-green  { background:#1a3a2a; color:#68D391; border:1px solid #276749; }
    .badge-yellow { background:#3a2e1a; color:#F6AD55; border:1px solid #744210; }
    .badge-red    { background:#3a1a1a; color:#FC8181; border:1px solid #C53030; }
    .badge-blue   { background:#1a2a3a; color:#63B3ED; border:1px solid #2b6cb0; }
    .route-card { background: #141927; border: 1px solid #2d3748; border-radius: 14px; padding: 24px; }
    .route-airport { font-size: 2.8rem; font-weight: 900; color: #e2e8f0; letter-spacing: -1px; }
    .route-city { font-size: 12px; color: #718096; margin-top: 2px; }
    .shap-bar-container { margin: 8px 0; padding: 10px 14px; background: #141927; border-radius: 10px; border: 1px solid #2d3748; }
    .shap-label { font-size: 13px; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }
    .shap-bar-track { background: #2d3748; border-radius: 6px; height: 8px; width: 100%; position: relative; }
    .why-section { background: #1a2035; border: 1px solid #2d3748; border-radius: 14px; padding: 20px; margin-top: 16px; }
    .why-title { font-size: 14px; font-weight: 700; color: #e2e8f0; margin-bottom: 14px; }
    .rec-box { background: #1a2a3a; border-left: 4px solid #3182CE; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 13px; color: #90CDF4; }
    .rec-box-warn { background: #2a2a1a; border-left: 4px solid #D69E2E; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 13px; color: #F6AD55; }
    .rec-box-danger { background: #2a1a1a; border-left: 4px solid #E53E3E; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 13px; color: #FC8181; }
    .section-header { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; border-left: 4px solid #3182ce; padding-left: 0.7rem; margin: 1rem 0 0.8rem 0; }
    div[data-testid="metric-container"] { background: #1a2035; border: 1px solid #2d3748; border-radius: 10px; padding: 0.8rem; }
    div[data-testid="metric-container"] label { color: #718096 !important; }
    div[data-testid="metric-container"] div { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────
import os
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aerosights_artifacts')
DAY_NAMES   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
CONUS_LAT   = (24.0, 49.5)
CONUS_LON   = (-125.0, -66.0)

WEATHER_CODES = {
    0:'Clear sky',1:'Mainly clear',2:'Partly cloudy',3:'Overcast',
    45:'Foggy',48:'Icing fog',51:'Light drizzle',53:'Drizzle',
    55:'Heavy drizzle',61:'Slight rain',63:'Moderate rain',65:'Heavy rain',
    71:'Slight snow',73:'Moderate snow',75:'Heavy snow',77:'Snow grains',
    80:'Slight showers',81:'Moderate showers',82:'Violent showers',
    85:'Slight snow showers',86:'Heavy snow showers',95:'Thunderstorm',
    96:'Thunderstorm w/ hail',99:'Thunderstorm w/ heavy hail'
}

AIRLINE_FULL = {
    'AA':'American Airlines','DL':'Delta Air Lines','UA':'United Airlines',
    'WN':'Southwest Airlines','B6':'JetBlue Airways','AS':'Alaska Airlines',
    'NK':'Spirit Airlines','F9':'Frontier Airlines','G4':'Allegiant Air',
    'HA':'Hawaiian Airlines','MQ':'Envoy Air','OO':'SkyWest Airlines',
    '9E':'Endeavor Air','YX':'Republic Airways','OH':'PSA Airlines',
    'YV':'Mesa Airlines','QX':'Horizon Air','CP':'Compass Airlines',
}

FEATURE_LABELS = {
    'DepHour':'Departure Hour','day_of_week':'Day of Week','month':'Month',
    'distance':'Route Distance','IsWeekend':'Weekend Flight',
    'Airline_enc':'Airline','Origin_enc':'Origin Airport','Dest_enc':'Destination',
    'AvgWeatherDelay_Route':'Avg Weather Delay (Route)',
    'AvgLateAircraft_Airline':'Avg Late Aircraft','temp_c':'Temperature (°C)',
    'windspeed_kmh':'Wind Speed (km/h)','precip_mm':'Precipitation (mm)',
    'weathercode':'Weather Condition',
}

# ── Loaders ────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    mp = Path(ARTIFACTS_DIR)/'model.pkl'
    ep = Path(ARTIFACTS_DIR)/'metadata.json'
    tp = Path(ARTIFACTS_DIR)/'best_threshold.json'
    if not mp.exists():
        st.error(f"Model not found at: {ARTIFACTS_DIR}")
        st.write("Files available:", list(Path(ARTIFACTS_DIR).parent.iterdir()) if Path(ARTIFACTS_DIR).parent.exists() else "Parent dir missing")
        return None, {}, 0.70, []
    model    = joblib.load(mp)
    metadata = json.load(open(ep, encoding='utf-8')) if ep.exists() else {}
    threshold, selected = 0.70, metadata.get('all_features', [])
    if tp.exists():
        t = json.load(open(tp, encoding='utf-8'))
        threshold = t.get('best_threshold', 0.70)
        selected  = t.get('selected_features', selected)
    return model, metadata, threshold, selected

@st.cache_resource
def load_encoders():
    enc = {}
    for n in ['le_airline','le_origin','le_dest']:
        p = Path(ARTIFACTS_DIR)/f'{n}.pkl'
        if p.exists(): enc[n] = joblib.load(p)
    return enc

@st.cache_data
def load_stats():
    s = {}
    for n in ['airline_stats','route_stats','hourly_stats',
              'weather_stats','monthly_weather_stats','airport_stats']:
        p = Path(ARTIFACTS_DIR)/f'{n}.csv'
        if p.exists(): s[n] = pd.read_csv(p)
    return s

@st.cache_data
def load_route_data():
    lp = Path(ARTIFACTS_DIR)/'route_lookup.csv'
    tp = Path(ARTIFACTS_DIR)/'flight_times.csv'
    return (pd.read_csv(lp) if lp.exists() else pd.DataFrame(),
            pd.read_csv(tp) if tp.exists() else pd.DataFrame())

@st.cache_data
def load_explainability():
    fi, med = {}, {}
    fp = Path(ARTIFACTS_DIR)/'feature_importance.json'
    mp = Path(ARTIFACTS_DIR)/'feature_medians.json'
    if fp.exists(): fi  = json.load(open(fp))
    if mp.exists(): med = json.load(open(mp))
    return fi, med

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

# ── Helpers ────────────────────────────────────────────────
def airline_name(code): return AIRLINE_FULL.get(code, code)
def weather_desc(code): return WEATHER_CODES.get(int(code), f"Code {int(code)}")

def format_time(h):
    if h == 0:    return "12:00 AM"
    elif h < 12:  return f"{h}:00 AM"
    elif h == 12: return "12:00 PM"
    else:         return f"{h-12}:00 PM"

def risk_level(prob):
    if prob < 0.25:   return "LOW RISK",      "#C6F6D5","#276749","🟢","low"
    elif prob < 0.35: return "MODERATE RISK", "#FEFCBF","#744210","🟡","medium"
    else:             return "HIGH RISK",      "#FED7D7","#C53030","🔴","high"

def ontime_pct(prob): return round((1-prob)*100, 1)

def safe_encode(le, value):
    try:
        if value in le.classes_: return int(le.transform([value])[0])
    except: pass
    return 0

def get_airport_coords(code, stats):
    if 'airport_stats' not in stats: return None
    row = stats['airport_stats'][stats['airport_stats']['Airport']==code]
    if not row.empty: return float(row['lat'].iloc[0]), float(row['lon'].iloc[0])
    return None

@st.cache_data(ttl=1800)
def get_live_weather(lat, lon):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude":lat,"longitude":lon,"current_weather":True,
            "hourly":"precipitation,weathercode","forecast_days":1,"timezone":"auto"
        }, timeout=10).json()
        cw = r.get('current_weather', {})
        h  = datetime.now().hour
        pr = r.get('hourly',{}).get('precipitation',[0]*24)
        return {
            'temp_c':        round(cw.get('temperature',20),1),
            'windspeed_kmh': round(cw.get('windspeed',0),1),
            'weathercode':   int(cw.get('weathercode',0)),
            'precip_mm':     round(pr[h] if h<len(pr) else 0, 2),
        }
    except: return None

def get_route_distance(airline, origin, dest, lookup):
    if lookup.empty: return 900
    row = lookup[(lookup['Airline']==airline)&(lookup['Origin']==origin)&(lookup['Dest']==dest)]
    if not row.empty: return int(row['typical_distance'].iloc[0])
    row2 = lookup[(lookup['Origin']==origin)&(lookup['Dest']==dest)]
    if not row2.empty: return int(row2['typical_distance'].iloc[0])
    return 900

def get_available_hours(airline, origin, dest, times):
    if times.empty: return list(range(24))
    rows = times[(times['Airline']==airline)&(times['Origin']==origin)&(times['Dest']==dest)]
    if rows.empty: rows = times[(times['Origin']==origin)&(times['Dest']==dest)]
    return sorted(rows['DepHour'].unique().tolist()) if not rows.empty else list(range(24))

def get_route_avg_delay(origin, dest, stats):
    if 'route_stats' not in stats: return None
    row = stats['route_stats'][(stats['route_stats']['Origin']==origin)&(stats['route_stats']['Dest']==dest)]
    return round(float(row['avg_delay'].iloc[0]),1) if not row.empty else None

def get_route_delay_rate(origin, dest, stats):
    if 'route_stats' not in stats: return None
    row = stats['route_stats'][(stats['route_stats']['Origin']==origin)&(stats['route_stats']['Dest']==dest)]
    return round(float(row['delay_rate'].iloc[0])*100,1) if not row.empty else None

def estimate_passengers(airline, origin, dest, stats):
    regional = ['9E','YX','OH','YV','QX','CP','MQ']
    if airline in regional: return 76
    if 'route_stats' in stats:
        row = stats['route_stats'][(stats['route_stats']['Origin']==origin)&(stats['route_stats']['Dest']==dest)]
        if not row.empty:
            f = int(row['total_flights'].iloc[0])
            return 200 if f>5000 else 160 if f>2000 else 130
    return 150

def estimate_connections(dest, stats):
    hubs = {
        'ATL':12,'ORD':10,'DFW':10,'DEN':9,'LAX':8,'CLT':8,'MIA':7,
        'LAS':6,'PHX':6,'MCO':5,'SEA':5,'EWR':5,'JFK':5,'BOS':4,
        'MSP':4,'DTW':4,'SLC':4,'IAH':4,
    }
    return hubs.get(dest, 2)

def get_badges(dep_hour, weathercode, precip, windspeed, origin, dest, stats):
    badges = []
    wd = weather_desc(weathercode)
    if precip==0 and weathercode<45: badges.append(('green', f'✓ {wd}'))
    elif weathercode>=61:            badges.append(('red',   f'⚠ {wd}'))
    else:                            badges.append(('yellow',f'~ {wd}'))
    if (7<=dep_hour<=9) or (16<=dep_hour<=20): badges.append(('yellow','⚠ Peak Hour Slot'))
    else:                                        badges.append(('green', '✓ Off-Peak Slot'))
    dr = get_route_delay_rate(origin, dest, stats)
    if dr is not None:
        if dr>25:   badges.append(('red',   f'↑ High Delay Route ({dr}%)'))
        elif dr>15: badges.append(('yellow',f'~ Moderate Route ({dr}%)'))
        else:       badges.append(('blue',  f'✓ Reliable Route ({dr}%)'))
    if windspeed>35: badges.append(('red', f'💨 Strong Winds {windspeed}km/h'))
    return badges

def build_features(airline, origin, dest, dep_hour, dow, month,
                   distance, temp_c, windspeed, precip, weathercode,
                   encoders, stats, selected_features):
    le_airline = encoders.get('le_airline')
    le_origin  = encoders.get('le_origin')
    le_dest    = encoders.get('le_dest')
    awd, ala = 0.0, 0.0
    if 'route_stats' in stats:
        row = stats['route_stats'][(stats['route_stats']['Origin']==origin)&(stats['route_stats']['Dest']==dest)]
        if not row.empty: awd = float(row['avg_weather_delay'].iloc[0])
    if 'airline_stats' in stats:
        row = stats['airline_stats'][stats['airline_stats']['Airline']==airline]
        if not row.empty and 'avg_late_aircraft_delay' in stats['airline_stats'].columns:
            ala = float(row['avg_late_aircraft_delay'].iloc[0])
    raw = {
        'DepHour':dep_hour,'day_of_week':dow,'month':month,'distance':distance,
        'IsWeekend':1 if dow>=6 else 0,
        'Airline_enc':safe_encode(le_airline,airline) if le_airline else 0,
        'Origin_enc': safe_encode(le_origin, origin)  if le_origin  else 0,
        'Dest_enc':   safe_encode(le_dest,   dest)    if le_dest    else 0,
        'AvgWeatherDelay_Route':awd,'AvgNASDelay_Origin':0.0,
        'AvgLateAircraft_Airline':ala,'AvgCarrierDelay_Airline':0.0,
        'temp_c':temp_c,'windspeed_kmh':windspeed,'precip_mm':precip,
        'weathercode':float(weathercode),
        'weather_severity':(
            (1 if temp_c<0 else 0)*2+(1 if windspeed>40 else 0)*2+
            (1 if precip>5 else 0)*3+(1 if weathercode>=61 else 0)*2
        ),
        'IsPeakHour':1 if (7<=dep_hour<=9 or 16<=dep_hour<=20) else 0,
    }
    feats = selected_features if selected_features else list(raw.keys())
    return pd.DataFrame([{f: raw.get(f,0) for f in feats}])

def shap_to_english(feature, value, shap_val):
    positive = shap_val > 0
    impact   = abs(shap_val)
    if feature == 'DepHour':
        h, ts = int(value), format_time(int(value))
        if 7<=h<=9:     return f"Departure at {ts} — morning peak congestion window", positive, impact
        elif 16<=h<=20: return f"Departure at {ts} — evening peak, highest delay period of day", positive, impact
        elif 0<=h<=5:   return f"Departure at {ts} — early morning, typically low congestion", positive, impact
        else:           return f"Departure at {ts} — off-peak window, lower congestion", positive, impact
    elif feature == 'temp_c':
        v = round(value,1)
        if v<0:    return f"Freezing temperature ({v}°C) — de-icing risk increases ground time", positive, impact
        elif v>38: return f"Extreme heat ({v}°C) — performance limits may affect operations", positive, impact
        else:      return f"Temperature {v}°C — mild conditions, no thermal impact", positive, impact
    elif feature == 'precip_mm':
        v = round(value,1)
        if v>10:   return f"Heavy precipitation ({v}mm) — significant ground ops disruption", positive, impact
        elif v>5:  return f"Moderate rain ({v}mm) — some ground delays expected", positive, impact
        elif v>0:  return f"Light precipitation ({v}mm) — minor weather impact", positive, impact
        else:      return f"No precipitation at origin — clear conditions for departure", positive, impact
    elif feature == 'weathercode':
        desc = weather_desc(int(value))
        if value>=95:   return f"Thunderstorm conditions — severe weather disruption likely", positive, impact
        elif value>=61: return f"Rain ({desc}) — reduced ground efficiency expected", positive, impact
        elif value>=45: return f"Fog ({desc}) — visibility may affect operations", positive, impact
        else:           return f"Clear skies ({desc}) — weather not a delay factor", positive, impact
    elif feature == 'windspeed_kmh':
        v = round(value,1)
        if v>50:   return f"High winds ({v} km/h) — crosswind limits may apply", positive, impact
        elif v>35: return f"Strong winds ({v} km/h) — minor operational impact", positive, impact
        else:      return f"Calm winds ({v} km/h) — no wind-related delay risk", positive, impact
    elif feature == 'month':
        mn = MONTH_NAMES[int(value)-1] if 1<=int(value)<=12 else str(int(value))
        if value in [6,7,8]:     return f"{mn} — summer peak travel season, higher network load", positive, impact
        elif value in [11,12,1]: return f"{mn} — winter weather risk period", positive, impact
        else:                    return f"{mn} — moderate travel period", positive, impact
    elif feature == 'day_of_week':
        d = DAY_NAMES[int(value)-1] if 1<=int(value)<=7 else str(int(value))
        if value in [5,6,7]: return f"{d} — peak travel day, higher network congestion", positive, impact
        else:                return f"{d} — lower traffic day, favorable for on-time departure", positive, impact
    elif feature == 'distance':
        v = int(value)
        if v>2000:   return f"Long-haul route ({v:,} mi) — small delays compound over distance", positive, impact
        elif v>1000: return f"Medium-haul route ({v:,} mi) — moderate cascade delay risk", positive, impact
        else:        return f"Short-haul route ({v:,} mi) — lower cascade delay impact", positive, impact
    elif feature == 'AvgWeatherDelay_Route':
        v = round(value,1)
        if v>10:  return f"This route averages {v} min weather delay historically — high weather risk", positive, impact
        elif v>3: return f"This route averages {v} min weather delay historically — moderate exposure", positive, impact
        else:     return f"This route has minimal historical weather delays ({v} min avg)", positive, impact
    elif feature == 'AvgLateAircraft_Airline':
        v = round(value,1)
        if v>15:  return f"Airline has high late aircraft history ({v} min avg) — cascade risk elevated", positive, impact
        elif v>5: return f"Airline has moderate late aircraft delays ({v} min avg)", positive, impact
        else:     return f"Airline has low late aircraft delay history — good operational reliability", positive, impact
    elif feature == 'Origin_enc':  return f"Origin airport traffic pattern — based on historical congestion data", positive, impact
    elif feature == 'Dest_enc':    return f"Destination airport pattern — based on historical arrival congestion", positive, impact
    elif feature == 'Airline_enc': return f"Airline operational profile — based on 2024 performance data", positive, impact
    elif feature == 'IsWeekend':
        if value==1: return f"Weekend flight — leisure travel peak, higher load factors", positive, impact
        else:        return f"Weekday flight — business travel pattern", positive, impact
    else:
        d = "increases" if positive else "reduces"
        return f"{feature} {d} delay probability based on historical patterns", positive, impact

def render_shap_bars(explainer, input_df, selected_features):
    sv = explainer.shap_values(input_df)
    if isinstance(sv, list): sv = sv[1]
    vals = sv[0]
    items = []
    for feat, sv_ in zip(selected_features, vals):
        if feat not in input_df.columns: continue
        raw_val = float(input_df[feat].iloc[0])
        desc, is_risk, impact = shap_to_english(feat, raw_val, sv_)
        items.append((desc, is_risk, impact, sv_))
    items.sort(key=lambda x: abs(x[3]), reverse=True)
    top = items[:6]
    mx  = max(abs(x[3]) for x in top) if top else 1
    html = '<div class="why-section"><div class="why-title">🔍 Why this prediction?</div>'
    for desc, is_risk, impact, sv_ in top:
        pct    = int((abs(sv_)/mx)*100)
        color  = '#FC8181' if is_risk else '#68D391'
        contrib= int(abs(sv_)*100)
        icon   = '↑' if is_risk else '↓'
        html  += f"""
        <div class="shap-bar-container">
            <div class="shap-label">{icon} {desc}</div>
            <div style="display:flex;align-items:center;gap:10px;">
                <div class="shap-bar-track" style="flex:1;">
                    <div style="width:{pct}%;height:8px;background:{color};border-radius:6px;"></div>
                </div>
                <div style="font-size:12px;font-weight:700;color:{color};min-width:40px;text-align:right;">{contrib}%</div>
            </div>
        </div>"""
    html += '</div>'
    return html

def get_recommendations(prob, dep_hour, dow, distance, temp_c, windspeed, precip, weathercode):
    recs = []
    if weathercode>=95:   recs.append(("danger","Thunderstorm — evaluate hold or diversion options immediately."))
    elif weathercode>=61: recs.append(("warn","Rain — reduce taxi targets, extend turnaround buffer by 15 mins."))
    elif weathercode>=45: recs.append(("warn","Fog — review ILS approach availability and notify crew."))
    if temp_c<0:          recs.append(("danger",f"Freezing ({temp_c}°C) — mandatory de-icing before pushback."))
    elif temp_c>40:       recs.append(("warn",f"Extreme heat ({temp_c}°C) — check runway performance limits."))
    if windspeed>50:      recs.append(("danger",f"High winds ({windspeed} km/h) — verify crosswind limits."))
    elif windspeed>35:    recs.append(("warn",f"Strong winds ({windspeed} km/h) — review alternate airports."))
    if precip>10:         recs.append(("danger","Heavy precipitation — coordinate ground crew for drainage."))
    elif precip>5:        recs.append(("warn","Moderate precipitation — notify ground crew, check runway."))
    if prob>=0.35:
        recs.append(("danger","HIGH RISK — notify passengers proactively via app, SMS, and gate display."))
        recs.append(("danger","Pre-position gate agents 45 mins before standard boarding time."))
        recs.append(("warn","Review connecting flights on this tail number for cascade delay risk."))
        recs.append(("warn","Alert ground crew to expedite turnaround and baggage handling."))
    elif prob>=0.25:
        recs.append(("warn","MODERATE RISK — monitor closely, prepare contingency staffing plan."))
        recs.append(("warn","Check inbound aircraft status 2 hours before scheduled departure."))
    if dep_hour>=17:  recs.append(("info","Evening departure — network congestion likely, build 15-min buffer."))
    if distance>2000: recs.append(("info","Long-haul — small delays amplify significantly over distance."))
    if not recs:      recs.append(("info","Low risk conditions — standard monitoring procedures are sufficient."))
    return recs

def render_rec(recs):
    html = ""
    for i,(kind,text) in enumerate(recs,1):
        css   = 'rec-box-danger' if kind=='danger' else 'rec-box-warn' if kind=='warn' else 'rec-box'
        emoji = '🔴' if kind=='danger' else '⚠️' if kind=='warn' else 'ℹ️'
        html += f'<div class="{css}"><b>Step {i}:</b> {emoji} {text}</div>'
    return html

def render_risk_header(origin, dest, ontime, prob, risk_key, risk_color, subtitle, badge_html, context_line="DELAY RISK ASSESSMENT"):
    gauge_pct = int(prob*100)
    bg_map    = {'high':'#3a1a1a','medium':'#3a2e1a','low':'#1a3a2a'}
    label_map = {'high':'HIGH RISK','medium':'MODERATE RISK','low':'LOW RISK'}
    icon_map  = {'high':'🔴','medium':'🟡','low':'🟢'}
    return f"""
    <div class="risk-header">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div style="flex:1;">
                <div style="font-size:11px;font-weight:700;color:#718096;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">{context_line}</div>
                <div class="risk-title" style="color:{risk_color};">{ontime}% On-Time Departure</div>
                <div class="risk-subtitle">{subtitle}</div>
                <div style="margin-top:14px;">{badge_html}</div>
            </div>
            <div style="text-align:center;min-width:160px;">
                <div style="font-size:4rem;font-weight:900;color:{risk_color};line-height:1;">{gauge_pct}%</div>
                <div style="font-size:12px;color:#718096;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-top:4px;">DELAY PROBABILITY</div>
                <div style="margin-top:8px;">
                    <span style="background:{bg_map[risk_key]};color:{risk_color};border-radius:20px;padding:4px 14px;font-size:13px;font-weight:700;">
                        {icon_map[risk_key]} {label_map[risk_key]}
                    </span>
                </div>
            </div>
        </div>
    </div>"""

def render_stat_card(label, value, sub, color=None):
    val_style = f'style="color:{color};"' if color else ''
    return f"""
    <div class="card">
        <div class="card-label">{label}</div>
        <div class="card-value" {val_style}>{value}</div>
        <div class="card-sub">{sub}</div>
    </div>"""

# ── Load everything ────────────────────────────────────────
model, metadata, THRESHOLD, SELECTED_FEATURES = load_model_artifacts()
encoders      = load_encoders()
stats         = load_stats()
lookup, times = load_route_data()
fi, medians   = load_explainability()
if model: shap_explainer = get_shap_explainer(model)

airlines = sorted((metadata or {}).get('airline_classes', list(AIRLINE_FULL.keys())))
origins  = sorted((metadata or {}).get('origin_classes',  ['ATL','LAX','ORD','DFW','JFK']))
dests    = sorted((metadata or {}).get('dest_classes',     origins))

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ AeroSight")
    st.markdown("*Flight Delay Decision Support System*")
    st.divider()
    if model:
        st.success("Model loaded")
        bn   = metadata.get('model_name','LightGBM (Optuna)')
        perf = metadata.get('performance',{}).get(bn,{})
        st.markdown(f"**Model:** `{bn}`")
        st.markdown(f"**Threshold:** `{THRESHOLD}`")
        st.markdown(f"**Dataset:** 7M US domestic flights (2024)")
        st.markdown(f"**FAA Standard:** ≥ 15 min = delayed")
        st.divider()
        c1,c2 = st.columns(2)
        c1.metric("Accuracy",  f"{perf.get('accuracy',0)*100:.1f}%")
        c2.metric("AUC",       f"{perf.get('roc_auc',perf.get('auc',0)):.3f}")
        c1.metric("Precision", f"{perf.get('precision',0)*100:.1f}%")
        c2.metric("F1 Score",  f"{perf.get('f1',0):.3f}")
        st.divider()
        st.markdown("🟢 **Low** — below 25%")
        st.markdown("🟡 **Moderate** — 25% to 35%")
        st.markdown("🔴 **High** — above 35%")
        st.divider()
        st.caption("LightGBM · Optuna · SHAP · Open-Meteo · 2024 BTS")
    else:
        st.error("Model not found")

if not model:
    st.error("Model artifacts not found. Run pipeline_v2.py first.")
    st.stop()

st.markdown('<p class="main-title">✈️ AeroSight — Flight Delay DSS</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Prescriptive Analytics · Weather-Aware · SHAP Explainability · 7M Flights</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict Delay",
    "🔧 Prescriptive Engine",
    "🌍 Network Map",
    "📊 Network Dashboard",
    "📈 Model Performance"
])

# ================================================================
# TAB 1 — PREDICT
# ================================================================
with tab1:
    c1,c2,c3 = st.columns(3)
    with c1: p_airline = st.selectbox("Airline", airlines, format_func=lambda x:f"{x} — {airline_name(x)}", key='p_al')
    with c2: p_origin  = st.selectbox("Origin Airport", origins, key='p_or')
    with c3:
        p_dest_opts = [d for d in dests if d!=p_origin] or dests
        p_dest = st.selectbox("Destination", p_dest_opts, key='p_de')

    c4,c5,c6 = st.columns(3)
    with c4:
        p_day = st.selectbox("Day of Week", DAY_NAMES, key='p_dw')
        p_dow = DAY_NAMES.index(p_day)+1
    with c5:
        p_month_n = st.selectbox("Month", MONTH_NAMES, key='p_mo')
        p_month   = MONTH_NAMES.index(p_month_n)+1
    with c6:
        avail = get_available_hours(p_airline, p_origin, p_dest, times)
        tlbls = {format_time(h):h for h in avail}
        p_tl  = st.selectbox("Departure Time", list(tlbls.keys()), key='p_hr')
        p_hour = tlbls[p_tl]

    p_distance = get_route_distance(p_airline, p_origin, p_dest, lookup)
    coords     = get_airport_coords(p_origin, stats)
    live_w     = get_live_weather(coords[0], coords[1]) if coords else None
    w_temp     = live_w['temp_c']        if live_w else 20.0
    w_wind     = live_w['windspeed_kmh'] if live_w else 10.0
    w_precip   = live_w['precip_mm']     if live_w else 0.0
    w_code     = live_w['weathercode']   if live_w else 0

    if st.button("Run Prediction", type="primary", use_container_width=True, key='pred_btn'):
        input_df = build_features(p_airline, p_origin, p_dest, p_hour, p_dow, p_month,
                                   p_distance, w_temp, w_wind, w_precip, w_code,
                                   encoders, stats, SELECTED_FEATURES)
        prob   = float(model.predict_proba(input_df)[0,1])
        ontime = ontime_pct(prob)
        label, bg, fg, icon, rk = risk_level(prob)
        rc     = '#FC8181' if rk=='high' else '#F6AD55' if rk=='medium' else '#68D391'
        badges = get_badges(p_hour, w_code, w_precip, w_wind, p_origin, p_dest, stats)
        bhtml  = "".join(f'<span class="badge badge-{bt}">{bx}</span>' for bt,bx in badges)
        sub    = (f"High delay risk detected — immediate action recommended for {p_origin}→{p_dest}." if rk=='high'
                  else f"Moderate conditions at {p_origin} — monitor closely before departure." if rk=='medium'
                  else f"Low-risk conditions at {p_origin} — flight likely to depart on schedule.")
        avg_delay = get_route_avg_delay(p_origin, p_dest, stats)

        st.divider()
        st.markdown(render_risk_header(p_origin, p_dest, ontime, prob, rk, rc, sub, bhtml), unsafe_allow_html=True)

        # Stats row
        s1,s2,s3,s4 = st.columns(4)
        with s1: st.markdown(render_stat_card("DEPARTS", p_tl, f"{p_day} · {p_month_n} 2024"), unsafe_allow_html=True)
        with s2: st.markdown(render_stat_card("ROUTE DISTANCE", f"{p_distance:,} mi",
                              "Long haul" if p_distance>2000 else "Medium haul" if p_distance>1000 else "Short haul"), unsafe_allow_html=True)
        with s3:
            dv = f"{avg_delay} min" if avg_delay else "N/A"
            dc = '#FC8181' if avg_delay and avg_delay>20 else '#F6AD55' if avg_delay and avg_delay>10 else '#68D391'
            st.markdown(render_stat_card("AVG DELAY (ROUTE)", dv, "Historical average", dc), unsafe_allow_html=True)
        with s4: st.markdown(render_stat_card(f"WEATHER AT {p_origin}", f"{w_temp}°C",
                              f"{weather_desc(w_code)} · {w_wind}km/h wind"), unsafe_allow_html=True)

        st.divider()

        # Route card + SHAP side by side
        rp_col, why_col = st.columns([1,1])

        with rp_col:
            st.markdown(f"""
            <div class="route-card">
                <div style="display:flex;align-items:center;justify-content:center;margin-bottom:20px;">
                    <div style="text-align:center;">
                        <div class="route-airport">{p_origin}</div>
                        <div class="route-city">{p_origin} Airport</div>
                    </div>
                    <div style="margin:0 24px;text-align:center;">
                        <div style="color:#4299e1;font-size:1.5rem;">→</div>
                        <div style="font-size:11px;color:#718096;margin-top:4px;">{p_distance:,} mi</div>
                    </div>
                    <div style="text-align:center;">
                        <div class="route-airport">{p_dest}</div>
                        <div class="route-city">{p_dest} Airport</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div style="background:#1a2035;border-radius:10px;padding:12px;text-align:center;">
                        <div style="font-size:11px;color:#718096;font-weight:600;">DEPARTS</div>
                        <div style="font-size:18px;font-weight:800;color:#e2e8f0;margin-top:4px;">{p_tl}</div>
                    </div>
                    <div style="background:#1a2035;border-radius:10px;padding:12px;text-align:center;">
                        <div style="font-size:11px;color:#718096;font-weight:600;">OPERATED BY</div>
                        <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-top:4px;">{p_airline} — {airline_name(p_airline)}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        with why_col:
            with st.spinner("Analysing prediction factors..."):
                st.markdown(render_shap_bars(shap_explainer, input_df, SELECTED_FEATURES), unsafe_allow_html=True)

        # Recommendations BELOW both columns
        st.divider()
        st.markdown('<p class="section-header">Operational Recommendations</p>', unsafe_allow_html=True)
        recs = get_recommendations(prob, p_hour, p_dow, p_distance, w_temp, w_wind, w_precip, w_code)
        st.markdown(render_rec(recs), unsafe_allow_html=True)

# ================================================================
# TAB 2 — PRESCRIPTIVE ENGINE
# ================================================================
with tab2:
    st.markdown('<p class="section-header">Prescriptive Action Engine</p>', unsafe_allow_html=True)
    st.caption("Select a flight — get a full risk assessment and action plan")

    pe1,pe2,pe3 = st.columns(3)
    with pe1: pe_airline = st.selectbox("Airline", airlines, format_func=lambda x:f"{x} — {airline_name(x)}", key='pe_al')
    with pe2: pe_origin  = st.selectbox("Origin", origins, key='pe_or')
    with pe3:
        pe_dest_opts = [d for d in dests if d!=pe_origin] or dests
        pe_dest = st.selectbox("Destination", pe_dest_opts, key='pe_de')

    pe4,pe5,pe6 = st.columns(3)
    with pe4:
        pe_day = st.selectbox("Day of Week", DAY_NAMES, key='pe_dw')
        pe_dow = DAY_NAMES.index(pe_day)+1
    with pe5:
        pe_month_n = st.selectbox("Month", MONTH_NAMES, key='pe_mo')
        pe_month   = MONTH_NAMES.index(pe_month_n)+1
    with pe6:
        pe_hrs  = get_available_hours(pe_airline, pe_origin, pe_dest, times)
        pe_lbls = {format_time(h):h for h in pe_hrs}
        pe_tl   = st.selectbox("Departure Time", list(pe_lbls.keys()), key='pe_hr')
        pe_hour = pe_lbls[pe_tl]

    # Auto-estimate passengers and connections — shown as info cards
    pe_pax_auto  = estimate_passengers(pe_airline, pe_origin, pe_dest, stats)
    pe_conn_auto = estimate_connections(pe_dest, stats)

    ai1, ai2 = st.columns(2)
    with ai1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">PASSENGERS ON BOARD (AUTO-ESTIMATED)</div>
            <div class="card-value">{pe_pax_auto}</div>
            <div class="card-sub">Based on {pe_airline} aircraft type on this route</div>
        </div>""", unsafe_allow_html=True)
        pe_pax = pe_pax_auto
    with ai2:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">CONNECTING FLIGHTS AT RISK (AUTO-ESTIMATED)</div>
            <div class="card-value">{pe_conn_auto}</div>
            <div class="card-sub">Based on {pe_dest} hub size and traffic volume</div>
        </div>""", unsafe_allow_html=True)
        pe_conn = pe_conn_auto

    # Auto fill weather + distance
    pe_distance = get_route_distance(pe_airline, pe_origin, pe_dest, lookup)
    pe_coords   = get_airport_coords(pe_origin, stats)
    pe_live_w   = get_live_weather(pe_coords[0], pe_coords[1]) if pe_coords else None
    pe_temp     = pe_live_w['temp_c']        if pe_live_w else 20.0
    pe_wind     = pe_live_w['windspeed_kmh'] if pe_live_w else 10.0
    pe_precip   = pe_live_w['precip_mm']     if pe_live_w else 0.0
    pe_wcode    = pe_live_w['weathercode']   if pe_live_w else 0

    if st.button("Generate Action Plan", type="primary", use_container_width=True, key='pe_btn'):
        input_df = build_features(pe_airline, pe_origin, pe_dest, pe_hour, pe_dow, pe_month,
                                   pe_distance, pe_temp, pe_wind, pe_precip, pe_wcode,
                                   encoders, stats, SELECTED_FEATURES)
        prob   = float(model.predict_proba(input_df)[0,1])
        ontime = ontime_pct(prob)
        label, bg, fg, icon, rk = risk_level(prob)
        rc     = '#FC8181' if rk=='high' else '#F6AD55' if rk=='medium' else '#68D391'
        badges = get_badges(pe_hour, pe_wcode, pe_precip, pe_wind, pe_origin, pe_dest, stats)
        bhtml  = "".join(f'<span class="badge badge-{bt}">{bx}</span>' for bt,bx in badges)
        sub    = f"{pe_airline} · {pe_tl} · {pe_day} {pe_month_n} 2024"

        delay_min = 45 if prob>=0.35 else 20 if prob>=0.25 else 5
        total     = delay_min*74 + pe_conn*150*0.1*pe_pax
        savings   = total*0.6

        st.divider()
        st.markdown(render_risk_header(
            pe_origin, pe_dest, ontime, prob, rk, rc, sub, bhtml,
            f"ACTION PLAN — {pe_origin} → {pe_dest}"
        ), unsafe_allow_html=True)

        s1,s2,s3,s4 = st.columns(4)
        with s1: st.markdown(render_stat_card("PASSENGERS", pe_pax, "At risk of delay"), unsafe_allow_html=True)
        with s2: st.markdown(render_stat_card("CONNECTING FLIGHTS", pe_conn, "Cascade risk flights",
                              '#FC8181' if pe_conn>5 else '#F6AD55' if pe_conn>2 else '#68D391'), unsafe_allow_html=True)
        with s3: st.markdown(render_stat_card("EST. COST IF NO ACTION", f"${total:,.0f}",
                              f"Based on {delay_min} min expected delay", '#FC8181'), unsafe_allow_html=True)
        with s4: st.markdown(render_stat_card("POTENTIAL SAVINGS", f"${savings:,.0f}",
                              "With early intervention", '#68D391'), unsafe_allow_html=True)

        st.divider()
        ac1, ac2 = st.columns([1,1])

        with ac1:
            st.markdown('<p class="section-header">Immediate Action Plan</p>', unsafe_allow_html=True)
            recs = get_recommendations(prob, pe_hour, pe_dow, pe_distance, pe_temp, pe_wind, pe_precip, pe_wcode)
            st.markdown(render_rec(recs), unsafe_allow_html=True)
            st.divider()
            st.markdown('<p class="section-header">Recommended Departure Window</p>', unsafe_allow_html=True)
            if prob>=0.35:
                if pe_precip>5 or pe_wcode>=95: st.error("Delay departure 45–60 mins — severe weather conditions")
                elif pe_temp<0:                  st.warning("Allow 30 mins extra for mandatory de-icing")
                else:                            st.warning("Delay departure 20–30 mins — high network risk")
            elif prob>=0.25: st.warning("Proceed with caution — build 10-min buffer into schedule")
            else:            st.success("Proceed on schedule — conditions acceptable")

        with ac2:
            with st.spinner("Analysing risk factors..."):
                st.markdown(render_shap_bars(shap_explainer, input_df, SELECTED_FEATURES), unsafe_allow_html=True)

# ================================================================
# TAB 3 — NETWORK MAP
# ================================================================
with tab3:
    st.markdown('<p class="section-header">Continental US Airport Delay Risk Map</p>', unsafe_allow_html=True)
    if 'airport_stats' not in stats:
        st.warning("Airport stats not found. Run pipeline_v2.py first.")
    else:
        ast = stats['airport_stats'].copy()
        ast = ast[(ast['lat']>=CONUS_LAT[0])&(ast['lat']<=CONUS_LAT[1])&
                  (ast['lon']>=CONUS_LON[0])&(ast['lon']<=CONUS_LON[1])].copy()
        ast['delay_pct'] = ast['delay_rate']*100
        ast['size']      = np.log1p(ast['total_flights'])*3
        fig_map = px.scatter_geo(
            ast, lat='lat', lon='lon', color='delay_pct', size='size',
            hover_name='Airport',
            hover_data={'delay_pct':':.1f','avg_delay':':.1f','avg_temp':':.1f',
                        'total_flights':':,','lat':False,'lon':False,'size':False},
            color_continuous_scale='RdYlGn_r', range_color=[0,40],
            title="Airport Delay Risk — Bubble Size = Volume | Color = Delay Rate %",
            labels={'delay_pct':'Delay Rate (%)'}
        )
        fig_map.update_layout(
            height=580,
            geo=dict(scope='usa', showframe=False, showcoastlines=True,
                     showland=True, landcolor='#1a1a2e', bgcolor='rgba(0,0,0,0)',
                     showlakes=True, lakecolor='#16213e', projection_type='albers usa'),
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(title="Delay %")
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.divider()
        m1,m2,m3 = st.columns(3)
        with m1:
            st.markdown("#### Worst Delay Airports")
            st.dataframe(ast.nlargest(10,'delay_rate')[['Airport','delay_pct','avg_delay']]
                         .rename(columns={'delay_pct':'Delay %','avg_delay':'Avg Delay (min)'}),
                         use_container_width=True, hide_index=True)
        with m2:
            st.markdown("#### Best On-Time Airports")
            st.dataframe(ast[ast['total_flights']>1000].nsmallest(10,'delay_rate')[['Airport','delay_pct','avg_delay']]
                         .rename(columns={'delay_pct':'Delay %','avg_delay':'Avg Delay (min)'}),
                         use_container_width=True, hide_index=True)
        with m3:
            st.markdown("#### Temperature vs Delay")
            fig_c = px.scatter(ast.dropna(subset=['avg_temp','delay_pct']),
                               x='avg_temp', y='delay_pct', size='size', color='delay_pct',
                               hover_name='Airport', color_continuous_scale='RdYlGn_r',
                               labels={'avg_temp':'Avg Temp (°C)','delay_pct':'Delay %'},
                               trendline='ols')
            fig_c.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_c, use_container_width=True)

# ================================================================
# TAB 4 — NETWORK DASHBOARD
# ================================================================
with tab4:
    st.markdown('<p class="section-header">Network Operations Dashboard</p>', unsafe_allow_html=True)
    if 'airline_stats' in stats:
        df_a = stats['airline_stats']
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Airlines Tracked",   len(df_a))
        k2.metric("Network Delay Rate", f"{df_a['delay_rate'].mean()*100:.1f}%")
        k3.metric("Best Carrier",        df_a.loc[df_a['delay_rate'].idxmin(),'Airline'])
        k4.metric("Most Delayed",        df_a.loc[df_a['delay_rate'].idxmax(),'Airline'])
        st.divider()
        df_a2 = df_a.copy()
        df_a2['Name'] = df_a2['Airline'].apply(lambda x: f"{x} — {airline_name(x)}")
        df_a2 = df_a2.sort_values('delay_rate')
        col_l,col_r = st.columns(2)
        with col_l:
            fig_al = go.Figure(go.Bar(
                x=df_a2['delay_rate']*100, y=df_a2['Name'], orientation='h',
                marker=dict(color=df_a2['delay_rate']*100, colorscale='RdYlGn_r',
                            showscale=True, colorbar=dict(title='Delay %')),
                text=[f"{v*100:.1f}%" for v in df_a2['delay_rate']], textposition='outside'
            ))
            fig_al.update_layout(title='Delay Rate by Airline', xaxis_title='Delay Rate (%)',
                                  height=450, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_al, use_container_width=True)
        with col_r:
            fig_sc = px.scatter(df_a2, x='total_flights', y='avg_delay',
                                size='total_flights', color='delay_rate', hover_name='Name',
                                color_continuous_scale='RdYlGn_r',
                                labels={'total_flights':'Total Flights','avg_delay':'Avg Delay (min)'},
                                title='Flight Volume vs Average Delay')
            fig_sc.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_sc, use_container_width=True)
        st.divider()

    if 'monthly_weather_stats' in stats:
        df_mw = stats['monthly_weather_stats']
        fig_mw = go.Figure()
        fig_mw.add_trace(go.Bar(x=df_mw['month'], y=df_mw['delay_rate']*100,
                                name='Delay Rate %', marker_color='#E53E3E', opacity=0.8))
        fig_mw.add_trace(go.Scatter(x=df_mw['month'], y=df_mw['avg_temp'],
                                    name='Avg Temp (°C)', line=dict(color='#63B3ED',width=2.5),
                                    mode='lines+markers', yaxis='y2'))
        fig_mw.add_trace(go.Scatter(x=df_mw['month'], y=df_mw['avg_precip']*10,
                                    name='Precipitation x10', line=dict(color='#68D391',width=2,dash='dot'),
                                    mode='lines+markers', yaxis='y2'))
        fig_mw.update_layout(
            title='Monthly Delay Rate vs Weather',
            xaxis=dict(tickvals=list(range(1,13)), ticktext=MONTH_NAMES),
            yaxis=dict(title='Delay Rate (%)', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis2=dict(title='Temp / Precip', overlaying='y', side='right'),
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=1.1), hovermode='x unified'
        )
        st.plotly_chart(fig_mw, use_container_width=True)

    if 'hourly_stats' in stats:
        df_h = stats['hourly_stats']
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=df_h['DepHour'], y=df_h['delay_rate']*100,
                                   mode='lines+markers', fill='tozeroy',
                                   line=dict(color='#E53E3E',width=2.5),
                                   marker=dict(size=8,color='#E53E3E'),
                                   fillcolor='rgba(229,62,62,0.15)'))
        fig_h.add_vrect(x0=7,x1=9,fillcolor='#D69E2E',opacity=0.12,
                        annotation_text="AM Peak",annotation_font_color='#F6AD55')
        fig_h.add_vrect(x0=16,x1=20,fillcolor='#D69E2E',opacity=0.12,
                        annotation_text="PM Peak",annotation_font_color='#F6AD55')
        fig_h.update_layout(
            title='Delay Rate by Hour of Day',
            xaxis=dict(title='Departure Hour', tickvals=list(range(0,24,2)),
                       ticktext=[format_time(h) for h in range(0,24,2)], tickangle=-30),
            yaxis_title='Delay Rate (%)', height=360,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified'
        )
        st.plotly_chart(fig_h, use_container_width=True)

    if 'weather_stats' in stats:
        df_ws = stats['weather_stats'].copy()
        df_ws['condition'] = df_ws['weathercode'].apply(weather_desc)
        df_ws = df_ws[df_ws['total_flights']>100].sort_values('delay_rate',ascending=False).head(12)
        fig_wc = go.Figure(go.Bar(
            x=df_ws['condition'], y=df_ws['delay_rate']*100,
            marker=dict(color=df_ws['delay_rate']*100, colorscale='RdYlGn_r',
                        showscale=True, colorbar=dict(title='Delay %')),
            text=[f"{v*100:.1f}%" for v in df_ws['delay_rate']], textposition='outside'
        ))
        fig_wc.update_layout(title='Delay Rate by Weather Condition', xaxis_tickangle=-30,
                              yaxis_title='Delay Rate (%)', height=400,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wc, use_container_width=True)

    if 'route_stats' in stats:
        df_r = stats['route_stats'].copy()
        df_r['Route'] = df_r['Origin']+' → '+df_r['Dest']
        top_r = df_r.sort_values('delay_rate',ascending=False).head(15)
        fig_r = go.Figure(go.Bar(
            x=top_r['Route'], y=top_r['delay_rate']*100,
            marker=dict(color=top_r['avg_delay'], colorscale='YlOrRd',
                        showscale=True, colorbar=dict(title='Avg Delay (min)')),
            text=[f"{v*100:.1f}%" for v in top_r['delay_rate']], textposition='outside'
        ))
        fig_r.update_layout(title='Top 15 Highest-Risk Routes', xaxis_tickangle=-40,
                             yaxis_title='Delay Rate (%)', height=420,
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_r, use_container_width=True)

# ================================================================
# TAB 5 — MODEL PERFORMANCE
# ================================================================
with tab5:
    st.markdown('<p class="section-header">Model Evaluation & Justification</p>', unsafe_allow_html=True)
    perf_all  = metadata.get('performance',{})
    best_name = metadata.get('model_name','LightGBM (Optuna)')
    if perf_all:
        st.markdown("#### All Models Compared")
        rows = [{'Model':n,'Accuracy':f"{m.get('accuracy',0)*100:.2f}%",
                 'Precision':f"{m.get('precision',0)*100:.2f}%",
                 'Recall':f"{m.get('recall',0)*100:.2f}%",
                 'F1':f"{m.get('f1',0):.4f}",
                 'AUC':f"{m.get('roc_auc',m.get('auc',0)):.4f}",
                 'Threshold':f"{m.get('threshold',THRESHOLD):.2f}"}
                for n,m in perf_all.items()]
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)
    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
**Why LightGBM?**
- Best AUC (0.7247) across all tested models
- Optuna hyperparameter tuning — 50 trials
- `scale_pos_weight` handles 80/20 class imbalance natively
- SHAP-compatible for full prediction explainability
- 10x faster training than Random Forest on 7M rows

**Why threshold = 0.70?**
- Optimised via full threshold sweep (0.10 → 0.90)
- Catches 55% of actual delays (recall)
- Ops managers prefer fewer false alarms
        """)
    with c2:
        st.markdown("""
**Risk Thresholds:**

| Level | Probability | Action |
|---|---|---|
| 🟢 Low | < 25% | Standard monitoring |
| 🟡 Moderate | 25–35% | Heightened watch |
| 🔴 High | > 35% | Immediate action |

**Class Imbalance Fix:**
- 20.6% delayed vs 79.4% on-time
- Fix: Undersample + oversample to 50/50
- Evaluated on real-world distribution
        """)
    st.divider()
    if fi:
        fi_df = pd.Series(fi).sort_values(ascending=True)
        fi_df.index = [FEATURE_LABELS.get(i,i) for i in fi_df.index]
        fig_fi = go.Figure(go.Bar(
            x=fi_df.values, y=fi_df.index, orientation='h',
            marker=dict(color=fi_df.values, colorscale='Blues', showscale=False),
            text=[f"{v*100:.1f}%" for v in fi_df.values], textposition='outside'
        ))
        fig_fi.update_layout(title=f"Feature Importance — {best_name}",
                              xaxis_title="Importance Score", height=380,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=10,r=80,t=40,b=20))
        st.plotly_chart(fig_fi, use_container_width=True)
    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Flights",   "6,964,246")
    c2.metric("Features Used",   len(SELECTED_FEATURES))
    c3.metric("Weather Features","4")
    c4.metric("Training Method", "Balanced 50/50 + Optuna")