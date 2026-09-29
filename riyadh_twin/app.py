# -*- coding: utf-8 -*-
"""
التوأم الرقمي لمنظومة الإسعاف والطوارئ في مدينة الرياض
Python → النموذج الرقمي → SimPy → خوارزمية القرار → Streamlit → خريطة الرياض التفاعلية
تشغيل:  streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:  # الخريطة الاحتياطية (Plotly) تعمل بدون folium
    HAS_FOLIUM = False

from data import (SIM_LABEL, RIYADH_CENTER, INCIDENT_TYPES, SEVERITIES, SPECIALTIES, ROAD_CONDITIONS,
                  LOCATIONS, LEVEL_COLOR, SEVERITY_COLOR, STATUS_COLOR, OUT_OF_SERVICE, DEMO_SPEC,
                  create_digital_twin, apply_demo_state, add_incident, pending_incidents,
                  default_incident_spec, occupancy_series, pressure_level, UNDER_PRESSURE_THRESHOLD)
from recommendation import recommend_hospital, WEIGHTS, WEIGHT_LABELS, DISCLAIMER
from simulation import (STRATEGIES, WHAT_IFS, simulate_all, simulate_what_if, twin_metrics, preview_dispatch,
                        candidate_ambulances, timeline_for, calculate_system_readiness, readiness_label)

FONT = "IBM Plex Sans Arabic, Tajawal, Tahoma, sans-serif"
NAVY, INK, PAPER, LINE = "#0F2A43", "#16283B", "#F0F3F7", "#D9E1EA"
SCEN_COLOR = {"nearest": "#E0692B", "capacity": "#2C6FBB", "balanced": "#0F8B8D"}

st.set_page_config(page_title="التوأم الرقمي لمنظومة الإسعاف والطوارئ — الرياض", page_icon="🚑",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# التنسيق (RTL + مظهر مركز عمليات)
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
:root {{
  --navy:{NAVY}; --ink:{INK}; --muted:#5B6B7C; --line:{LINE}; --bg:{PAPER}; --surface:#FFFFFF;
  --teal:#0F8B8D; --teal-soft:#E3F3F3; --red:#C8372D; --amber:#E2A31B; --blue:#2C6FBB; --r:8px;
}}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select, [data-baseweb] {{
  font-family: {FONT} !important; }}
.stApp {{ direction: rtl; background: var(--bg); color: var(--ink); line-height: 1.7; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.1rem; max-width: 1500px; }}
h1, h2, h3, h4, p, label, div {{ text-align: right; }}
h1, h2, h3, h4 {{ color: var(--navy); font-weight: 700; letter-spacing: 0; }}
h2 {{ font-size: 1.45rem; margin-top: 1.4rem; padding-inline-start: 12px; border-inline-start: 5px solid var(--teal); }}
h3 {{ font-size: 1.2rem; }}
[data-testid="stPlotlyChart"], iframe, [data-testid="stIFrame"] {{ direction: ltr; }}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], iframe {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); overflow: hidden; }}
[data-testid="stDataFrame"] {{ direction: rtl; }}

/* الشريط الجانبي */
[data-testid="stSidebar"] {{ direction: rtl; background: var(--surface); border-inline-start: 1px solid var(--line); }}
[data-testid="stSidebar"] h3 {{ background: var(--navy); color: #fff; padding: 10px 14px; border-radius: var(--r); font-size: 1.1rem; }}
[data-testid="stSidebar"] label p {{ font-weight: 600; color: var(--navy); font-size: .95rem; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ border-radius: 6px !important; border-color: var(--line) !important; background: #F8FAFC !important; }}
[data-testid="stSlider"] [role="slider"] {{ background: var(--teal) !important; }}

/* الأزرار */
div.stButton > button {{ width: 100%; border-radius: var(--r); font-weight: 600; border: 1.5px solid var(--navy);
  color: var(--navy); background: var(--surface); min-height: 2.6rem; transition: background .15s, color .15s; }}
div.stButton > button:hover {{ background: var(--navy); color: #fff; border-color: var(--navy); }}
div.stButton > button:focus-visible {{ outline: 3px solid var(--teal); outline-offset: 2px; }}
div.stButton > button[kind="primary"] {{ background: var(--red); border-color: var(--red); color: #fff; }}
div.stButton > button[kind="primary"]:hover {{ background: #A82C24; border-color: #A82C24; }}
.st-key-demo button {{ background: var(--red) !important; border-color: var(--red) !important; color: #fff !important;
  font-size: 1.25rem !important; font-weight: 700 !important; min-height: 3.5rem; }}
.st-key-demo button:hover {{ background: #A82C24 !important; }}
.st-key-runsim button {{ background: var(--navy) !important; border-color: var(--navy) !important; color: #fff !important;
  font-size: 1.15rem !important; font-weight: 700 !important; min-height: 3.5rem; }}
.st-key-runsim button:hover {{ background: #1B4468 !important; }}

/* تبويبات وقوائم وتنبيهات */
[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--line); }}
[data-baseweb="tab"] {{ font-weight: 600; color: var(--muted); padding: 10px 16px; }}
[data-baseweb="tab"][aria-selected="true"] {{ color: var(--navy); }}
[data-baseweb="tab-highlight"] {{ background: var(--teal) !important; height: 3px; }}
[data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--line) !important; border-radius: var(--r); }}
[data-testid="stAlert"] {{ border-radius: var(--r); }}
[data-testid="stStatusWidget"], [data-testid="stStatus"] {{ border-radius: var(--r); border: 1px solid var(--line); background: var(--surface); }}
[data-testid="stRadio"] label p {{ font-weight: 500; }}

/* الترويسة */
.hero {{ background: var(--navy); color: #fff; padding: 22px 28px 20px; border-radius: var(--r); margin-bottom: 12px;
  border-bottom: 5px solid var(--teal); }}
.hero h1 {{ color: #fff; font-size: 2.1rem; margin: 0 0 2px 0; line-height: 1.35; }}
.hero .sub {{ color: #C4D3E3; font-size: 1.08rem; margin-bottom: 14px; font-weight: 400; }}
.badge {{ display: inline-flex; align-items: center; gap: 8px; background: var(--amber); color: #2B1B00;
  font-weight: 700; padding: 4px 14px; border-radius: 999px; font-size: .9rem; margin-inline-end: 10px; }}
.badge i {{ width: 9px; height: 9px; border-radius: 50%; background: var(--red); display: inline-block; animation: blink 1.4s ease-in-out infinite; }}
@keyframes blink {{ 50% {{ opacity: .25; }} }}
@media (prefers-reduced-motion: reduce) {{ .badge i {{ animation: none; }} div.stButton > button {{ transition: none; }} }}
.simlabel {{ display: inline-block; border: 1px solid rgba(255,255,255,.45); color: #E6EEF6; padding: 3px 12px; border-radius: 4px; font-size: .86rem; }}
.explain {{ border-inline-start: 5px solid var(--blue); background: var(--surface); padding: 10px 16px; margin: 0 0 14px 0;
  color: #34495E; border-radius: var(--r); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}

/* البطاقات */
.kpi {{ background: var(--surface); border: 1px solid var(--line); border-inline-start: 7px solid var(--c); border-radius: var(--r); padding: 14px 18px; min-height: 132px; }}
.kpi .l {{ color: var(--muted); font-size: 1rem; font-weight: 500; }}
.kpi .v {{ color: var(--navy); font-size: 2.8rem; font-weight: 700; line-height: 1.2; font-variant-numeric: tabular-nums; }}
.kpi .v small {{ font-size: 1rem; font-weight: 500; color: var(--muted); margin-inline-start: 6px; }}
.kpi .d {{ font-size: .88rem; color: var(--dc, var(--muted)); font-weight: 600; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: 16px 18px; margin-bottom: 12px; }}
.card.rec {{ border-inline-start: 7px solid var(--teal); }}
.card h4 {{ margin: 0 0 6px 0; font-size: 1.05rem; color: var(--muted); font-weight: 600; }}
.recname {{ font-size: 1.65rem; font-weight: 700; color: var(--navy); line-height: 1.4; }}
.recscore {{ font-size: 1.25rem; font-weight: 700; color: var(--teal); font-variant-numeric: tabular-nums; }}
.reason {{ margin: 3px 0; font-size: .98rem; }}
.reason.ok::before {{ content: "✓ "; color: var(--teal); font-weight: 700; }}
.reason.warn::before {{ content: "⚠ "; color: #E0692B; font-weight: 700; }}
.disc {{ margin-top: 10px; padding: 6px 10px; background: #FFF6DD; border: 1px solid #F0D38A; border-radius: 6px; font-size: .88rem; color: #5B4300; }}
.chip {{ display: inline-block; padding: 1px 10px; border-radius: 4px; font-weight: 700; font-size: .85rem; color: #fff; background: var(--c); }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: 10px 14px; }}
.stat .l {{ color: var(--muted); font-size: .88rem; }}
.stat .v {{ font-size: 1.45rem; font-weight: 700; color: var(--navy); font-variant-numeric: tabular-nums; }}

/* الخط الزمني */
.tl {{ position: relative; margin: 6px 0; padding-inline-start: 26px; border-inline-start: 3px solid var(--line); }}
.tl .it {{ position: relative; padding: 0 0 14px 0; }}
.tl .it::before {{ content: ""; position: absolute; inset-inline-start: -35px; top: 8px; width: 13px; height: 13px; background: var(--teal); border-radius: 50%; border: 3px solid var(--bg); }}
.tl .tm {{ font-weight: 700; color: var(--navy); font-size: 1.15rem; font-variant-numeric: tabular-nums; }}
.tl .tx {{ color: #34495E; }}

/* لماذا توأم رقمي */
.flow {{ display: flex; flex-direction: column; align-items: stretch; }}
.flow .n {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: 10px 14px; text-align: center; font-weight: 600; color: var(--navy); }}
.flow .n.reality {{ background: var(--navy); color: #fff; border-color: var(--navy); }}
.flow .n.result {{ background: var(--teal); color: #fff; border-color: var(--teal); }}
.flow .a {{ text-align: center; color: var(--muted); font-size: 1.3rem; line-height: 1.25; }}
.msg {{ background: var(--navy); color: #fff; border-radius: var(--r); padding: 16px 22px; font-size: 1.2rem; font-weight: 600; margin: 10px 0 14px 0; border-inline-start: 7px solid var(--teal); }}
.ba {{ background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: 12px 16px; }}
.ba.after {{ border-inline-start: 7px solid var(--teal); }} .ba.before {{ border-inline-start: 7px solid #5B6B7C; }}
.ba .r {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dashed var(--line); }}
</style>
""", unsafe_allow_html=True)

# قالب Plotly موحّد (خط، ألوان، شبكة)
import plotly.io as pio
pio.templates["ops"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=INK, size=13),
    colorway=["#0F8B8D", "#2C6FBB", "#E0692B", "#E2A31B", "#6B4FA3", "#5B6B7C"],
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
    xaxis=dict(gridcolor="#E6ECF2", linecolor="#D9E1EA", zerolinecolor="#D9E1EA"),
    yaxis=dict(gridcolor="#E6ECF2", linecolor="#D9E1EA", zerolinecolor="#D9E1EA"),
    title=dict(font=dict(size=16, color=NAVY)),
))
pio.templates.default = "ops"


# ----------------------------------------------------------------------------
# الحالة
# ----------------------------------------------------------------------------
VIEW_NOW = "الحالة الحالية"


def fmt_min(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}"


def init_state():
    ss = st.session_state
    if "base" not in ss:
        ss.base = create_digital_twin()      # الحالة قبل أي بلاغ جديد
        ss.twin = ss.base                    # الحالة مع البلاغ الجديد (إن وجد)
        ss.incident_id = None
        ss.results = None
        ss.whatif = None
        ss.demo_ran = False
        ss.setdefault("inc_type", "حادث مروري")
        ss.setdefault("severity", "مرتفع")
        ss.setdefault("n_injured", 4)
        ss.setdefault("n_critical", 1)
        ss.setdefault("specialty", "إصابات وحوادث")
        ss.setdefault("road_cond", "طبيعية")
        ss.setdefault("place", "النخيل")


def spec_from_state():
    ss = st.session_state
    return dict(type=ss.inc_type, severity=ss.severity, patients=ss.n_injured, critical=ss.n_critical,
                specialty=ss.specialty, place=ss.place, road_condition=ss.road_cond)


def create_incident(base):
    ss = st.session_state
    twin = add_incident(base, spec_from_state())
    ss.twin = twin
    ss.incident_id = twin["incidents"].iloc[-1]["id"]
    ss.results = None
    ss.whatif = None
    ss.view = VIEW_NOW


def start_demo():
    """callback: يضبط عناصر الشريط الجانبي ثم يترك المحاكاة للتشغيل التالي."""
    ss = st.session_state
    ss.inc_type, ss.severity = DEMO_SPEC["type"], DEMO_SPEC["severity"]
    ss.n_injured, ss.n_critical = DEMO_SPEC["patients"], DEMO_SPEC["critical"]
    ss.specialty, ss.road_cond, ss.place = DEMO_SPEC["specialty"], DEMO_SPEC["road_condition"], DEMO_SPEC["place"]
    ss.run_demo = True


def reset_all():
    for k in ("base", "results", "whatif", "run_demo", "view"):
        st.session_state.pop(k, None)


def current_incident():
    ss = st.session_state
    if ss.incident_id is None:
        return None
    df = ss.twin["incidents"]
    row = df[df["id"] == ss.incident_id]
    return row.iloc[0] if len(row) else None


def display_twin():
    """التوأم المعروض: الحالة الحالية (مع معاينة الإرسال) أو نتيجة محاكاة/ماذا لو."""
    ss = st.session_state
    if ss.results and ss.get("view", VIEW_NOW) != VIEW_NOW:
        key = [k for k, v in STRATEGIES.items() if f"بعد: {v}" == ss.view]
        if key:
            return ss.results[key[0]]["twin_after"]
    if ss.whatif:
        return preview_dispatch(ss.whatif["twin_modified"])
    return preview_dispatch(ss.twin)


# ----------------------------------------------------------------------------
# الرسم والعرض
# ----------------------------------------------------------------------------
def kpi_card(label, value, unit="", color="#2C6FBB", delta=None, good_when_down=True):
    d = ""
    if delta is not None and delta != 0:
        up = delta > 0
        bad = up if good_when_down else not up
        col = "#C8372D" if bad else "#0F8B8D"
        d = f'<div class="d" style="--dc:{col}">{"▲" if up else "▼"} {abs(delta):.1f}'.replace(".0", "") + " عن الوضع السابق</div>"
    else:
        d = '<div class="d">&nbsp;</div>'
    return f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div><div class="v">{value}<small>{unit}</small></div>{d}</div>'


def render_kpis(m, m0):
    c = st.columns(4)
    resp = m["avg_response"]
    items = [
        kpi_card("البلاغات النشطة", m["active_incidents"], "", "#C8372D", m["active_incidents"] - m0["active_incidents"]),
        kpi_card("سيارات الإسعاف المتاحة", m["available_ambulances"], "", "#0F8B8D",
                 m["available_ambulances"] - m0["available_ambulances"], good_when_down=False),
        kpi_card("المستشفيات تحت الضغط", m["under_pressure"], "", "#E0692B", m["under_pressure"] - m0["under_pressure"]),
        kpi_card("متوسط زمن الاستجابة", fmt_min(resp), "دقيقة", "#2C6FBB",
                 None if np.isnan(resp) else resp - m0["avg_response"]),
    ]
    for col, html in zip(c, items):
        col.markdown(html, unsafe_allow_html=True)


def _divicon(emoji, color, size=34, ring=None, badge=None):
    ring_css = f"box-shadow:0 0 0 4px {ring};" if ring else ""
    b = f'<div style="position:absolute;top:-8px;right:-8px;background:#16283B;color:#fff;border-radius:9px;padding:0 5px;font:700 11px sans-serif">{badge}</div>' if badge else ""
    return folium.DivIcon(
        icon_size=(size, size), icon_anchor=(size // 2, size // 2),
        html=f'<div style="position:relative;width:{size}px;height:{size}px;border-radius:50%;background:{color};'
             f'border:2px solid #fff;display:flex;align-items:center;justify-content:center;font-size:{int(size*0.55)}px;{ring_css}">{emoji}{b}</div>')


LEGEND_HTML = """
<div style="position:fixed;bottom:18px;left:12px;z-index:9999;background:#fff;border:1px solid #D9E1EA;border-radius:6px;
 padding:8px 12px;font:12px/1.7 Tahoma,sans-serif;direction:rtl;text-align:right;box-shadow:0 1px 4px rgba(0,0,0,.15)">
<b>دليل الخريطة</b><br>
🚑 سيارة إسعاف &nbsp; 🏥 مستشفى &nbsp; 🚨 حادث<br>
<span style="color:#2E9A6A">━</span> سالك &nbsp;<span style="color:#E2A31B">━</span> متوسط &nbsp;
<span style="color:#E0692B">━</span> مرتفع &nbsp;<span style="color:#8E1B24">┅</span> مغلق<br>
<span style="color:#0F8B8D">┅</span> المسار المقترح
</div>"""


def render_map(twin, incident, rec, engine="folium"):
    h, a, roads, inc = twin["hospitals"], twin["ambulances"], twin["roads"], twin["incidents"]
    inc = inc[inc["status"] != "تمت الاستجابة"]
    best_id = rec["best"]["id"] if rec else None
    occ = occupancy_series(h)

    if engine == "folium" and HAS_FOLIUM:
        m = folium.Map(location=RIYADH_CENTER, zoom_start=11, tiles="CartoDB positron", control_scale=True)
        for _, r in roads.iterrows():
            tip = f"<div dir='rtl'><b>{r['name']}</b><br>مستوى الازدحام: {r['congestion']}<br>" \
                  f"زمن الرحلة المتوقع: {'غير متاح' if pd.isna(r['trip_min']) else str(r['trip_min']) + ' دقيقة'}<br>حالة الطريق: {r['status']}</div>"
            folium.PolyLine(list(r["coords"]), color=LEVEL_COLOR[r["congestion"]], weight=6 if r["congestion"] != "منخفض" else 4,
                            opacity=.85, dash_array="8 8" if r["congestion"] == "مغلق" else None, tooltip=tip).add_to(m)
        for i, r in h.iterrows():
            o = float(occ[i]); lvl, col = pressure_level(o)
            html = (f"<div dir='rtl' style='font-family:Tahoma;min-width:210px'><b style='font-size:14px'>{r['name']}</b><hr style='margin:4px 0'>"
                    f"أسرة الطوارئ المتاحة: <b>{int(r['er_avail'])}</b> / {int(r['er_beds'])}<br>"
                    f"العناية المركزة المتاحة: <b>{int(r['icu_avail'])}</b> / {int(r['icu_beds'])}<br>"
                    f"نسبة الإشغال: <b>{o*100:.0f}%</b><br>مستوى الضغط: <b style='color:{col}'>{lvl}</b><br>"
                    f"التخصصات: {'، '.join(r['specialties'])}<br><small>{SIM_LABEL}</small></div>")
            is_best = best_id is not None and int(r["id"]) == best_id
            folium.Marker([r["lat"], r["lon"]], popup=folium.Popup(html, max_width=280),
                          tooltip=f"{r['name']} — {o*100:.0f}%",
                          icon=_divicon("🏥", col, 42 if is_best else 34, ring="rgba(31,138,112,.55)" if is_best else None,
                                        badge="★" if is_best else None)).add_to(m)
        for _, r in a.iterrows():
            col = STATUS_COLOR.get(r["status"], "#5B6B7C")
            tip = f"<div dir='rtl'><b>{r['name']}</b><br>الحالة: {r['status']}<br>المهمة: {r['mission']}<br>" \
                  f"وقت التوفر المتوقع: {'—' if r['status'] in ('متاحة', OUT_OF_SERVICE) else str(round(float(r['avail_in']))) + ' دقيقة'}</div>"
            folium.Marker([r["lat"], r["lon"]], tooltip=tip, icon=_divicon("🚑", col, 30)).add_to(m)
        for _, r in inc.iterrows():
            col = SEVERITY_COLOR[r["severity"]]
            tip = f"<div dir='rtl'><b>{r['type']}</b> ({r['id']})<br>الموقع: {r['place']}<br>مستوى الخطورة: {r['severity']}<br>" \
                  f"المصابون: {int(r['patients'])} — الحرجة: {int(r['critical'])}<br>الحالة: {r['status']}</div>"
            folium.Circle([r["lat"], r["lon"]], radius=900, color=col, fill=True, fill_opacity=.12, weight=1).add_to(m)
            folium.Marker([r["lat"], r["lon"]], tooltip=tip, icon=_divicon("🚨", col, 40)).add_to(m)
        if incident is not None and rec is not None:
            b = h[h["id"] == best_id].iloc[0]
            folium.PolyLine([[incident["lat"], incident["lon"]], [b["lat"], b["lon"]]], color="#0F8B8D", weight=4,
                            dash_array="4 10", tooltip="المسار المقترح إلى المستشفى").add_to(m)
        m.get_root().html.add_child(folium.Element(LEGEND_HTML))
        try:
            st_folium(m, height=590, use_container_width=True, returned_objects=[], key="riyadh_map")
        except TypeError:
            st_folium(m, height=590, width=1100, returned_objects=[], key="riyadh_map")
        return

    # خريطة احتياطية (Plotly) تعمل دون اتصال
    fig = go.Figure()
    for _, r in roads.iterrows():
        lat, lon = zip(*r["coords"])
        fig.add_trace(go.Scatter(x=lon, y=lat, mode="lines", name=r["name"], showlegend=False,
                                 line=dict(color=LEVEL_COLOR[r["congestion"]], width=5, dash="dash" if r["congestion"] == "مغلق" else "solid"),
                                 hovertext=f"{r['name']} — {r['congestion']} — {r['status']}", hoverinfo="text"))
    fig.add_trace(go.Scatter(x=h["lon"], y=h["lat"], mode="markers+text", text=[f"🏥 {n}" for n in h["name"]], textposition="top center",
                             marker=dict(size=[24 if int(i) == best_id else 16 for i in h["id"]], color=[pressure_level(o)[1] for o in occ], line=dict(width=2, color="#fff")),
                             hovertext=[f"{n}<br>الإشغال {o*100:.0f}%<br>أسرة الطوارئ {e}<br>العناية {c}" for n, o, e, c in zip(h["name"], occ, h["er_avail"], h["icu_avail"])],
                             hoverinfo="text", name="مستشفيات"))
    fig.add_trace(go.Scatter(x=a["lon"], y=a["lat"], mode="markers+text", text=["🚑"] * len(a), marker=dict(size=14, color=[STATUS_COLOR.get(s, "#5B6B7C") for s in a["status"]]),
                             hovertext=[f"{n}<br>{s}<br>{ms}" for n, s, ms in zip(a["name"], a["status"], a["mission"])], hoverinfo="text", name="سيارات الإسعاف"))
    if len(inc):
        fig.add_trace(go.Scatter(x=inc["lon"], y=inc["lat"], mode="markers+text", text=["🚨"] * len(inc), marker=dict(size=22, color=[SEVERITY_COLOR[s] for s in inc["severity"]]),
                                 hovertext=[f"{t}<br>{s}" for t, s in zip(inc["type"], inc["severity"])], hoverinfo="text", name="حوادث"))
    fig.update_layout(height=590, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#fff", plot_bgcolor="#F4F7FA", showlegend=False,
                      font=dict(family=FONT), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    st.plotly_chart(fig, key="fallback_map")


def render_readiness_gauge(rd, height=260):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=rd["score"], number=dict(suffix="%", font=dict(size=54, family=FONT, color=NAVY)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=rd["color"], thickness=.35),
                   steps=[dict(range=[0, 25], color="#F3D0CC"), dict(range=[25, 50], color="#F8DDCF"),
                          dict(range=[50, 75], color="#FBEBC2"), dict(range=[75, 90], color="#D8EBCF"), dict(range=[90, 100], color="#BFE3D2")])))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT))
    st.plotly_chart(fig, key=f"gauge_{height}")


def render_hospital_capacity(twin, title="نسبة إشغال المستشفيات", compare=None, key="occ"):
    h = twin["hospitals"].copy()
    h["occ"] = occupancy_series(h) * 100
    h = h.sort_values("occ")
    fig = go.Figure()
    if compare is not None:
        ch = compare["hospitals"]
        omap = dict(zip(ch["name"], occupancy_series(ch) * 100))
        fig.add_bar(y=h["name"], x=[omap[n] for n in h["name"]], orientation="h", name="قبل", marker_color="#B8C4D0")
    fig.add_bar(y=h["name"], x=h["occ"], orientation="h", name="بعد" if compare is not None else "الإشغال",
                marker_color=[pressure_level(o / 100)[1] for o in h["occ"]],
                text=[f"{o:.0f}%" for o in h["occ"]], textposition="outside")
    fig.add_vline(x=75, line_dash="dot", line_color="#8E1B24", annotation_text="حد الضغط 75%", annotation_position="top")
    fig.update_layout(title=title, height=380, barmode="group", margin=dict(l=10, r=40, t=50, b=10), showlegend=compare is not None,
                      xaxis=dict(range=[0, 112], title="نسبة الإشغال %"), yaxis=dict(title=""), font=dict(family=FONT),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff")
    st.plotly_chart(fig, key=key)


def scenario_table(results):
    rows = []
    for k, r in results.items():
        m = r["metrics"]
        rows.append({
            "السيناريو": r["name"], "زمن الوصول للمستشفى (د)": round(m["avg_arrival"], 1),
            "متوسط زمن الاستجابة (د)": round(m["avg_response"], 1),
            "أعلى إشغال بعد الاستقبال": f"{m['max_occupancy']*100:.0f}%",
            "الجاهزية": f"{m['readiness']['score']}% — {m['readiness']['label']}",
            "المستشفيات تحت الضغط": m["under_pressure"], "سيارات متاحة (بعد 25 د)": m["amb_available"],
            "الحالات الموزعة": f"{m['distributed']} على {m['hospitals_used']} مستشفى", "حالات بلا سرير": m["overflow"],
        })
    return pd.DataFrame(rows)


def render_scenario_comparison(results):
    names = [r["name"] for r in results.values()]
    keys = list(results.keys())
    cols = [SCEN_COLOR[k] for k in keys]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("زمن الوصول للمستشفى (دقيقة)", "أعلى إشغال بعد الاستقبال (%)", "جاهزية المنظومة (%)"),
                        horizontal_spacing=.08)
    series = [[r["metrics"]["avg_arrival"] for r in results.values()],
              [r["metrics"]["max_occupancy"] * 100 for r in results.values()],
              [r["metrics"]["readiness"]["score"] for r in results.values()]]
    for i, vals in enumerate(series, start=1):
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=cols, text=[f"{v:.1f}" if i == 1 else f"{v:.0f}" for v in vals],
                             textposition="outside", showlegend=False), row=1, col=i)
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), font=dict(family=FONT), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff")
    st.plotly_chart(fig, key="scenario_cmp")


def render_timeline(items):
    html = '<div class="tl">' + "".join(
        f'<div class="it"><div class="tm">{i["time"]}</div><div class="tx">{i["text"]}</div></div>' for i in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_recommendation(rec, incident):
    b = rec["best"]
    reasons = "".join(f'<div class="reason {"ok" if ok else "warn"}">{t}</div>' for ok, t in b["reasons"])
    st.markdown(f"""<div class="card rec"><h4>المستشفى المقترح</h4>
      <div class="recname">{b['hospital']}</div><div class="recscore">النتيجة: {b['score']:.0f}/100</div>
      <div style="margin-top:8px">{reasons}</div><div class="disc">{DISCLAIMER}</div></div>""", unsafe_allow_html=True)
    with st.expander("تفاصيل حساب النتيجة"):
        comp = pd.DataFrame({"المعيار": [WEIGHT_LABELS[k] for k in WEIGHTS], "الوزن": [f"{int(w*100)}%" for w in WEIGHTS.values()],
                             "الدرجة": [round(b["components"][k]) for k in WEIGHTS]})
        st.dataframe(comp, hide_index=True)
        st.dataframe(rec["ranking"], hide_index=True)


def render_before_after(b, a, title_b="قبل المحاكاة", title_a="بعد المحاكاة"):
    def col(m, cls, title):
        rows = [("متوسط زمن الاستجابة", f"{fmt_min(m['avg_response'])} د"), ("زمن الوصول للمستشفى", f"{fmt_min(m['avg_arrival'])} د"),
                ("سيارات متاحة (بعد 25 د)", m["amb_available"]), ("أعلى إشغال بعد الاستقبال", f"{m['max_occupancy']*100:.0f}%"),
                ("حالات بلا سرير", m["overflow"]), ("جاهزية المنظومة", f"{m['readiness']['score']}% — {m['readiness']['label']}")]
        inner = "".join(f'<div class="r"><span>{k}</span><b>{v}</b></div>' for k, v in rows)
        return f'<div class="ba {cls}"><h4 style="margin:0 0 6px">{title}</h4>{inner}</div>'
    c1, c2 = st.columns(2)
    c1.markdown(col(b, "before", title_b), unsafe_allow_html=True)
    c2.markdown(col(a, "after", title_a), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# تشغيل السيناريو التجريبي
# ----------------------------------------------------------------------------
def run_demo_pipeline():
    ss = st.session_state
    with st.status("جارٍ تشغيل السيناريو التجريبي…", expanded=True) as status:
        base = apply_demo_state(create_digital_twin())
        ss.base = base
        steps = []
        st.write("🚨 ١) عرض البلاغ على الخريطة: حادث مروري كبير — 6 مصابين، منهم 2 حرجة، ازدحام شديد")
        create_incident(base)
        time.sleep(.35)
        inc = current_incident()
        st.write("🚑 ٢) إظهار سيارات الإسعاف المتاحة (5 مركبات)")
        cand = candidate_ambulances(ss.twin, inc, 3)
        time.sleep(.35)
        st.write("🏥 ٣) عرض القدرة الاستيعابية: مستشفى تحت ضغط مرتفع وآخر بعناية مركزة جيدة")
        time.sleep(.35)
        st.write("🧮 ٤) تشغيل خوارزمية التوصية")
        rec = recommend_hospital(ss.twin, inc)
        time.sleep(.35)
        st.write("⏱️ ٥) تشغيل السيناريوهات الثلاثة عبر SimPy")
        ss.results = simulate_all(ss.twin)
        st.write("📊 ٦) مقارنة السيناريوهات")
        time.sleep(.3)
        st.write("🟢 ٧) تحديث جاهزية المنظومة")
        time.sleep(.3)
        st.write(f"🗺️ ٨) أثر القرار على التوأم الرقمي — التوصية: {rec['best']['hospital']}")
        ss.demo_ran = True
        ss.view = VIEW_NOW
        status.update(label="اكتمل السيناريو التجريبي — راجع الخريطة والنتائج أدناه", state="complete", expanded=False)


# ----------------------------------------------------------------------------
# الصفحة
# ----------------------------------------------------------------------------
init_state()
ss = st.session_state

# الشريط الجانبي: محاكاة بلاغ طارئ
with st.sidebar:
    st.markdown("### محاكاة بلاغ طارئ")
    st.selectbox("نوع الحادث", INCIDENT_TYPES, key="inc_type")
    st.selectbox("مستوى الخطورة", SEVERITIES, key="severity")
    st.slider("عدد المصابين", 1, 10, key="n_injured")
    if ss.n_critical > ss.n_injured:
        ss.n_critical = ss.n_injured
    st.slider("عدد الحالات الحرجة", 0, ss.n_injured, key="n_critical")
    st.selectbox("التخصص المطلوب", SPECIALTIES, key="specialty")
    st.selectbox("حالة الطريق", ROAD_CONDITIONS, key="road_cond")
    st.selectbox("موقع الحادث", list(LOCATIONS.keys()), key="place")
    if st.button("إنشاء البلاغ", type="primary", key="create"):
        create_incident(ss.base)
    st.button("إعادة ضبط التوأم الرقمي", on_click=reset_all)
    st.divider()
    engine = st.radio("نوع الخريطة", ["تفاعلية (Folium)", "احتياطية (Plotly، دون اتصال)"], index=0 if HAS_FOLIUM else 1)
    st.caption(SIM_LABEL)

# الترويسة
st.markdown(f"""<div class="hero">
  <h1>التوأم الرقمي لمنظومة الإسعاف والطوارئ</h1>
  <div class="sub">محاكاة ذكية للاستجابة للطوارئ وتوزيع الموارد في مدينة الرياض</div>
  <span class="badge"><i></i>وضع المحاكاة</span><span class="simlabel">{SIM_LABEL}</span>
</div>
<div class="explain">النظام عبارة عن نموذج رقمي يحاكي منظومة الإسعاف والطوارئ، ويسمح بتجربة القرارات المختلفة ومعرفة تأثيرها المتوقع قبل تنفيذها.</div>
""", unsafe_allow_html=True)

# أزرار التشغيل
b1, b2 = st.columns([3, 2])
with b1:
    with st.container(key="demo"):
        st.button("تشغيل سيناريو تجريبي", on_click=start_demo)
with b2:
    with st.container(key="runsim"):
        run_clicked = st.button("تشغيل المحاكاة")

if ss.pop("run_demo", False):
    run_demo_pipeline()
elif run_clicked:
    if current_incident() is None:
        st.warning("أنشئ بلاغًا من الشريط الجانبي أو شغّل السيناريو التجريبي أولًا.")
    else:
        with st.spinner("جارٍ محاكاة السيناريوهات الثلاثة…"):
            ss.results = simulate_all(ss.whatif["twin_modified"] if ss.whatif else ss.twin)
            ss.view = VIEW_NOW

# اختيار عرض التوأم بعد المحاكاة
if ss.results:
    opts = [VIEW_NOW] + [f"بعد: {v}" for v in STRATEGIES.values()]
    if ss.get("view") not in opts:
        ss.view = VIEW_NOW
    st.radio("عرض حالة التوأم الرقمي", opts, horizontal=True, key="view")

if ss.whatif:
    wc1, wc2 = st.columns([5, 1])
    wc1.info(f"يعرض التوأم الآن سيناريو «ماذا لو»: {ss.whatif['title']} — {ss.whatif['description']}")
    if wc2.button("إلغاء ماذا لو"):
        ss.whatif = None
        ss.results = None
        st.rerun()

twin = display_twin()
incident = current_incident()
metrics = twin_metrics(twin)
metrics0 = twin_metrics(ss.base)
rec = recommend_hospital(ss.whatif["twin_modified"] if ss.whatif else ss.twin, incident) if incident is not None else None

render_kpis(metrics, metrics0)
st.write("")

# الخريطة + القرار
mcol, dcol = st.columns([7, 4])
with mcol:
    render_map(twin, incident, rec, "folium" if engine.startswith("تفاعلية") else "plotly")
with dcol:
    st.markdown("#### جاهزية المنظومة")
    rd = metrics["readiness"]
    render_readiness_gauge(rd)
    st.markdown(f'<div style="text-align:center;margin-top:-10px"><span class="chip" style="--c:{rd["color"]}">{rd["label"]}</span></div>', unsafe_allow_html=True)
    st.write("")
    if incident is not None:
        st.markdown(f"""<div class="card" style="border-inline-start:7px solid {SEVERITY_COLOR[incident['severity']]}">
          <h4>البلاغ الحالي</h4><b>{incident['type']}</b> — {incident['place']}<br>
          مستوى الخطورة: <span class="chip" style="--c:{SEVERITY_COLOR[incident['severity']]}">{incident['severity']}</span>
          &nbsp; المصابون: <b>{int(incident['patients'])}</b> (حرجة: <b>{int(incident['critical'])}</b>)<br>
          التخصص المطلوب: {incident['specialty']} — حالة الطريق: {incident['road_condition']}</div>""", unsafe_allow_html=True)
        render_recommendation(rec, incident)
        cand = candidate_ambulances(ss.whatif["twin_modified"] if ss.whatif else ss.twin, incident, 4)
        if len(cand):
            with st.expander("أقرب سيارات الإسعاف المتاحة"):
                st.dataframe(cand, hide_index=True)
    else:
        st.info("لا يوجد بلاغ جديد. أنشئ بلاغًا من الشريط الجانبي أو اضغط «تشغيل سيناريو تجريبي».")

# نتائج المحاكاة
sim_input = ss.whatif["twin_modified"] if ss.whatif else ss.twin
st.markdown("## نتائج المحاكاة ومقارنة السيناريوهات")
if ss.results:
    res = ss.results
    render_scenario_comparison(res)
    st.dataframe(scenario_table(res), hide_index=True)
    best_k = max(res, key=lambda k: res[k]["metrics"]["readiness"]["score"] - 0.2 * res[k]["metrics"]["avg_arrival"])
    st.success(f"الأنسب لجاهزية المنظومة ضمن هذا البلاغ: «{res[best_k]['name']}» — "
               f"جاهزية {res[best_k]['metrics']['readiness']['score']}% وأعلى إشغال {res[best_k]['metrics']['max_occupancy']*100:.0f}%. "
               "نتيجة لدعم القرار وليست قرارًا طبيًا.")
    left, right = st.columns([3, 2])
    with left:
        sc = st.selectbox("أثر السيناريو على سعة المستشفيات", list(STRATEGIES.keys()), index=2, format_func=lambda k: STRATEGIES[k], key="impact_sel")
        render_hospital_capacity(res[sc]["twin_after"], f"إشغال المستشفيات بعد «{STRATEGIES[sc]}» (الرمادي = قبل)",
                                 compare=sim_input, key=f"impact_{sc}")
    with right:
        st.markdown("#### الخط الزمني للاستجابة")
        tsc = st.selectbox("السيناريو", list(STRATEGIES.keys()), index=2, format_func=lambda k: STRATEGIES[k], key="tl_sel")
        tl = timeline_for(res[tsc], incident["id"], 0) if incident is not None else []
        render_timeline(tl)
        ld = res[tsc]["metrics"]["load_by_hospital"]
        if ld:
            st.caption("توزيع الحالات: " + "، ".join(f"{k} ({v})" for k, v in ld.items()))
else:
    st.info("اضغط «تشغيل المحاكاة» بعد إنشاء بلاغ لمقارنة السيناريوهات الثلاثة: أقرب مستشفى، أعلى قدرة استيعابية، التوزيع المتوازن.")

# تبويبات: ماذا لو / الحالة الحالية / لماذا توأم رقمي
tab_w, tab_s, tab_d = st.tabs(["ماذا لو؟", "الحالة الحالية للتوأم الرقمي", "لماذا يعتبر النظام توأمًا رقميًا؟"])

with tab_w:
    st.markdown("# ماذا لو؟")
    kind = st.radio("اختر سيناريو", list(WHAT_IFS.keys()), format_func=lambda k: WHAT_IFS[k], key="wi_kind")
    params = {}
    if kind == "road":
        params["road"] = st.selectbox("الطريق", list(ss.base["roads"]["name"]), key="wi_road")
    if kind == "full":
        params["hospital"] = st.selectbox("المستشفى", list(ss.base["hospitals"]["name"]),
                                          index=list(ss.base["hospitals"]["name"]).index(rec["best"]["hospital"]) if rec else 0, key="wi_hosp")
    strat = st.selectbox("استراتيجية التوجيه", list(STRATEGIES.keys()), index=2, format_func=lambda k: STRATEGIES[k], key="wi_strat")
    if st.button("تشغيل «ماذا لو»", key="wi_run"):
        base_tw = ss.twin
        if not len(pending_incidents(base_tw)):
            ss.twin = base_tw = add_incident(ss.base, default_incident_spec())
            ss.incident_id = base_tw["incidents"].iloc[-1]["id"]
        ss.whatif = simulate_what_if(base_tw, kind, params, strat)
        ss.results = None
        st.rerun()
    if ss.whatif:
        w = ss.whatif
        st.markdown(f"**{w['title']}** — {w['description']}")
        render_before_after(w["before"]["metrics"], w["after"]["metrics"])
        render_hospital_capacity(w["after"]["twin_after"], "إشغال المستشفيات: قبل (رمادي) وبعد «ماذا لو»", compare=w["before"]["twin_after"], key="wi_occ")
        d = w["after"]["metrics"]["readiness"]["score"] - w["before"]["metrics"]["readiness"]["score"]
        st.caption(f"تغيّر جاهزية المنظومة: {d:+d} نقطة. الخريطة أعلاه تعرض الحالة المعدّلة.")

with tab_s:
    st.markdown("## الحالة الحالية للتوأم الرقمي")
    amb = twin["ambulances"]
    idx = metrics["traffic_index"]
    road_lbl = "يوجد إغلاق" if metrics["closed_roads"] else ("سالكة" if idx < 1.15 else "حركة متوسطة" if idx < 1.4 else "مزدحمة")
    stats = [("عدد البلاغات", metrics["active_incidents"]), ("سيارات الإسعاف المتاحة", f"{metrics['available_ambulances']} / {len(amb)}"),
             ("متوسط إشغال المستشفيات", f"{metrics['avg_occupancy']*100:.0f}%"), ("حالة الطرق", road_lbl),
             ("متوسط زمن الاستجابة", f"{fmt_min(metrics['avg_response'])} د"), ("مستوى الجاهزية", f"{metrics['readiness']['score']}% — {metrics['readiness']['label']}")]
    for col, (l, v) in zip(st.columns(6), stats):
        col.markdown(f'<div class="stat"><div class="l">{l}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        render_hospital_capacity(twin, key="state_occ")
        levels = ["مستقر", "ضغط متوسط", "ضغط مرتفع", "حرج"]
        cnt = pd.Series([pressure_level(o)[0] for o in occupancy_series(twin["hospitals"])]).value_counts()
        st.caption("مستويات الضغط: " + " — ".join(f"{l}: {int(cnt.get(l, 0))}" for l in levels))
    with c2:
        sc_ = amb["status"].value_counts()
        pie = go.Figure(go.Pie(labels=list(sc_.index), values=list(sc_.values), hole=.55,
                               marker=dict(colors=[STATUS_COLOR.get(s, "#5B6B7C") for s in sc_.index]), textinfo="value+label"))
        pie.update_layout(title="حالة سيارات الإسعاف", height=380, showlegend=False, margin=dict(l=10, r=10, t=50, b=10), font=dict(family=FONT), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(pie, key="amb_pie")
    r = twin["roads"][["name", "congestion", "trip_min", "status"]].copy()
    r["trip_min"] = r["trip_min"].apply(lambda x: "غير متاح" if pd.isna(x) else f"{x:.1f} دقيقة")
    r.columns = ["الطريق", "مستوى الازدحام", "زمن الرحلة المتوقع", "حالة الطريق"]
    st.dataframe(r, hide_index=True)
    a2 = amb[["name", "status", "mission", "avail_in"]].copy()
    a2["avail_in"] = a2.apply(lambda x: "—" if x["status"] in ("متاحة", OUT_OF_SERVICE) else f"{x['avail_in']:.0f} دقيقة", axis=1)
    a2.columns = ["رقم المركبة", "الحالة", "المهمة الحالية", "وقت التوفر المتوقع"]
    with st.expander("جدول سيارات الإسعاف"):
        st.dataframe(a2, hide_index=True)

with tab_d:
    st.markdown("# لماذا يعتبر النظام توأمًا رقميًا؟")
    f1, f2 = st.columns([2, 3])
    with f1:
        st.markdown("""<div class="flow">
          <div class="n reality">الواقع<br>🚑 سيارات الإسعاف · 🏥 المستشفيات · 🚗 الطرق · 🚨 الحوادث</div><div class="a">↓</div>
          <div class="n">البيانات</div><div class="a">↓</div>
          <div class="n">النموذج الرقمي</div><div class="a">↓</div>
          <div class="n">المحاكاة</div><div class="a">↓</div>
          <div class="n">اختبار القرارات</div><div class="a">↓</div>
          <div class="n result">التأثير المتوقع</div></div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="card">التوأم الرقمي هو نموذج رقمي لمنظومة حقيقية، يستطيع تمثيل حالتها ومحاكاة التغييرات والقرارات المختلفة.
        في هذا المشروع، يمثل النظام منظومة الإسعاف والطوارئ في مدينة الرياض، ويتيح تجربة سيناريوهات مختلفة ومعرفة تأثيرها المتوقع
        على زمن الاستجابة، وسيارات الإسعاف، والقدرة الاستيعابية للمستشفيات.</div>""", unsafe_allow_html=True)
        st.warning("البيانات المستخدمة في النموذج تجريبية وليست بيانات تشغيلية حقيقية.")
        st.markdown("```\nPython → النموذج الرقمي → محاكاة SimPy → خوارزمية القرار → لوحة Streamlit → خريطة الرياض التفاعلية\n```")
    st.markdown('<div class="msg">النظام التقليدي يخبرك بما يحدث الآن، أما التوأم الرقمي فيتيح لك محاكاة ما قد يحدث قبل اتخاذ القرار.</div>', unsafe_allow_html=True)

st.caption(f"{SIM_LABEL} — لا تمثل السعة أو المواقع معلومات آنية أو حقيقية.")
