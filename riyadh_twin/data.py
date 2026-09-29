# -*- coding: utf-8 -*-
"""
data.py — النموذج الرقمي (Digital Twin Model)
بيانات تجريبية بالكامل لمدينة الرياض. لا تمثل بيانات تشغيلية حقيقية.
"""
import copy
import numpy as np
import pandas as pd

SIM_LABEL = "بيانات محاكاة لأغراض العرض والتجربة"
RIYADH_CENTER = (24.7136, 46.6753)

INCIDENT_TYPES = ["حادث مروري", "حالة قلبية", "حريق", "حادث جماعي", "حالة طبية طارئة"]
SEVERITIES = ["حرج", "مرتفع", "متوسط", "منخفض"]
SPECIALTIES = ["طوارئ عامة", "إصابات وحوادث", "قلب", "أعصاب", "حروق"]
ROAD_CONDITIONS = ["طبيعية", "ازدحام متوسط", "ازدحام شديد", "إغلاق طريق"]
AMB_STATUSES = ["متاحة", "في الطريق", "في موقع الحادث", "تنقل مصابًا", "في المستشفى"]
OUT_OF_SERVICE = "خارج الخدمة"

# مواقع الحوادث المتاحة في الواجهة (إحداثيات تقريبية)
LOCATIONS = {
    "النخيل": (24.7350, 46.6300),
    "العليا": (24.6950, 46.6850),
    "الملز": (24.6650, 46.7350),
    "حطين": (24.7750, 46.5850),
    "الياسمين": (24.8250, 46.6450),
    "الروضة": (24.7350, 46.7700),
    "السويدي": (24.6000, 46.6000),
    "النسيم": (24.7300, 46.8300),
    "العزيزية": (24.5750, 46.7700),
    "الدرعية": (24.7400, 46.5650),
}

# مستويات الازدحام المروري
LEVELS = ["منخفض", "متوسط", "مرتفع", "مغلق"]
LEVEL_MULT = {"منخفض": 1.0, "متوسط": 1.3, "مرتفع": 1.7, "مغلق": 2.0}
LEVEL_STATUS = {"منخفض": "سالك", "متوسط": "حركة بطيئة", "مرتفع": "ازدحام شديد", "مغلق": "مغلق"}
LEVEL_COLOR = {"منخفض": "#2E9A6A", "متوسط": "#E2A31B", "مرتفع": "#E0692B", "مغلق": "#8E1B24"}
COND_MULT = {"طبيعية": 1.0, "ازدحام متوسط": 1.15, "ازدحام شديد": 1.3, "إغلاق طريق": 1.5}
COND_TO_LEVEL = {"طبيعية": None, "ازدحام متوسط": "متوسط", "ازدحام شديد": "مرتفع", "إغلاق طريق": "مغلق"}

BASE_SPEED_KMH = 65.0
CIRCUITY = 1.3
ROAD_SPEED_KMH = 70.0

SEVERITY_COLOR = {"حرج": "#C8372D", "مرتفع": "#E0692B", "متوسط": "#E2A31B", "منخفض": "#2C6FBB"}
STATUS_COLOR = {
    "متاحة": "#2E9A6A", "في الطريق": "#2C6FBB", "في موقع الحادث": "#6B4FA3",
    "تنقل مصابًا": "#E0692B", "في المستشفى": "#5B6B7C", OUT_OF_SERVICE: "#2B2F36",
}


# ----------------------------------------------------------------------------
# أدوات جغرافية
# ----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _densify(coords, step_km=2.0):
    pts = []
    for (a, b), (c, d) in zip(coords[:-1], coords[1:]):
        n = max(1, int(haversine_km(a, b, c, d) // step_km))
        for i in range(n):
            t = i / n
            pts.append((a + (c - a) * t, b + (d - b) * t))
    pts.append(coords[-1])
    return pts


def road_length_km(coords):
    return float(sum(haversine_km(a, b, c, d) for (a, b), (c, d) in zip(coords[:-1], coords[1:])))


class RoadModel:
    """نموذج زمن الرحلة: يعتمد على مستوى الازدحام في أقرب طريق للمسار."""

    def __init__(self, roads: pd.DataFrame):
        pts, mult = [], []
        for _, r in roads.iterrows():
            m = LEVEL_MULT[r["congestion"]]
            for p in _densify(list(r["coords"])):
                pts.append(p)
                mult.append(m)
        self.pts = np.array(pts)
        self.mult = np.array(mult)

    def mult_at(self, lat, lon):
        d = haversine_km(lat, lon, self.pts[:, 0], self.pts[:, 1])
        i = int(d.argmin())
        return float(self.mult[i]) if d[i] < 5.0 else 1.0

    def route(self, lat1, lon1, lat2, lon2, extra=1.0):
        """يعيد (الزمن بالدقائق، معامل الازدحام على المسار)."""
        dist = float(haversine_km(lat1, lon1, lat2, lon2)) * CIRCUITY
        m = (self.mult_at(lat1, lon1) + self.mult_at((lat1 + lat2) / 2, (lon1 + lon2) / 2)
             + self.mult_at(lat2, lon2)) / 3.0
        m *= extra
        eff = 1.0 + 0.5 * (m - 1.0)          # أولوية المركبات الطارئة تخفف أثر الازدحام
        minutes = dist / (BASE_SPEED_KMH / eff) * 60.0 + 1.0
        return minutes, m


# ----------------------------------------------------------------------------
# توليد البيانات
# ----------------------------------------------------------------------------
def _roads_df():
    rows = [
        ("طريق الملك فهد", [(24.880, 46.640), (24.820, 46.650), (24.760, 46.668), (24.700, 46.680), (24.640, 46.690), (24.580, 46.700)], "مرتفع"),
        ("طريق العليا", [(24.800, 46.615), (24.750, 46.640), (24.700, 46.655), (24.650, 46.670)], "متوسط"),
        ("الطريق الدائري الشمالي", [(24.795, 46.520), (24.800, 46.600), (24.800, 46.680), (24.798, 46.760), (24.790, 46.840)], "متوسط"),
        ("الطريق الدائري الشرقي", [(24.820, 46.800), (24.770, 46.815), (24.720, 46.825), (24.660, 46.810), (24.600, 46.790)], "منخفض"),
        ("الطريق الدائري الجنوبي", [(24.590, 46.560), (24.585, 46.640), (24.585, 46.720), (24.590, 46.800)], "منخفض"),
        ("الطريق الدائري الغربي", [(24.820, 46.530), (24.760, 46.520), (24.700, 46.515), (24.640, 46.530), (24.590, 46.560)], "منخفض"),
        ("طريق مكة المكرمة", [(24.660, 46.500), (24.680, 46.580), (24.700, 46.650), (24.712, 46.705)], "مرتفع"),
        ("طريق الملك عبدالله", [(24.730, 46.680), (24.740, 46.740), (24.745, 46.800), (24.750, 46.860)], "متوسط"),
    ]
    df = pd.DataFrame(rows, columns=["name", "coords", "congestion"])
    return refresh_roads(df)


def refresh_roads(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["length_km"] = df["coords"].apply(road_length_km)
    df["status"] = df["congestion"].map(LEVEL_STATUS)
    mult = df["congestion"].map(LEVEL_MULT)
    df["trip_min"] = np.where(df["congestion"] == "مغلق", np.nan,
                              df["length_km"] / (ROAD_SPEED_KMH / mult) * 60.0).round(1)
    return df


def _hospitals_df():
    g = ["طوارئ عامة"]
    rows = [
        # id, الاسم, lat, lon, أسرة طوارئ, أسرة متاحة, عناية مركزة, عناية متاحة, التخصصات
        (1, "مستشفى الرياض العام", 24.6300, 46.7100, 20, 3, 8, 2, g + ["إصابات وحوادث", "قلب"]),
        (2, "مستشفى شمال الرياض", 24.8200, 46.6500, 18, 6, 8, 4, g + ["قلب", "أعصاب"]),
        (3, "مستشفى شرق الرياض", 24.7400, 46.8300, 16, 4, 7, 2, g + ["إصابات وحوادث", "حروق"]),
        (4, "مستشفى غرب الرياض", 24.7000, 46.5000, 15, 7, 6, 4, g + ["إصابات وحوادث"]),
        (5, "مستشفى جنوب الرياض", 24.5600, 46.7000, 14, 5, 5, 3, g + ["حروق"]),
        (6, "المركز الطبي المركزي", 24.6900, 46.6900, 24, 11, 12, 7, g + ["إصابات وحوادث", "قلب", "أعصاب"]),
        (7, "مستشفى الطوارئ المتخصص", 24.7700, 46.7200, 18, 6, 8, 3, g + ["إصابات وحوادث", "حروق", "أعصاب"]),
        (8, "المستشفى الجامعي", 24.7100, 46.6200, 22, 2, 10, 1, g + ["إصابات وحوادث", "قلب", "أعصاب", "حروق"]),
        (9, "مستشفى شمال شرق الرياض", 24.8000, 46.7900, 12, 5, 4, 2, g + ["قلب"]),
    ]
    return pd.DataFrame(rows, columns=["id", "name", "lat", "lon", "er_beds", "er_avail",
                                       "icu_beds", "icu_avail", "specialties"])


def _ambulances_df():
    rows = [
        # id, الاسم, lat, lon, الحالة, يتوفر بعد (د), المهمة
        (1, "AMB-01", 24.7000, 46.6800, "متاحة", 0, "—"),
        (2, "AMB-02", 24.7400, 46.6400, "متاحة", 0, "—"),
        (3, "AMB-03", 24.8000, 46.6500, "متاحة", 0, "—"),
        (4, "AMB-04", 24.6500, 46.7200, "متاحة", 0, "—"),
        (5, "AMB-05", 24.7500, 46.7800, "متاحة", 0, "—"),
        (6, "AMB-06", 24.6100, 46.6500, "متاحة", 0, "—"),
        (7, "AMB-07", 24.7100, 46.5600, "متاحة", 0, "—"),
        (8, "AMB-08", 24.7640, 46.7250, "في الطريق", 6, "بلاغ I-101"),
        (9, "AMB-09", 24.7650, 46.7300, "في موقع الحادث", 14, "بلاغ I-101"),
        (10, "AMB-10", 24.6900, 46.6900, "تنقل مصابًا", 9, "بلاغ I-102"),
        (11, "AMB-11", 24.6300, 46.7100, "في المستشفى", 4, "تسليم مصاب"),
        (12, "AMB-12", 24.8100, 46.6200, "في الطريق", 11, "بلاغ I-103"),
    ]
    df = pd.DataFrame(rows, columns=["id", "name", "lat", "lon", "status", "avail_in", "mission"])
    df["avail_in"] = df["avail_in"].astype(float)
    return df


INCIDENT_COLS = ["id", "type", "severity", "lat", "lon", "place", "patients", "critical",
                 "specialty", "status", "road_condition", "dest_hospital", "arrive_min",
                 "response_min", "offset"]


def _incidents_df():
    rows = [
        ("I-101", "حادث مروري", "مرتفع", 24.7650, 46.7300, "طريق الملك عبدالله", 3, 1, "إصابات وحوادث", "قيد المعالجة", "طبيعية", 3, 5.0, 7.2, 0),
        ("I-102", "حالة قلبية", "حرج", 24.6800, 46.7050, "الملز", 1, 1, "قلب", "قيد المعالجة", "طبيعية", 6, 8.0, 9.6, 0),
        ("I-103", "حريق", "مرتفع", 24.8300, 46.6100, "الياسمين", 2, 0, "حروق", "قيد المعالجة", "طبيعية", 2, 12.0, 8.1, 0),
        ("I-104", "حالة طبية طارئة", "متوسط", 24.6000, 46.7300, "طريق الملك فهد الجنوبي", 1, 0, "طوارئ عامة", "قيد المعالجة", "طبيعية", 5, 6.0, 8.7, 0),
    ]
    return pd.DataFrame(rows, columns=INCIDENT_COLS)


def generate_demo_data(demo: bool = False, seed: int = 7) -> dict:
    """يولّد بيانات تجريبية لمدينة الرياض (لا تمثل بيانات حقيقية)."""
    data = {
        "ambulances": _ambulances_df(),
        "hospitals": _hospitals_df(),
        "roads": _roads_df(),
        "incidents": _incidents_df(),
    }
    return data


def create_digital_twin(demo: bool = False) -> dict:
    """يبني النموذج الرقمي للمنظومة من البيانات التجريبية."""
    twin = generate_demo_data(demo=demo)
    twin["meta"] = {"is_demo": False, "override": {}}
    if demo:
        twin = apply_demo_state(twin)
    return twin


def clone_twin(twin: dict) -> dict:
    return {
        "ambulances": twin["ambulances"].copy(),
        "hospitals": twin["hospitals"].copy(),
        "roads": twin["roads"].copy(),
        "incidents": twin["incidents"].copy(),
        "meta": copy.deepcopy(twin["meta"]),
    }


def apply_demo_state(twin: dict) -> dict:
    """يهيّئ حالة السيناريو التجريبي: 5 مركبات متاحة، مستشفى تحت ضغط، وآخر بعناية مركزة جيدة."""
    t = clone_twin(twin)
    a = t["ambulances"]
    for name in ("AMB-06", "AMB-07"):
        i = a.index[a["name"] == name][0]
        a.loc[i, ["status", "avail_in", "mission"]] = ["في الطريق", 9 if name == "AMB-06" else 12, "بلاغ سابق"]
    h = t["hospitals"]
    h.loc[h["name"] == "مستشفى الرياض العام", ["er_avail", "icu_avail"]] = [6, 3]     # ~68%
    h.loc[h["name"] == "المستشفى الجامعي", ["er_avail", "icu_avail"]] = [2, 0]        # ~94% (ضغط مرتفع)
    h.loc[h["name"] == "المركز الطبي المركزي", ["er_avail", "icu_avail"]] = [8, 10]   # عناية مركزة جيدة
    h.loc[h["name"] == "مستشفى غرب الرياض", ["er_avail", "icu_avail"]] = [9, 4]      # سعة جيدة لكنه أبعد
    t["meta"]["is_demo"] = True
    return t


# ----------------------------------------------------------------------------
# مؤشرات مشتقة
# ----------------------------------------------------------------------------
def hospital_occupancy(h) -> float:
    total = h["er_beds"] + h["icu_beds"]
    return float(1 - (h["er_avail"] + h["icu_avail"]) / total) if total else 1.0


def occupancy_series(hospitals: pd.DataFrame) -> pd.Series:
    return 1 - (hospitals["er_avail"] + hospitals["icu_avail"]) / (hospitals["er_beds"] + hospitals["icu_beds"])


def pressure_level(occ: float):
    """يعيد (الوصف، اللون)."""
    if occ >= 0.90:
        return "حرج", "#8E1B24"
    if occ >= 0.75:
        return "ضغط مرتفع", "#E0692B"
    if occ >= 0.60:
        return "ضغط متوسط", "#E2A31B"
    return "مستقر", "#2E9A6A"


UNDER_PRESSURE_THRESHOLD = 0.75


# ----------------------------------------------------------------------------
# إضافة البلاغات
# ----------------------------------------------------------------------------
def _nearest_road_index(roads: pd.DataFrame, lat: float, lon: float) -> int:
    best, best_d = 0, 1e9
    for i, (_, r) in enumerate(roads.iterrows()):
        for p in _densify(list(r["coords"]), 1.0):
            d = float(haversine_km(lat, lon, p[0], p[1]))
            if d < best_d:
                best, best_d = i, d
    return best


def next_incident_id(twin: dict) -> str:
    nums = [int(str(i).split("-")[1]) for i in twin["incidents"]["id"] if "-" in str(i)]
    return f"I-{max(nums + [100]) + 1}"


def add_incident(twin: dict, spec: dict) -> dict:
    """يضيف بلاغًا جديدًا (بانتظار الإرسال) ويحدّث حالة الطريق القريب حسب الحالة المختارة."""
    t = clone_twin(twin)
    lat, lon = LOCATIONS[spec["place"]] if "lat" not in spec else (spec["lat"], spec["lon"])
    inc = dict(
        id=spec.get("id") or next_incident_id(t), type=spec["type"], severity=spec["severity"],
        lat=lat, lon=lon, place=spec["place"], patients=int(spec["patients"]),
        critical=int(min(spec["critical"], spec["patients"])), specialty=spec["specialty"],
        status="بانتظار الإرسال", road_condition=spec.get("road_condition", "طبيعية"),
        dest_hospital=0, arrive_min=np.nan, response_min=np.nan, offset=float(spec.get("offset", 0)),
    )
    t["incidents"] = pd.concat([t["incidents"], pd.DataFrame([inc])[INCIDENT_COLS]], ignore_index=True)
    lvl = COND_TO_LEVEL[inc["road_condition"]]
    if lvl:
        roads = t["roads"]
        i = _nearest_road_index(roads, lat, lon)
        if LEVELS.index(lvl) > LEVELS.index(roads.loc[i, "congestion"]):
            roads.loc[i, "congestion"] = lvl
        t["roads"] = refresh_roads(roads)
    return t


def pending_incidents(twin: dict) -> pd.DataFrame:
    inc = twin["incidents"]
    return inc[inc["status"] == "بانتظار الإرسال"]


def default_incident_spec() -> dict:
    return dict(type="حادث مروري", severity="مرتفع", patients=4, critical=1, specialty="إصابات وحوادث",
                place="النخيل", road_condition="طبيعية")


DEMO_SPEC = dict(type="حادث مروري", severity="حرج", patients=6, critical=2, specialty="إصابات وحوادث",
                 place="النخيل", road_condition="ازدحام شديد")
