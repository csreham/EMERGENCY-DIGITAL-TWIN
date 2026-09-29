# -*- coding: utf-8 -*-
"""
simulation.py — محاكاة الأحداث عبر الزمن باستخدام SimPy، وحساب جاهزية المنظومة.
"""
import math
import numpy as np
import pandas as pd
import simpy

from data import (RoadModel, COND_MULT, LEVEL_MULT, clone_twin, pending_incidents, refresh_roads,
                  occupancy_series, UNDER_PRESSURE_THRESHOLD, OUT_OF_SERVICE, add_incident,
                  _nearest_road_index)
from recommendation import calculate_hospital_score

STRATEGIES = {
    "nearest": "أقرب مستشفى",
    "capacity": "أعلى قدرة استيعابية",
    "balanced": "التوزيع المتوازن",
}

CHECKPOINT = 25.0      # لحظة قياس جاهزية المنظومة (دقيقة من أول بلاغ)
SIM_END = 240.0
DISPATCH_MIN = 1.0     # معالجة البلاغ وإسناد المركبة
SCENE_MIN = 4.0        # زمن التثبيت في الموقع
SCENE_CRIT_MIN = 6.0
HANDOFF_MIN = 3.0      # تسليم المصاب
OVERFLOW_DELAY = 12.0  # انتظار إضافي عند عدم توفر سرير
MAX_WAIT = 45.0
CLOCK_START = 14 * 60 + 2   # 14:02


def clock(minutes: float) -> str:
    t = CLOCK_START + int(round(minutes))
    return f"{(t // 60) % 24:02d}:{t % 60:02d}"


# ----------------------------------------------------------------------------
# جاهزية المنظومة
# ----------------------------------------------------------------------------
def readiness_label(score: float):
    if score >= 90:
        return "جاهزية ممتازة", "#2E9A6A"
    if score >= 75:
        return "جاهزية جيدة", "#5FA36A"
    if score >= 50:
        return "جاهزية متوسطة", "#E2A31B"
    if score >= 25:
        return "جاهزية منخفضة", "#E0692B"
    return "حالة حرجة", "#8E1B24"


def _c(x):
    return max(0.0, min(1.0, float(x)))


def traffic_index(roads: pd.DataFrame) -> float:
    return float(roads["congestion"].map(LEVEL_MULT).mean())


def calculate_system_readiness(ambulances, hospitals, roads, incidents_active, avg_response, overflow: int = 0) -> dict:
    """جاهزية المنظومة (0–100) من: المركبات، السعة، الطرق، عدد البلاغات، زمن الاستجابة."""
    in_service = ambulances[ambulances["status"] != OUT_OF_SERVICE]
    avail = int((ambulances["status"] == "متاحة").sum())
    amb = _c(avail / max(1.0, 0.5 * max(len(in_service), 1)))            # 50% متاحة = جاهزية كاملة
    occ = occupancy_series(hospitals)
    cap = _c(1 - (0.5 * max(0.0, occ.mean() - 0.5) / 0.5 + 0.5 * max(0.0, occ.max() - 0.75) / 0.25))
    cap = _c(cap - 0.06 * overflow)                                      # كل حالة بلا سرير تخفض السعة الفعلية
    traffic = _c(1 - (traffic_index(roads) - 1) / 1.5)
    patients = float(incidents_active["patients"].sum()) if len(incidents_active) else 0.0
    load = _c(1 - patients / 30.0)
    resp = _c(1 - (avg_response - 5) / 15.0) if avg_response and not math.isnan(avg_response) else 0.8
    parts = {"المركبات": amb, "سعة المستشفيات": cap, "الطرق": traffic, "عبء البلاغات": load, "زمن الاستجابة": resp}
    score = 100 * (0.25 * amb + 0.30 * cap + 0.15 * traffic + 0.15 * load + 0.15 * resp)
    label, color = readiness_label(score)
    return {"score": int(round(score)), "label": label, "color": color, "parts": parts}


def twin_metrics(twin: dict) -> dict:
    """مؤشرات لوحة المتابعة للتوأم الرقمي في حالته الحالية."""
    inc = twin["incidents"]
    active = inc[inc["status"] != "تمت الاستجابة"]
    ov = twin["meta"].get("override", {})
    occ = occupancy_series(twin["hospitals"])
    resp = active["response_min"].dropna()
    avg_resp = float(ov["avg_response"]) if "avg_response" in ov else (float(resp.mean()) if len(resp) else float("nan"))
    avail = int(ov["amb_available"]) if "amb_available" in ov else int((twin["ambulances"]["status"] == "متاحة").sum())
    if "readiness" in ov:
        rd = ov["readiness"]
    else:
        amb = twin["ambulances"]
        rd = calculate_system_readiness(amb, twin["hospitals"], twin["roads"], active, avg_resp)
    return {
        "active_incidents": int(len(active)), "available_ambulances": avail,
        "under_pressure": int((occ >= UNDER_PRESSURE_THRESHOLD).sum()),
        "avg_response": avg_resp, "avg_occupancy": float(occ.mean()), "max_occupancy": float(occ.max()),
        "traffic_index": traffic_index(twin["roads"]), "readiness": rd,
        "closed_roads": int((twin["roads"]["congestion"] == "مغلق").sum()),
    }


# ----------------------------------------------------------------------------
# إرسال مبدئي (معاينة) عند إنشاء البلاغ
# ----------------------------------------------------------------------------
def trips_for(patients: int, critical: int):
    """كل حالة حرجة في مركبة، وغير الحرجة مركبتان لكل رحلة كحد أقصى."""
    trips = [(1, True)] * critical
    non = patients - critical
    while non > 0:
        k = min(2, non)
        trips.append((k, False))
        non -= k
    return trips


def candidate_ambulances(twin: dict, incident, k: int = 5) -> pd.DataFrame:
    roads = RoadModel(twin["roads"])
    extra = COND_MULT[incident["road_condition"]]
    amb = twin["ambulances"]
    rows = []
    for _, a in amb[amb["status"] == "متاحة"].iterrows():
        t, _m = roads.route(a["lat"], a["lon"], incident["lat"], incident["lon"], extra)
        rows.append({"رقم المركبة": a["name"], "زمن الوصول المتوقع (د)": round(t, 1)})
    df = pd.DataFrame(rows)
    return df.sort_values("زمن الوصول المتوقع (د)").head(k).reset_index(drop=True) if len(df) else df


def preview_dispatch(twin: dict) -> dict:
    """نسخة للعرض: أقرب المركبات المتاحة تتحرك نحو البلاغات الجديدة (بدون تغيير الحالة المحفوظة)."""
    t = clone_twin(twin)
    roads = RoadModel(t["roads"])
    for idx, inc in pending_incidents(t).iterrows():
        extra = COND_MULT[inc["road_condition"]]
        first = None
        for _ in trips_for(int(inc["patients"]), int(inc["critical"])):
            av = t["ambulances"][t["ambulances"]["status"] == "متاحة"]
            if av.empty:
                break
            best_i, best_t = None, 1e9
            for i, a in av.iterrows():
                tt, _m = roads.route(a["lat"], a["lon"], inc["lat"], inc["lon"], extra)
                if tt < best_t:
                    best_i, best_t = i, tt
            t["ambulances"].loc[best_i, ["status", "avail_in", "mission"]] = ["في الطريق", best_t, f"بلاغ {inc['id']}"]
            first = best_t if first is None else first
        t["incidents"].loc[idx, "response_min"] = round(first if first is not None else MAX_WAIT, 1)
    return t


# ----------------------------------------------------------------------------
# المحاكاة (SimPy)
# ----------------------------------------------------------------------------
class _Sim:
    def __init__(self, twin: dict, strategy: str):
        self.env = simpy.Environment()
        self.strategy = strategy
        self.roads = RoadModel(twin["roads"])
        self.h = {int(r["id"]): dict(r) for r in twin["hospitals"].to_dict("records")}
        for v in self.h.values():
            v["overflow"] = 0
            v["received"] = 0
        self.a = {int(r["id"]): dict(r) for r in twin["ambulances"].to_dict("records")}
        self.incidents = twin["incidents"]
        self.log, self.responses, self.arrivals = [], [], []
        self.delivered = 0
        self.unserved = 0
        self.checkpoint_amb = None
        self.load_by_hospital = {}

    # -- أدوات
    def emit(self, text, inc, trip, step, **kw):
        self.log.append(dict(t=self.env.now, text=text, inc=inc, trip=trip, step=step, **kw))

    def tm(self, la1, lo1, la2, lo2, extra):
        return self.roads.route(la1, lo1, la2, lo2, extra)

    def nearest_available(self, inc, extra):
        best, bt = None, 1e9
        for a in self.a.values():
            if a["status"] != "متاحة":
                continue
            t, _ = self.tm(a["lat"], a["lon"], inc["lat"], inc["lon"], extra)
            if t < bt:
                best, bt = a, t
        return best, bt

    def admit(self, hid, load, crit):
        """يحجز الأسرة؛ يعيد عدد الحالات التي لم تجد سريرًا."""
        h = self.h[hid]
        over = 0
        for _ in range(load):
            if crit and h["icu_avail"] > 0:
                h["icu_avail"] -= 1
            elif h["er_avail"] > 0:
                h["er_avail"] -= 1
            elif h["icu_avail"] > 0:
                h["icu_avail"] -= 1
            else:
                over += 1
        h["overflow"] += over
        h["received"] += load
        return over

    def choose_hospital(self, inc, load, crit, extra):
        opts = []
        for hid, h in self.h.items():
            t, m = self.tm(inc["lat"], inc["lon"], h["lat"], h["lon"], extra)
            opts.append((hid, h, t, m))
        if self.strategy == "nearest":
            hid, _, t, _ = min(opts, key=lambda o: o[2])
            return hid, t
        if self.strategy == "capacity":
            def free_key(o):
                h = o[1]
                return (h["icu_avail"] if crit else h["er_avail"], h["er_avail"] + h["icu_avail"])
            hid, _, t, _ = max(opts, key=free_key)
            return hid, t
        # التوزيع المتوازن: مستشفى مناسب التخصص وبه أسرة، ولا يتجاوز إشغاله 75% بعد الاستقبال (حد الضغط)،
        # مع عقوبة صغيرة على تكديس الحالات في مستشفى واحد.
        scored = []
        for hid, h, t, m in opts:
            r = calculate_hospital_score(h, inc, t, m, patients=load, critical=load if crit else 0)
            scored.append((hid, t, r))
        suitable = [s for s in scored if s[2]["specialty_ok"]
                    and (self.h[s[0]]["er_avail"] + self.h[s[0]]["icu_avail"]) >= load] or scored
        safe = [s for s in suitable if s[2]["post_occupancy"] <= 0.75] or suitable
        hid, t, _ = max(safe, key=lambda s: s[2]["score"] - 40.0 * max(0.0, s[2]["post_occupancy"] - 0.5)
                        - 3.0 * self.load_by_hospital.get(self.h[s[0]]["name"], 0))
        return hid, t

    # -- العمليات
    def release_ambulance(self, aid, delay):
        yield self.env.timeout(max(delay, 0.01))
        self.a[aid]["status"] = "متاحة"
        self.a[aid]["mission"] = "—"

    def committed_arrival(self, inc):
        yield self.env.timeout(float(inc["arrive_min"]))
        self.admit(int(inc["dest_hospital"]), int(inc["patients"]), int(inc["critical"]) > 0)

    def trip(self, inc, k, load, crit):
        env = self.env
        extra = COND_MULT[inc["road_condition"]]
        t0 = float(inc["offset"])
        yield env.timeout(t0)
        self.emit("تم استقبال البلاغ", inc["id"], k, "report")
        yield env.timeout(DISPATCH_MIN)
        while True:
            amb, t_to = self.nearest_available(inc, extra)
            if amb is not None:
                break
            if env.now - t0 > MAX_WAIT:
                self.unserved += load
                self.emit("تعذّر إسناد مركبة", inc["id"], k, "unserved")
                return
            yield env.timeout(1.0)
        amb["status"], amb["mission"] = "في الطريق", f"بلاغ {inc['id']}"
        self.emit(f"تم اختيار سيارة الإسعاف {amb['name']}", inc["id"], k, "assign", amb=amb["name"])
        yield env.timeout(t_to)
        amb["status"], amb["lat"], amb["lon"] = "في موقع الحادث", inc["lat"], inc["lon"]
        self.responses.append(env.now - t0)
        self.emit("وصلت سيارة الإسعاف إلى موقع الحادث", inc["id"], k, "arrive", amb=amb["name"])
        yield env.timeout(SCENE_CRIT_MIN if crit else SCENE_MIN)
        amb["status"] = "تنقل مصابًا"
        self.emit("بدأ نقل المصاب" if load == 1 else "بدأ نقل المصابين", inc["id"], k, "start")
        hid, t_h = self.choose_hospital(inc, load, crit, extra)
        h = self.h[hid]
        over = self.admit(hid, load, crit)
        self.load_by_hospital[h["name"]] = self.load_by_hospital.get(h["name"], 0) + load
        self.emit(f"تم اختيار {h['name']}", inc["id"], k, "choose", hospital=h["name"])
        yield env.timeout(t_h)
        penalty = OVERFLOW_DELAY if over else 0.0
        amb["lat"], amb["lon"], amb["status"] = h["lat"], h["lon"], "في المستشفى"
        self.arrivals.append(t_h + penalty)
        self.emit("وصل المصاب إلى المستشفى" if load == 1 else "وصل المصابون إلى المستشفى",
                  inc["id"], k, "hospital", hospital=h["name"])
        self.emit(f"تم تحديث القدرة الاستيعابية لـ {h['name']}", inc["id"], k, "update", hospital=h["name"])
        yield env.timeout(HANDOFF_MIN + penalty)
        amb["status"], amb["mission"] = "متاحة", "—"
        self.delivered += load

    def monitor(self):
        yield self.env.timeout(CHECKPOINT)
        self.checkpoint_amb = pd.DataFrame(self.a.values())[["name", "status"]]


def simulate_scenario(twin: dict, strategy: str) -> dict:
    """يشغّل محاكاة SimPy لاستراتيجية توجيه واحدة ويعيد المؤشرات والحالة النهائية."""
    S = _Sim(twin, strategy)
    env = S.env
    for aid, a in S.a.items():
        if a["status"] not in ("متاحة", OUT_OF_SERVICE):
            env.process(S.release_ambulance(aid, float(a["avail_in"])))
    inc_df = twin["incidents"]
    for _, inc in inc_df[inc_df["status"] == "قيد المعالجة"].iterrows():
        env.process(S.committed_arrival(inc))
    new = pending_incidents(twin).sort_values(["offset"], kind="stable")
    order = {"حرج": 0, "مرتفع": 1, "متوسط": 2, "منخفض": 3}
    new = new.assign(_o=new["severity"].map(order)).sort_values(["offset", "_o"], kind="stable")
    total_new_patients = int(new["patients"].sum())
    for _, inc in new.iterrows():
        for k, (load, crit) in enumerate(trips_for(int(inc["patients"]), int(inc["critical"]))):
            env.process(S.trip(inc.to_dict(), k, load, crit))
    env.process(S.monitor())
    env.run(until=SIM_END)

    hosp = pd.DataFrame(S.h.values())
    hosp["specialties"] = hosp["specialties"].apply(list)
    amb_final = pd.DataFrame(S.a.values())
    amb_final["avail_in"] = 0
    occ = occupancy_series(hosp)
    avg_resp = float(np.mean(S.responses)) if S.responses else float("nan")
    avg_arr = float(np.mean(S.arrivals)) if S.arrivals else float("nan")

    cp = S.checkpoint_amb if S.checkpoint_amb is not None else amb_final[["name", "status"]]
    cp_amb = cp.copy()
    amb_cp_available = int((cp_amb["status"] == "متاحة").sum())
    # المركبات عند نقطة القياس + السعة النهائية للمستشفيات (أثر القرار)
    active = twin["incidents"][twin["incidents"]["status"] != "تمت الاستجابة"]
    total_overflow = int(sum(v["overflow"] for v in S.h.values()))
    readiness = calculate_system_readiness(cp_amb, hosp, twin["roads"], active, avg_resp, total_overflow)
    recv = hosp[hosp["name"].isin(S.load_by_hospital.keys())]
    max_occ_recv = float(occupancy_series(recv).max()) if len(recv) else float(occ.max())

    twin_after = clone_twin(twin)
    twin_after["hospitals"] = hosp[twin["hospitals"].columns].reset_index(drop=True)
    twin_after["ambulances"] = amb_final[twin["ambulances"].columns].reset_index(drop=True)
    twin_after["incidents"] = twin_after["incidents"].assign(status="تمت الاستجابة")
    twin_after["meta"]["override"] = {"avg_response": avg_resp, "amb_available": amb_cp_available, "readiness": readiness}

    metrics = {
        "avg_arrival": avg_arr, "avg_response": avg_resp,
        "max_occupancy": max_occ_recv, "max_occupancy_all": float(occ.max()), "avg_occupancy": float(occ.mean()),
        "under_pressure": int((occ >= UNDER_PRESSURE_THRESHOLD).sum()),
        "amb_available": amb_cp_available, "readiness": readiness,
        "distributed": int(S.delivered), "requested": total_new_patients,
        "hospitals_used": len(S.load_by_hospital), "overflow": total_overflow,
        "unserved": int(S.unserved), "load_by_hospital": dict(S.load_by_hospital),
    }
    return {"strategy": strategy, "name": STRATEGIES[strategy], "metrics": metrics,
            "twin_after": twin_after, "log": S.log}


def simulate_all(twin: dict) -> dict:
    """يشغّل السيناريوهات الثلاثة ويعيد النتائج."""
    return {k: simulate_scenario(twin, k) for k in STRATEGIES}


def timeline_for(result: dict, incident_id: str, trip: int = 0):
    """الخط الزمني لأول رحلة في البلاغ (أول حالة حرجة إن وجدت)."""
    order = ["report", "assign", "arrive", "start", "choose", "hospital", "update"]
    ev = {e["step"]: e for e in result["log"] if e["inc"] == incident_id and e["trip"] == trip}
    out = []
    for s in order:
        if s in ev:
            out.append({"time": clock(ev[s]["t"]), "text": ev[s]["text"], "t": ev[s]["t"]})
    return out


# ----------------------------------------------------------------------------
# ماذا لو؟
# ----------------------------------------------------------------------------
WHAT_IFS = {
    "road": "ماذا لو أُغلق طريق رئيسي؟",
    "extra": "ماذا لو وصل 3 مصابين إضافيين؟",
    "full": "ماذا لو أصبح أحد المستشفيات ممتلئًا؟",
    "amb": "ماذا لو خرجت سيارتا إسعاف من الخدمة؟",
    "multi": "ماذا لو وقعت عدة حوادث في نفس الوقت؟",
}


def apply_what_if(twin: dict, kind: str, params: dict):
    """يعدّل نسخة من التوأم وفق سيناريو 'ماذا لو' ويعيد (التوأم المعدّل، وصف التغيير)."""
    t = clone_twin(twin)
    if kind == "road":
        name = params["road"]
        t["roads"].loc[t["roads"]["name"] == name, "congestion"] = "مغلق"
        t["roads"] = refresh_roads(t["roads"])
        return t, f"إغلاق {name} بالكامل"
    if kind == "extra":
        idx = pending_incidents(t).index[-1]
        t["incidents"].loc[idx, "patients"] = int(t["incidents"].loc[idx, "patients"]) + 3
        t["incidents"].loc[idx, "critical"] = int(t["incidents"].loc[idx, "critical"]) + 1
        return t, "وصول 3 مصابين إضافيين (بينهم حالة حرجة) إلى موقع البلاغ"
    if kind == "full":
        name = params["hospital"]
        t["hospitals"].loc[t["hospitals"]["name"] == name, ["er_avail", "icu_avail"]] = [0, 0]
        return t, f"امتلاء {name} بالكامل"
    if kind == "amb":
        inc = pending_incidents(t).iloc[-1]
        roads = RoadModel(t["roads"])
        av = t["ambulances"][t["ambulances"]["status"] == "متاحة"].copy()
        av["_t"] = [roads.route(a["lat"], a["lon"], inc["lat"], inc["lon"])[0] for _, a in av.iterrows()]
        out = av.sort_values("_t").head(2)
        t["ambulances"].loc[out.index, ["status", "mission", "avail_in"]] = [OUT_OF_SERVICE, "خارج الخدمة", 9999]
        return t, "خروج أقرب سيارتي إسعاف متاحتين من الخدمة: " + "، ".join(out["name"])
    if kind == "multi":
        specs = [
            dict(type="حادث مروري", severity="مرتفع", patients=3, critical=1, specialty="إصابات وحوادث",
                 place="الملز", road_condition="طبيعية", offset=0.0),
            dict(type="حريق", severity="مرتفع", patients=4, critical=1, specialty="حروق",
                 place="حطين", road_condition="طبيعية", offset=1.0),
        ]
        for s in specs:
            t = add_incident(t, s)
        return t, "وقوع حادثين إضافيين في الملز وحطين بالتزامن مع البلاغ الحالي"
    raise ValueError(kind)


def simulate_what_if(twin: dict, kind: str, params: dict, strategy: str = "balanced") -> dict:
    """يقارن قبل وبعد تطبيق التغيير على التوأم الرقمي."""
    before = simulate_scenario(twin, strategy)
    modified, desc = apply_what_if(twin, kind, params)
    after = simulate_scenario(modified, strategy)
    return {"kind": kind, "title": WHAT_IFS[kind], "description": desc, "strategy": strategy,
            "before": before, "after": after, "twin_modified": modified}
