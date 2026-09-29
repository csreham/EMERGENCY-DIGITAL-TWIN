# -*- coding: utf-8 -*-
"""
recommendation.py — خوارزمية توصية المستشفى (قائمة على قواعد شفافة، بدون تعلّم آلي).
الأوزان: 30% قدرة استيعابية | 25% زمن الوصول | 20% توافق التخصص | 15% حالة الطريق | 10% القدرة بعد الاستقبال
"""
import pandas as pd
from data import RoadModel, COND_MULT, hospital_occupancy

WEIGHTS = {"capacity": 0.30, "time": 0.25, "specialty": 0.20, "road": 0.15, "post": 0.10}
WEIGHT_LABELS = {
    "capacity": "القدرة الاستيعابية", "time": "زمن الوصول", "specialty": "توافق التخصص",
    "road": "حالة الطريق", "post": "القدرة بعد الاستقبال",
}
DISCLAIMER = "توصية لدعم القرار — وليست قرارًا طبيًا مستقلاً"


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def calculate_hospital_score(h, inc, travel_min, route_mult, patients=None, critical=None) -> dict:
    """يحسب نتيجة المستشفى من 0 إلى 100 مع تفصيل المكوّنات وأسباب التوصية."""
    n = int(inc["patients"] if patients is None else patients)
    c = int(inc["critical"] if critical is None else critical)
    non = n - c
    total = h["er_beds"] + h["icu_beds"]
    used = total - h["er_avail"] - h["icu_avail"]
    occ = hospital_occupancy(h)

    # 1) القدرة الاستيعابية: الحرجة تحتاج عناية مركزة، وغيرها أسرة طوارئ
    icu_fit = 1.0 if c == 0 else min(1.0, h["icu_avail"] / c)
    er_fit = 1.0 if non == 0 else min(1.0, h["er_avail"] / non)
    fit = (c * icu_fit + non * er_fit) / max(n, 1)
    capacity = 100 * (0.6 * fit + 0.4 * (1 - occ))

    # 2) زمن الوصول: 3 دقائق = 100، 30 دقيقة = 0
    time_s = 100 * _clamp(1 - (travel_min - 3) / 27)

    # 3) توافق التخصص
    specialty_ok = inc["specialty"] in h["specialties"]
    specialty = 100.0 if specialty_ok else 35.0

    # 4) حالة الطريق: معامل 1.0 = سالك، 2.2 فأكثر = شديد الازدحام
    road = 100 * _clamp(1 - (route_mult - 1) / 1.2)

    # 5) القدرة المتوقعة بعد استقبال الحالة
    post_occ = (used + n) / total
    post = 0.0 if used + n > total else 100 * _clamp(1 - max(0.0, post_occ - 0.5) / 0.5)

    comps = {"capacity": capacity, "time": time_s, "specialty": specialty, "road": road, "post": post}
    score = sum(WEIGHTS[k] * v for k, v in comps.items())

    reasons = []
    reasons.append((capacity >= 65, "قدرة استيعابية مناسبة" if capacity >= 65 else "قدرة استيعابية محدودة"))
    if c > 0:
        reasons.append((h["icu_avail"] >= c, "أسرة عناية مركزة كافية للحالات الحرجة" if h["icu_avail"] >= c
                        else "أسرة العناية المركزة أقل من عدد الحالات الحرجة"))
    reasons.append((specialty_ok, "التخصص المطلوب متوفر" if specialty_ok else "التخصص المطلوب غير متوفر بالكامل"))
    reasons.append((time_s >= 55, f"زمن وصول مقبول ({travel_min:.1f} دقيقة)" if time_s >= 55
                    else f"زمن وصول طويل نسبيًا ({travel_min:.1f} دقيقة)"))
    reasons.append((road >= 55, "الطريق أقل ازدحامًا" if road >= 55 else "الطريق مزدحم"))
    reasons.append((post >= 55, "الضغط المتوقع بعد الاستقبال منخفض" if post >= 55
                    else "الضغط المتوقع بعد الاستقبال مرتفع"))
    return {
        "score": round(score, 1), "components": comps, "reasons": reasons,
        "travel_min": travel_min, "post_occupancy": min(1.0, post_occ), "specialty_ok": specialty_ok,
    }


def recommend_hospital(twin: dict, incident) -> dict:
    """يقيّم كل المستشفيات ويعيد الأفضل مع ترتيب كامل."""
    roads = RoadModel(twin["roads"])
    extra = COND_MULT[incident["road_condition"]]
    rows, results = [], []
    for _, h in twin["hospitals"].iterrows():
        tmin, mult = roads.route(incident["lat"], incident["lon"], h["lat"], h["lon"], extra)
        r = calculate_hospital_score(h, incident, tmin, mult)
        r["hospital"] = h["name"]
        r["id"] = int(h["id"])
        results.append(r)
        rows.append({
            "المستشفى": h["name"], "النتيجة": r["score"], "زمن الوصول (د)": round(tmin, 1),
            "القدرة الاستيعابية": round(r["components"]["capacity"]), "زمن الوصول ": round(r["components"]["time"]),
            "التخصص": round(r["components"]["specialty"]), "الطريق": round(r["components"]["road"]),
            "بعد الاستقبال": round(r["components"]["post"]),
        })
    ranking = pd.DataFrame(rows).sort_values("النتيجة", ascending=False).reset_index(drop=True)
    best = max(results, key=lambda r: r["score"])
    return {"best": best, "ranking": ranking, "results": results, "disclaimer": DISCLAIMER}
