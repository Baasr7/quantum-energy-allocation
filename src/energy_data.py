"""Stage 1 (v2): richer inputs for the energy allocation engine.

Everything here is DATA. The QUBO stays the same:
    Score(x) = sum_i c_i x_i + lam * (sum_i x_i - D)^2
only the numbers c_i (block cost) and the block counts change.

Units: cost = Rs thousand per block-hour (1 block = 10 MW for 1 h = 10,000 kWh).
All values are rough, India-flavoured assumptions. Change freely.
"""
import numpy as np
import pandas as pd

MW_PER_BLOCK = 10
BLOCK_KWH = MW_PER_BLOCK * 1000
WINDOW_H = 24            # resource limits are per 24 h scheduling window
DEMAND_MW = 100
DEMAND_SITE = "kochi"

# ---- tunable weights (Rs thousand per unit) --------------------------------
CARBON_PRICE = 2.0       # per tonne CO2  (Rs 2000/t shadow price)
POLLUTION_RATE = 6.0     # per (million people exposed x pollution index)
RISK_WEIGHT = 20.0       # per unit of risk (0..1)
PERCEPTION_WEIGHT = 10.0 # per unit of public opposition (1 - acceptance)
ALPHA = 1.0              # strength of weather priority on cost
DISCOUNT = 0.08          # for amortising infrastructure

# ---- RAW DATA 1: sites (location + weather) --------------------------------
#            lat    lon   alt_m  pop_M  sun_h uv  wind  cloud rain  p_rain snow
SITES_RAW = pd.DataFrame.from_dict({
    "bhadla":    (27.54, 71.92, 280, 0.1, 9.0, 9, 5.5, 15, 0.5, 10, 0),
    "muppandal": (8.25, 77.55, 100, 1.2, 7.5, 9, 7.5, 40, 2.0, 25, 0),
    "idukki":    (9.85, 76.97, 700, 0.3, 4.5, 6, 3.0, 75, 8.0, 70, 0),
    "talcher":   (20.95, 85.22, 150, 1.3, 6.0, 8, 3.5, 50, 4.0, 45, 0),
    "puthuvype": (10.01, 76.21, 5, 2.5, 5.5, 8, 3.5, 60, 6.0, 60, 0),
    "kochi":     (9.93, 76.27, 5, 3.0, 5.5, 8, 3.5, 60, 6.0, 60, 0),
}, orient="index", columns=["lat", "lon", "altitude_m", "population_m",
                            "sunlight_h", "uv_index", "wind_ms", "cloud_pct",
                            "rain_mm", "chance_rain_pct", "snow_cm"])
SITES_RAW.index.name = "site"


def _clip(x):
    return np.clip(x, 0.0, 1.0)


def solar_priority(s):
    base = (0.40 * s.sunlight_h / 10 + 0.15 * s.uv_index / 11
            + 0.25 * (1 - s.cloud_pct / 100) + 0.20 * (1 - s.chance_rain_pct / 100))
    return _clip(base * (1 - 0.5 * _clip(s.snow_cm / 20)))       # snow covers panels


def wind_priority(s):
    speed = _clip((s.wind_ms - 3) / (12 - 3))                    # cut-in 3, rated 12 m/s
    return _clip(speed * np.exp(-s.altitude_m / 8500))           # thinner air at altitude


def hydro_priority(s):
    return _clip(0.40 * s.rain_mm / 10 + 0.25 * s.chance_rain_pct / 100
                 + 0.20 * s.snow_cm / 30 + 0.15 * s.altitude_m / 1500)   # rain, melt, head


def _km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def build_sites(overrides: dict | None = None) -> pd.DataFrame:
    """Raw site table WITH weather priorities written into it.
    overrides = {"idukki": {"rain_mm": 1.0}} lets you test a weather scenario."""
    s = SITES_RAW.copy()
    for site, changes in (overrides or {}).items():
        for col, val in changes.items():
            s.loc[site, col] = val
    s["solar_priority"] = solar_priority(s).round(3)
    s["wind_priority"] = wind_priority(s).round(3)
    s["hydro_priority"] = hydro_priority(s).round(3)
    s["best_renewable"] = s[["solar_priority", "wind_priority", "hydro_priority"]] \
        .idxmax(axis=1).str.replace("_priority", "")
    d = s.loc[DEMAND_SITE]
    s["air_km_to_demand"] = _km(s.lat, s.lon, d.lat, d.lon).round(0)
    return s


# ---- RAW DATA 2: transport modes -------------------------------------------
# rate = Rs per unit per km, fixed = Rs thousand per block-hour (terminals),
# route = road/rail distance / straight-line distance, co2 = kg per unit-km
MODES = pd.DataFrame([
    ("power", "HVAC",     2.7, 0.0, 1.25, 0.00),
    ("power", "HVDC",     1.8, 1.2, 1.25, 0.00),   # cheaper per km, converter cost up front
    ("coal",  "rail",     1.5, 0.0, 1.25, 0.03),
    ("coal",  "road",     4.0, 0.0, 1.35, 0.10),
    ("gas",   "pipeline", 0.6, 0.6, 1.25, 0.01),
    ("gas",   "tanker",   6.0, 0.3, 1.35, 0.10),
], columns=["cargo", "mode", "rate", "fixed", "route", "co2"])

# ---- RAW DATA 3: plants ------------------------------------------------------
# Thermal plants sit at the load centre (fuel is hauled in); renewables at the resource.
PLANTS = [
    dict(source="solar", site="bhadla", origin="bhadla", cargo="power", cargo_per_block=1,
         installed=2, marginal=0.5, capex_cr_mw=4.5, life=25, co2=0.040, pollution=0.02,
         acceptance=0.90, risk=0.05, kind="solar"),
    dict(source="wind", site="muppandal", origin="muppandal", cargo="power", cargo_per_block=1,
         installed=3, marginal=0.8, capex_cr_mw=6.5, life=25, co2=0.011, pollution=0.02,
         acceptance=0.70, risk=0.10, kind="wind"),
    dict(source="hydro", site="idukki", origin="idukki", cargo="power", cargo_per_block=1,
         installed=3, marginal=0.8, capex_cr_mw=12.0, life=50, co2=0.024, pollution=0.05,
         acceptance=0.55, risk=0.20, kind="hydro",
         resource="water", catchment_km2=40, runoff=0.5, stock=0.30, head_m=550),   # Mm3
    dict(source="coal", site="kochi", origin="talcher", cargo="coal", cargo_per_block=6.5,
         installed=4, marginal=3.0, capex_cr_mw=8.0, life=30, co2=0.95, pollution=1.00,
         acceptance=0.30, risk=0.40, kind=None,
         resource="coal", supply=250, stock=350, per_block_hr=6.5),                 # tonnes
    dict(source="gas", site="kochi", origin="puthuvype", cargo="gas", cargo_per_block=1.43,
         installed=3, marginal=6.5, capex_cr_mw=5.0, life=25, co2=0.45, pollution=0.25,
         acceptance=0.50, risk=0.50, kind=None,
         resource="gas", supply=80, stock=60, per_block_hr=1.43),                   # tonnes LNG
    dict(source="battery", site="kochi", origin="kochi", cargo="power", cargo_per_block=1,
         installed=2, marginal=1.5, capex_cr_mw=10.0, life=15, co2=0.050, pollution=0.0,
         acceptance=0.80, risk=0.15, kind=None,
         resource="electricity", supply=0, stock=40, per_block_hr=10, run_hours=2), # MWh
]


def _amortised_infra(capex_cr_mw, life):
    crf = DISCOUNT * (1 + DISCOUNT) ** life / ((1 + DISCOUNT) ** life - 1)
    return capex_cr_mw * 1e7 * MW_PER_BLOCK * crf / 8760 / 1000


def build_sources(overrides: dict | None = None) -> pd.DataFrame:
    """One row per source. cost_per_block / blocks_available / capacity_mw keep the
    names qubo_builder.py expects; the extra columns explain where the cost came from."""
    sites = build_sites(overrides)
    rows = []
    for p in PLANTS:
        plant, origin = sites.loc[p["site"]], sites.loc[p["origin"]]
        prio = plant[f"{p['kind']}_priority"] if p["kind"] else 0.5      # 0.5 = neutral
        phi = 1 + ALPHA * (0.5 - prio)                                    # priority factor

        # transport: pick the mode with lowest (cost + carbon cost)
        dist = _km(origin.lat, origin.lon, plant.lat, plant.lon) if p["cargo"] != "power" \
            else _km(origin.lat, origin.lon, *sites.loc[DEMAND_SITE, ["lat", "lon"]])
        best = None
        for _, m in MODES[MODES.cargo == p["cargo"]].iterrows():
            km = dist * m.route
            cost = (m.rate * p["cargo_per_block"] * km) / 1000 + m.fixed
            co2_t = m.co2 * p["cargo_per_block"] * km / 1000
            tot = cost + CARBON_PRICE * co2_t
            if best is None or tot < best[0]:
                best = (tot, m["mode"], cost, co2_t)
        _, mode, transport, co2_transport_t = best

        generation = p["marginal"] * 10
        infra = _amortised_infra(p["capex_cr_mw"], p["life"])
        co2_t = p["co2"] * BLOCK_KWH / 1000 + co2_transport_t
        carbon = CARBON_PRICE * co2_t
        pollution = POLLUTION_RATE * p["pollution"] * plant.population_m
        risk = RISK_WEIGHT * p["risk"]
        perception = PERCEPTION_WEIGHT * (1 - p["acceptance"])
        cost = (generation + infra) * phi + transport + carbon + pollution + risk + perception

        # blocks: installed -> weather derating -> resource limit
        weather_factor = 0.3 + 0.7 * prio if p["kind"] else 1.0
        b_weather = int(np.floor(p["installed"] * weather_factor + 0.5))
        b_res = p["installed"]
        if p.get("resource"):
            if p["resource"] == "water":
                per_hr = BLOCK_KWH * 3.6e6 / (1000 * 9.81 * p["head_m"] * 0.9) / 1e6   # Mm3
                supply = plant.rain_mm * p["catchment_km2"] * p["runoff"] * 1e-3      # Mm3/day
            else:
                per_hr, supply = p["per_block_hr"], p["supply"]
            need = per_hr * p.get("run_hours", WINDOW_H)
            b_res = int(np.floor((supply + p["stock"]) / need))
        blocks = max(0, min(p["installed"], b_weather, b_res))

        rows.append(dict(source=p["source"], site=p["site"], mode=mode,
                         priority=round(float(prio), 3), priority_factor=round(float(phi), 3),
                         generation=generation, infra=round(infra, 2),
                         transport=round(transport, 2), carbon=round(carbon, 2),
                         pollution=round(pollution, 2), risk=risk, perception=perception,
                         cost_per_block=round(cost, 2), co2_t_per_block=round(co2_t, 2),
                         blocks_installed=p["installed"], blocks_weather=b_weather,
                         blocks_resource=b_res, blocks_available=blocks))
    df = pd.DataFrame(rows)
    df["mw_per_block"] = MW_PER_BLOCK
    df["capacity_mw"] = df["blocks_available"] * df["mw_per_block"]
    return df


def build_switches(sources: pd.DataFrame) -> pd.DataFrame:
    """One row per switch (= one block = one qubit)."""
    rows = [{"switch": f"{r.source}_{k}", "source": r.source, "cost": r.cost_per_block}
            for r in sources.itertuples() for k in range(int(r.blocks_available))]
    sw = pd.DataFrame(rows)
    sw.index.name = "qubit"
    return sw


def validate(sources: pd.DataFrame, demand_mw: int = DEMAND_MW):
    """Returns demand in blocks; raises if the scenario is impossible."""
    mw = sources["mw_per_block"].iloc[0]
    if demand_mw <= 0 or demand_mw % mw:
        raise ValueError(f"Demand {demand_mw} MW must be a positive multiple of {mw} MW")
    if (sources[["cost_per_block", "blocks_available"]] < 0).any().any():
        raise ValueError("Negative cost or blocks")
    if demand_mw > sources["capacity_mw"].sum():
        raise ValueError(f"Infeasible: demand {demand_mw} MW > capacity "
                         f"{sources['capacity_mw'].sum()} MW")
    return demand_mw // mw


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("SITES (raw weather + priorities):")
    print(build_sites().drop(columns=["lat", "lon"]).to_string(), "\n")
    src = build_sources()
    print("SOURCES (cost breakdown, Rs thousand per block-hour):")
    print(src[["source", "site", "mode", "priority", "generation", "infra", "transport",
               "carbon", "pollution", "risk", "perception", "cost_per_block"]].to_string(index=False), "\n")
    print("BLOCKS (installed -> weather -> resource -> final):")
    print(src[["source", "blocks_installed", "blocks_weather", "blocks_resource",
               "blocks_available"]].to_string(index=False), "\n")
    print(f"Demand {DEMAND_MW} MW = {validate(src)} blocks | qubits = {len(build_switches(src))}")

    dry = build_sources({"idukki": {"rain_mm": 1.0, "chance_rain_pct": 10}})
    print("\nSCENARIO: dry spell at Idukki ->",
          dry.set_index("source").loc["hydro", ["priority", "cost_per_block", "blocks_available"]].to_dict())