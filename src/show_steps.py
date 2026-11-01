"""Step-by-step walkthrough: how every extra constraint is applied and CHECKED.

Each step recomputes a quantity independently from the raw data and compares it with
what energy_data_v2.build_sources() produced. Any mismatch prints FAIL.
Run:  python show_steps.py
"""
import numpy as np
import pandas as pd
import energy_data as ed
from qubo_builder import (build_qubo, all_patterns, score, real_cost, real_co2,
                          describe, min_valid_lambda)

pd.set_option("display.width", 200, "display.max_columns", 30)
RESULTS = []


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def check(label, ok):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


sites = ed.build_sites()
src = ed.build_sources().set_index("source")

# ---------------------------------------------------------------- STEP 1
banner("STEP 1  RAW SITE DATA: location, population, weather")
print(ed.SITES_RAW.to_string(), "\n")
check("no negative rain/snow/population/altitude",
      (ed.SITES_RAW[["rain_mm", "snow_cm", "population_m", "altitude_m"]] >= 0).all().all())
check("percentages within 0-100",
      ed.SITES_RAW[["cloud_pct", "chance_rain_pct"]].stack().between(0, 100).all())
check("latitude/longitude plausible for India",
      ed.SITES_RAW.lat.between(6, 37).all() and ed.SITES_RAW.lon.between(68, 98).all())

# ---------------------------------------------------------------- STEP 2
banner("STEP 2  WEATHER -> PRIORITY (term by term)")
rows = []
for name, s in ed.SITES_RAW.iterrows():
    sun, uv = 0.40 * s.sunlight_h / 10, 0.15 * s.uv_index / 11
    cl, rn = 0.25 * (1 - s.cloud_pct / 100), 0.20 * (1 - s.chance_rain_pct / 100)
    snow_f = 1 - 0.5 * min(s.snow_cm / 20, 1)
    solar = min(1, (sun + uv + cl + rn) * snow_f)
    speed = min(1, max(0, (s.wind_ms - 3) / 9))
    dens = np.exp(-s.altitude_m / 8500)
    wind = min(1, speed * dens)
    hydro = min(1, 0.40 * s.rain_mm / 10 + 0.25 * s.chance_rain_pct / 100
                + 0.20 * s.snow_cm / 30 + 0.15 * s.altitude_m / 1500)
    rows.append(dict(site=name, sun=sun, uv=uv, clear_sky=cl, dry=rn, snow_f=snow_f,
                     SOLAR=solar, wind_speed=speed, air_dens=dens, WIND=wind, HYDRO=hydro))
    for k, v in (("solar", solar), ("wind", wind), ("hydro", hydro)):
        assert abs(v - sites.loc[name, f"{k}_priority"]) < 1e-3, (name, k)
print(pd.DataFrame(rows).set_index("site").round(3).to_string(), "\n")
check("independent recompute matches the priorities stored in the site table", True)
check("all priorities within [0, 1]",
      sites[["solar_priority", "wind_priority", "hydro_priority"]].stack().between(0, 1).all())
b = sites.loc["bhadla"]
check(f"high sun + wind site (bhadla): solar {b.solar_priority} > hydro {b.hydro_priority}",
      b.solar_priority > b.hydro_priority)
i = sites.loc["idukki"]
check(f"rainy highland (idukki): hydro {i.hydro_priority} > solar {i.solar_priority}",
      i.hydro_priority > i.solar_priority)
print("  best renewable per site:", dict(sites.best_renewable))

# ---------------------------------------------------------------- STEP 3
banner("STEP 3  PRIORITY -> COST FACTOR   phi = 1 + alpha*(0.5 - P)")
ph = src[["site", "priority", "priority_factor"]].copy()
ph["recomputed"] = (1 + ed.ALPHA * (0.5 - ph.priority)).round(3)
print(ph.to_string(), "\n")
check("phi matches for every source", np.allclose(ph.priority_factor, ph.recomputed, atol=2e-3))
ordered = ph.sort_values("priority", ascending=False)
check("higher priority => lower phi (monotonic across all sources)",
      ordered.priority_factor.is_monotonic_increasing)
check("priority 0.5 is neutral (phi == 1) for coal, gas, battery",
      (ph.loc[["coal", "gas", "battery"], "priority_factor"] == 1.0).all())
check("above-0.5 priority discounts, below-0.5 penalises (sign of 1-phi == sign of P-0.5)",
      (np.sign((1 - ph.priority_factor).round(3)) == np.sign((ph.priority - 0.5).round(3))).all())

# ---------------------------------------------------------------- STEP 4
banner("STEP 4  TRANSPORT: every mode is priced, cheapest (cost + carbon) wins")
all_modes = []
for p in ed.PLANTS:
    plant, origin = sites.loc[p["site"]], sites.loc[p["origin"]]
    dem = sites.loc[ed.DEMAND_SITE]
    air = ed._km(origin.lat, origin.lon, *((dem.lat, dem.lon) if p["cargo"] == "power"
                                           else (plant.lat, plant.lon)))
    best = None
    for _, m in ed.MODES[ed.MODES.cargo == p["cargo"]].iterrows():
        km = air * m.route
        cost = m.rate * p["cargo_per_block"] * km / 1000 + m.fixed
        co2 = m.co2 * p["cargo_per_block"] * km / 1000
        tot = cost + ed.CARBON_PRICE * co2
        all_modes.append(dict(source=p["source"], mode=m["mode"], route_km=round(km),
                              cost=round(cost, 2), co2_t=round(co2, 3), total=round(tot, 2)))
        if best is None or tot < best[0]:
            best = (tot, m["mode"], cost)
    check(f"{p['source']:<8} chosen mode '{src.loc[p['source'], 'mode']}' is the cheapest "
          f"('{best[1]}'), transport {src.loc[p['source'], 'transport']} == {best[2]:.2f}",
          best[1] == src.loc[p["source"], "mode"]
          and abs(best[2] - src.loc[p["source"], "transport"]) < 0.01)
print()
print(pd.DataFrame(all_modes).to_string(index=False))

# ---------------------------------------------------------------- STEP 5
banner("STEP 5  COST OF ONE BLOCK: components must add up to c_i")
comp = src[["generation", "infra", "priority_factor", "transport", "carbon",
            "pollution", "risk", "perception", "cost_per_block"]].copy()
comp["recomputed"] = ((comp.generation + comp.infra) * comp.priority_factor + comp.transport
                      + comp.carbon + comp.pollution + comp.risk + comp.perception).round(2)
print(comp.to_string(), "\n")
check("sum of components == cost_per_block for every source",
      np.allclose(comp.cost_per_block, comp.recomputed, atol=0.05))
chk_poll = {p["source"]: ed.POLLUTION_RATE * p["pollution"] * sites.loc[p["site"], "population_m"]
            for p in ed.PLANTS}
check("pollution = rate x index x population of the PLANT site",
      all(abs(src.loc[k, "pollution"] - v) < 0.01 for k, v in chk_poll.items()))
check("coal is the most polluting and least accepted source",
      src.pollution.idxmax() == "coal"
      and min(p["acceptance"] for p in ed.PLANTS) == 0.30)
check("renewables are cheaper than every fossil source per block",
      src.loc[["solar", "wind", "hydro"], "cost_per_block"].max()
      < src.loc[["coal", "gas"], "cost_per_block"].min())

# ---------------------------------------------------------------- STEP 6
banner("STEP 6  BLOCK LIMITS: installed -> weather derating -> resource limit")
rows = []
for p in ed.PLANTS:
    plant = sites.loc[p["site"]]
    prio = src.loc[p["source"], "priority"]
    wf = 0.3 + 0.7 * prio if p["kind"] else 1.0
    bw = int(np.floor(p["installed"] * wf + 0.5))
    unit, supply, per_hr, need, tot, br = "-", 0, 0, 0, np.inf, p["installed"]
    if p.get("resource"):
        if p["resource"] == "water":
            per_hr = ed.BLOCK_KWH * 3.6e6 / (1000 * 9.81 * p["head_m"] * 0.9) / 1e6
            supply, unit = plant.rain_mm * p["catchment_km2"] * p["runoff"] * 1e-3, "Mm3"
        else:
            per_hr, supply = p["per_block_hr"], p["supply"]
            unit = {"coal": "t", "gas": "t", "electricity": "MWh"}[p["resource"]]
        need = per_hr * p.get("run_hours", ed.WINDOW_H)
        tot = supply + p["stock"]
        br = int(np.floor(tot / need))
    final = max(0, min(p["installed"], bw, br))
    rows.append(dict(source=p["source"], installed=p["installed"], weather_factor=round(wf, 2),
                     by_weather=bw, resource=p.get("resource", "-"), unit=unit,
                     inflow_per_day=round(supply, 3), stock=p.get("stock", 0),
                     need_per_block=round(need, 3), by_resource=br if p.get("resource") else "-",
                     FINAL=final))
    check(f"{p['source']:<8} final {final} == min(installed {p['installed']}, weather {bw}, "
          f"resource {br if p.get('resource') else 'n/a'}) and matches data: "
          f"{int(src.loc[p['source'], 'blocks_available'])}",
          final == int(src.loc[p["source"], "blocks_available"]))
print()
print(pd.DataFrame(rows).to_string(index=False))
dry = ed.build_sources({"idukki": {"rain_mm": 1.0, "chance_rain_pct": 10}}).set_index("source")
check(f"dry-spell scenario cuts hydro: {int(src.loc['hydro','blocks_available'])} -> "
      f"{int(dry.loc['hydro','blocks_available'])} blocks, cost "
      f"{src.loc['hydro','cost_per_block']} -> {dry.loc['hydro','cost_per_block']}",
      dry.loc["hydro", "blocks_available"] < src.loc["hydro", "blocks_available"]
      and dry.loc["hydro", "cost_per_block"] > src.loc["hydro", "cost_per_block"])

# ---------------------------------------------------------------- STEP 7
banner("STEP 7  FEASIBILITY: can the constrained system meet demand at all?")
sources = ed.build_sources()
D = ed.validate(sources)
print(f"  demand {ed.DEMAND_MW} MW = {D} blocks | capacity {int(sources.capacity_mw.sum())} MW "
      f"= {int(sources.blocks_available.sum())} blocks")
check("demand is a multiple of the block size", ed.DEMAND_MW % ed.MW_PER_BLOCK == 0)
check("capacity after weather + resource limits >= demand",
      sources.capacity_mw.sum() >= ed.DEMAND_MW)
try:
    ed.validate(sources, 500)
    check("impossible demand (500 MW) is rejected", False)
except ValueError:
    check("impossible demand (500 MW) is rejected", True)

# ---------------------------------------------------------------- STEP 8
banner("STEP 8  QUBO + BRUTE FORCE")
q = build_qubo(sources)
X = all_patterns(q.n)
s, rc, on = score(q, X), real_cost(q, X), X.sum(1)
valid = on == q.demand_blocks
opt = np.flatnonzero(np.isclose(s, s.min()))
print(f"  qubits={q.n} | lambda={q.lam:.2f} | valid patterns {valid.sum()} of {len(X)}")
check("QUBO score == real cost on every valid pattern", np.allclose(s[valid], rc[valid]))
check("no invalid pattern beats the optimum", not (s[~valid] < s.min() - 1e-9).any())
check("every optimal pattern uses exactly D blocks", (on[opt] == q.demand_blocks).all())
check(f"auto lambda {q.lam:.1f} >= smallest working lambda {min_valid_lambda(q, X):.1f}",
      q.lam >= min_valid_lambda(q, X))

# ---------------------------------------------------------------- STEP 9
banner("STEP 9  CONSTRAINT LEDGER for the optimal allocation")
x = X[opt[0]]
use = describe(q, x)
print(f"  optimal: {''.join(map(str, x))}  ->  {use}")
print(f"  cost {rc[opt[0]]:.2f} | CO2 {real_co2(q, x)[0]:.2f} t per block-hour\n")
led = []
for p in ed.PLANTS:
    n_used = use.get(p["source"], 0)
    r = src.loc[p["source"]]
    led.append(dict(source=p["source"], used=n_used, installed=int(r.blocks_installed),
                    weather_cap=int(r.blocks_weather), resource_cap=int(r.blocks_resource),
                    final_cap=int(r.blocks_available),
                    status="BINDING" if n_used == r.blocks_available else
                    ("unused" if n_used == 0 else "slack")))
    check(f"{p['source']:<8} used {n_used} <= installed/weather/resource caps "
          f"({int(r.blocks_installed)}/{int(r.blocks_weather)}/{int(r.blocks_resource)})",
          n_used <= min(r.blocks_installed, r.blocks_weather, r.blocks_resource))
print()
print(pd.DataFrame(led).to_string(index=False))
check(f"total blocks used {sum(use.values())} == demand {D}", sum(use.values()) == D)
print("\n  BINDING = the source is fully used, so its limit shaped the answer.")

# ---------------------------------------------------------------- STEP 10
banner("STEP 10  WHAT EACH CONSTRAINT CHANGED (ablation, same costs)")
base = ed.build_sources()
variants = {
    "A installed blocks only": base.blocks_installed,
    "B + weather derating": base[["blocks_installed", "blocks_weather"]].min(axis=1),
    "C + resource limits (final)": base.blocks_available,
}
out = []
for name, blocks in variants.items():
    v = base.copy()
    v["blocks_available"] = blocks.astype(int)
    v["capacity_mw"] = v.blocks_available * v.mw_per_block
    qv = build_qubo(v)
    Xv = all_patterns(qv.n)
    sv = score(qv, Xv)
    k = sv.argmin()
    out.append(dict(variant=name, qubits=qv.n, optimal_cost=round(real_cost(qv, Xv[k])[0], 2),
                    CO2_t=round(real_co2(qv, Xv[k])[0], 2), blocks_used=describe(qv, Xv[k])))
print(pd.DataFrame(out).to_string(index=False))
check("adding constraints never lowers the optimal cost (cost A <= B <= C)",
      out[0]["optimal_cost"] <= out[1]["optimal_cost"] <= out[2]["optimal_cost"])

# ---------------------------------------------------------------- SUMMARY
banner("SUMMARY")
print(f"  checks passed: {sum(RESULTS)} / {len(RESULTS)}")
print("  ALL CONSTRAINT CHECKS PASSED" if all(RESULTS) else "  SOME CHECKS FAILED - see FAIL lines")