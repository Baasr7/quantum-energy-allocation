"""Serve the interactive, local energy atlas dashboard."""
from __future__ import annotations

import csv
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _solver_results(csv_path: Path, scenarios: list[dict[str, Any]],
                   results_dir: Path | None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if results_dir:
        for scenario in scenarios:
            result_path = results_dir / f"{scenario['name']}.json"
            if result_path.exists():
                with result_path.open(encoding="utf-8") as stream:
                    results[scenario["name"]] = json.load(stream)
    if len(results) == len(scenarios):
        return results

    # Running the dashboard directly (outside run_pipeline.ps1) still provides
    # real classical and QAOA results. The pipeline passes saved results to avoid
    # repeating those calculations after it has already run each scenario.
    import sys
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from main import allocate, load_scenario
    import energy_data as ed
    import qubo_builder as qb
    from qaoa_solver import solve_qaoa

    for scenario in scenarios:
        if scenario["name"] in results:
            continue
        sources, demand_mw, block_mw = load_scenario(csv_path, scenario["name"])
        demand_blocks = ed.validate(sources, demand_mw)
        q, greedy, optimum = allocate(sources, demand_blocks, block_mw)
        qaoa = solve_qaoa(q, p=1, shots=512, maxiter=50)
        qaoa_bits = [int(bit) for bit in qaoa["bitstring"]]

        def summary(bits: list[int]) -> dict[str, Any]:
            block_counts = qb.describe(q, bits)
            allocations = {name: int(count * block_mw) for name, count in block_counts.items()}
            return {
                "allocation_mw": allocations,
                "cost_per_hour": float(qb.real_cost(q, bits)[0]),
                "co2_tonnes_per_hour": float(qb.real_co2(q, bits)[0]),
                "feasible": sum(block_counts.values()) * block_mw == demand_mw,
            }

        method_results = {
            "Greedy": summary(greedy),
            "Brute force": summary(optimum),
            "QAOA sample": summary(qaoa_bits),
        }
        exact_cost = method_results["Brute force"]["cost_per_hour"]
        qaoa_cost = method_results["QAOA sample"]["cost_per_hour"]
        results[scenario["name"]] = {
            "scenario": scenario["name"], "demand_mw": demand_mw,
            "mw_per_block": block_mw, "qubits": q.n,
            "methods": method_results,
            "qaoa_cost_gap_pct": (qaoa_cost - exact_cost) / exact_cost * 100 if exact_cost else 0.0,
            "qaoa_diagnostics": qaoa["diagnostics"],
        }
    return results


def build_dashboard_data(results_dir: str | Path | None = None) -> dict[str, Any]:
    try:
        from . import energy_data as ed
    except ImportError:
        import energy_data as ed

    sites_frame = ed.build_sites()
    sites = {}
    for name, row in sites_frame.iterrows():
        sites[name] = {
            "name": name,
            "lat": float(row.lat), "lon": float(row.lon),
            "sunlight_h": float(row.sunlight_h), "uv_index": float(row.uv_index),
            "wind_ms": float(row.wind_ms), "cloud_pct": float(row.cloud_pct),
            "rain_mm": float(row.rain_mm), "chance_rain_pct": float(row.chance_rain_pct),
            "altitude_m": float(row.altitude_m), "population_m": float(row.population_m),
            "solar_priority": float(row.solar_priority),
            "wind_priority": float(row.wind_priority),
            "hydro_priority": float(row.hydro_priority),
            "best_renewable": str(row.best_renewable),
        }

    sources_frame = ed.build_sources()
    sources_by_name = sources_frame.set_index("source")
    sources = []
    for plant in ed.PLANTS:
        row = sources_by_name.loc[plant["source"]]
        sources.append({
            "source": plant["source"], "site": plant["site"], "origin": plant["origin"],
            "cargo": plant["cargo"], "mode": str(row["mode"]),
            "capacity_mw": float(row["capacity_mw"]),
            "cost_per_block": float(row["cost_per_block"]),
            "co2_t_per_block": float(row["co2_t_per_block"]),
            "blocks_available": int(row["blocks_available"]),
        })

    csv_path = Path(__file__).resolve().parents[1] / "data" / "energy_scenarios.csv"
    scenarios: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["scenario"]
            scenario = scenarios.setdefault(name, {
                "name": name, "demand_mw": int(row["demand_mw"]),
                "mw_per_block": int(row["mw_per_block"]), "sources": [],
            })
            scenario["sources"].append({
                "source": row["source"], "capacity_mw": float(row["capacity_mw"]),
                "cost_per_block": float(row["cost_per_block"]),
                "co2_t_per_block": float(row["co2_t_per_block"]),
            })

    scenario_list = list(scenarios.values())
    result_path = Path(results_dir) if results_dir else None
    solver_results = _solver_results(csv_path, scenario_list, result_path)

    return {
        "demand_site": ed.DEMAND_SITE,
        "model_demand_mw": int(ed.DEMAND_MW),
        "sites": sites,
        "sources": sources,
        "scenarios": scenario_list,
        "solver_results": solver_results,
    }


def serve_dashboard(host: str = "127.0.0.1", port: int = 8765,
                    open_browser: bool = True,
                    results_dir: str | Path | None = None) -> None:
    page_path = Path(__file__).with_name("dashboard.html")
    if results_dir:
        result_path = Path(results_dir)
        print(f"Loading scenario solver results from {result_path}", flush=True)
    else:
        print("Preparing classical and QAOA comparisons for the dashboard...", flush=True)
    dashboard_data = build_dashboard_data(results_dir)

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path in ("/", "/index.html"):
                body = page_path.read_bytes()
                content_type = "text/html; charset=utf-8"
                status = 200
            elif self.path == "/api/data":
                body = json.dumps(dashboard_data, separators=(",", ":")).encode("utf-8")
                content_type = "application/json; charset=utf-8"
                status = 200
            else:
                body = b"Not found"
                content_type = "text/plain; charset=utf-8"
                status = 404
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{server.server_port}/"
    print(f"Energy Atlas dashboard running at {url}", flush=True)
    print("Keep this window open while using the dashboard; press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the Energy Atlas dashboard.", flush=True)
    finally:
        server.server_close()
