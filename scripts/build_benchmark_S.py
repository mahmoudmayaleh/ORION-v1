#!/usr/bin/env python
"""§S Track B benchmark builder — 100 authored intents + K^A-consistent gold specs.

Intents are hand-authored (below); gold specs are DERIVED deterministically from the
K^A reference tables mirrored from src/orion/sim/slice_generator.py (SFC templates,
per-VNF vcr/intensity/tiers/cpu-ram, QoS bands, and v4 Eq.3 bandwidth law
  beta_{k,k+1} = beta_in * prod_{j<=k} rho_{f_j}).
No model is involved in authoring. Output frozen to data/benchmark_S/ with a SHA-256.

Run: python scripts/build_benchmark_S.py   (pure stdlib; no API, no LLM)
"""
import json, hashlib
from pathlib import Path

# ── K^A reference tables (mirror of slice_generator._VNF_TEMPLATES / _QOS_PROFILES) ──
TEMPLATES = {
    "eMBB": [
        {"type": "Firewall", "cpu": [2, 4], "ram": [2, 8],  "intensity": 0.8, "vcr": 1.0, "tiers": ["mec", "regional_cloud", "central_cloud"]},
        {"type": "CDN",      "cpu": [4, 8], "ram": [8, 16], "intensity": 1.2, "vcr": 0.7, "tiers": ["mec", "regional_cloud"]},
        {"type": "vEPC",     "cpu": [4, 8], "ram": [4, 16], "intensity": 1.0, "vcr": 1.0, "tiers": ["regional_cloud", "central_cloud"]},
    ],
    "URLLC": [
        {"type": "Firewall", "cpu": [1, 2], "ram": [1, 4], "intensity": 0.5, "vcr": 1.0, "tiers": ["ran_edge", "mec"]},
        {"type": "vUPF",     "cpu": [2, 4], "ram": [2, 8], "intensity": 0.6, "vcr": 1.0, "tiers": ["ran_edge", "mec"]},
    ],
    "mMTC": [
        {"type": "IoTGateway", "cpu": [1, 2], "ram": [1, 4],  "intensity": 0.4, "vcr": 0.3, "tiers": ["ran_edge", "mec"]},
        {"type": "Aggregator", "cpu": [2, 4], "ram": [2, 8],  "intensity": 0.6, "vcr": 0.5, "tiers": ["mec", "regional_cloud"]},
        {"type": "Analytics",  "cpu": [4, 8], "ram": [8, 16], "intensity": 1.5, "vcr": 1.0, "tiers": ["regional_cloud", "central_cloud"]},
    ],
    "V2X": [
        {"type": "Firewall",      "cpu": [1, 2], "ram": [1, 4], "intensity": 0.5, "vcr": 1.0, "tiers": ["ran_edge", "mec"]},
        {"type": "V2XController", "cpu": [2, 4], "ram": [4, 8], "intensity": 0.7, "vcr": 1.0, "tiers": ["mec"]},
        {"type": "vEPC",          "cpu": [2, 4], "ram": [2, 8], "intensity": 1.0, "vcr": 1.0, "tiers": ["regional_cloud"]},
    ],
    "XR": [
        {"type": "Firewall",  "cpu": [2, 4],  "ram": [2, 8],   "intensity": 0.8, "vcr": 1.0, "tiers": ["mec"]},
        {"type": "MediaProc", "cpu": [8, 16], "ram": [16, 32], "intensity": 2.0, "vcr": 1.2, "tiers": ["mec", "regional_cloud"]},
        {"type": "CDN",       "cpu": [4, 8],  "ram": [8, 16],  "intensity": 1.2, "vcr": 0.7, "tiers": ["regional_cloud", "central_cloud"]},
        {"type": "vEPC",      "cpu": [2, 4],  "ram": [2, 8],   "intensity": 1.0, "vcr": 1.0, "tiers": ["central_cloud"]},
    ],
}
QOS = {
    "eMBB":  {"delay": [20.0, 100.0], "beta_in": [50.0, 500.0]},
    "URLLC": {"delay": [1.0, 10.0],   "beta_in": [10.0, 50.0]},
    "mMTC":  {"delay": [50.0, 500.0], "beta_in": [1.0, 10.0]},
    "V2X":   {"delay": [5.0, 20.0],   "beta_in": [20.0, 80.0]},
    "XR":    {"delay": [5.0, 30.0],   "beta_in": [100.0, 500.0]},
}


def gold_for(slice_type, n_vnfs, beta_in, delay, note=None, defensible=None):
    """Derive the K^A-consistent gold spec. n_vnfs>=2 and <=len(template)."""
    tmpls = TEMPLATES[slice_type][:n_vnfs]
    vnfs = [{"vnf_type": t["type"], "permitted_tiers": t["tiers"],
             "computational_intensity": t["intensity"], "vcr": t["vcr"],
             "cpu_demand_range": t["cpu"], "ram_demand_range": t["ram"]} for t in tmpls]
    edges = []
    for k in range(len(vnfs) - 1):
        prod = 1.0
        for j in range(k + 1):
            prod *= tmpls[j]["vcr"]
        edges.append({"from": tmpls[k]["type"], "to": tmpls[k + 1]["type"],
                      "bandwidth_demand": round(beta_in * prod, 1)})
    g = {"slice_type": slice_type,
         "sfc": [t["type"] for t in tmpls],
         "n_vnfs": n_vnfs,
         "vnfs": vnfs,
         "beta_in": beta_in,
         "qos": {"max_e2e_delay": delay, "min_throughput": beta_in},
         "flow_edges": edges}
    if note:
        g["disambiguation_note"] = note
    if defensible:
        # Two-tier scoring for the ambiguous stratum: tier-1 (strict) = slice_type above;
        # tier-2 (lenient) = ANY class in this enumerated set, defensible from the intent
        # text alone. An explicit clarification/abstention is scored SEPARATELY (never as a
        # wrong class). Enumerated NOW, not judged later.
        g["defensible_classes"] = defensible
    return g


# ── Authored intents. (stratum, slice_type, n_vnfs, beta_in, delay, text) ──
# beta_in / delay are chosen within each class's K^A band; text is natural language.
INTENTS = []

def add(stratum, st, n, beta, delay, text, note=None, defensible=None):
    INTENTS.append((stratum, st, n, beta, delay, text, note, defensible))

# ---- eMBB x25 (delay 20-100, beta 50-500, chain Firewall[->CDN[->vEPC]]) ----
add("eMBB","eMBB",3,300.0,40.0,"Stand up a mobile video-streaming slice for a stadium crowd: about 300 Mbps downlink, keep end-to-end latency under 40 ms, with firewalling, a CDN cache, and packet-core.")
add("eMBB","eMBB",3,500.0,50.0,"We need a premium 4K streaming service, ~500 Mbps per user aggregate, 50 ms budget, full chain through firewall, CDN and the EPC.")
add("eMBB","eMBB",2,120.0,60.0,"Lightweight web-browsing slice, 120 Mbps, 60 ms is fine, just firewall then cache — no core needed.")
add("eMBB","eMBB",3,250.0,35.0,"Enterprise VPN + media slice at 250 Mbps, target 35 ms, secured, cached, routed through packet core.")
add("eMBB","eMBB",2,80.0,90.0,"Basic broadband access, 80 Mbps, latency up to 90 ms acceptable, firewall and CDN only.")
add("eMBB","eMBB",3,450.0,45.0,"High-throughput live-event uplink/downlink, roughly 450 Mbps, sub-45 ms, firewall + CDN + vEPC.")
add("eMBB","eMBB",3,200.0,55.0,"Consumer OTT video slice, 200 Mbps, 55 ms, standard eMBB chain end to end.")
add("eMBB","eMBB",2,150.0,70.0,"Social-media heavy mobile slice, 150 Mbps, 70 ms, firewall to CDN.")
add("eMBB","eMBB",3,350.0,30.0,"Low-latency cloud-gaming-adjacent streaming, 350 Mbps, tight 30 ms, all three functions.")
add("eMBB","eMBB",3,500.0,100.0,"Bulk content distribution overnight, 500 Mbps, relaxed 100 ms, firewall, CDN and core.")
add("eMBB","eMBB",2,60.0,80.0,"Budget mobile data plan slice, 60 Mbps, 80 ms, firewall and cache.")
add("eMBB","eMBB",3,275.0,48.0,"Campus-wide streaming for lecture halls, 275 Mbps, 48 ms, full eMBB chain.")
add("eMBB","eMBB",3,400.0,42.0,"Sports-bar multi-screen feed, 400 Mbps, 42 ms, firewalled, cached, cored.")
add("eMBB","eMBB",2,100.0,65.0,"Retail in-store media slice, 100 Mbps, 65 ms, firewall then CDN.")
add("eMBB","eMBB",3,320.0,38.0,"Concert-venue AR overlay streaming, 320 Mbps, 38 ms, three-function chain.")
add("eMBB","eMBB",3,180.0,75.0,"Regional IPTV distribution, 180 Mbps, 75 ms, firewall + CDN + vEPC.")
add("eMBB","eMBB",2,90.0,85.0,"Hotel guest Wi-Fi offload, 90 Mbps, 85 ms, just firewall and cache.")
add("eMBB","eMBB",3,500.0,25.0,"Flagship 8K demo slice, 500 Mbps, aggressive 25 ms, full chain.")
add("eMBB","eMBB",3,230.0,52.0,"City-park public streaming kiosk, 230 Mbps, 52 ms, all three VNFs.")
add("eMBB","eMBB",2,140.0,58.0,"Coffee-shop chain media slice, 140 Mbps, 58 ms, firewall to CDN.")
add("eMBB","eMBB",3,370.0,44.0,"Stadium replay-on-demand, 370 Mbps, 44 ms, firewall, CDN, packet core.")
add("eMBB","eMBB",3,290.0,33.0,"Newsroom live-feed ingest, 290 Mbps, 33 ms, complete eMBB chain.")
add("eMBB","eMBB",2,110.0,95.0,"Rural broadband fill-in, 110 Mbps, 95 ms tolerable, firewall + CDN.")
add("eMBB","eMBB",3,420.0,47.0,"E-sports arena spectator streams, 420 Mbps, 47 ms, three functions.")
add("eMBB","eMBB",3,160.0,62.0,"Shopping-mall digital-signage slice, 160 Mbps, 62 ms, full chain.")

# ---- URLLC x25 (delay 1-10, beta 10-50, chain Firewall->vUPF) ----
add("URLLC","URLLC",2,20.0,2.0,"Factory-floor robotic control slice: 20 Mbps, ultra-low 2 ms latency, firewall and a local user-plane function at the edge.")
add("URLLC","URLLC",2,15.0,1.0,"Motion-control loop for a robotic arm, 15 Mbps, 1 ms hard budget, firewall + vUPF pinned to the edge.")
add("URLLC","URLLC",2,30.0,5.0,"Remote-surgery haptics link, 30 Mbps, 5 ms, secured with edge user-plane.")
add("URLLC","URLLC",2,25.0,3.0,"AGV fleet coordination in a warehouse, 25 Mbps, 3 ms, firewall then vUPF.")
add("URLLC","URLLC",2,10.0,4.0,"Protective-relay teleprotection for a substation, 10 Mbps, 4 ms, edge core.")
add("URLLC","URLLC",2,40.0,8.0,"Cooperative-robotics cell, 40 Mbps, 8 ms, firewall and user-plane at edge.")
add("URLLC","URLLC",2,18.0,2.0,"Closed-loop PLC synchronization, 18 Mbps, 2 ms, firewall + vUPF.")
add("URLLC","URLLC",2,50.0,10.0,"Port-crane remote operation, 50 Mbps, 10 ms, edge user-plane secured.")
add("URLLC","URLLC",2,22.0,3.0,"Assembly-line vision-guided actuation, 22 Mbps, 3 ms, firewall to vUPF.")
add("URLLC","URLLC",2,12.0,1.0,"Precision motor sync, 12 Mbps, 1 ms, edge firewall and user-plane.")
add("URLLC","URLLC",2,35.0,6.0,"Teleoperated forklift, 35 Mbps, 6 ms, firewall + vUPF at the edge.")
add("URLLC","URLLC",2,28.0,4.0,"Grid-balancing fast control, 28 Mbps, 4 ms, edge core chain.")
add("URLLC","URLLC",2,16.0,2.0,"Haptic-feedback glove link, 16 Mbps, 2 ms, firewall then user-plane.")
add("URLLC","URLLC",2,45.0,9.0,"Autonomous mining truck control, 45 Mbps, 9 ms, edge vUPF.")
add("URLLC","URLLC",2,24.0,5.0,"Robotic welding cell, 24 Mbps, 5 ms, firewall and vUPF.")
add("URLLC","URLLC",2,14.0,3.0,"Drone-swarm formation control, 14 Mbps, 3 ms, edge firewall + user-plane.")
add("URLLC","URLLC",2,32.0,7.0,"Remote-driving test track, 32 Mbps, 7 ms, firewall to edge vUPF.")
add("URLLC","URLLC",2,19.0,2.0,"Bottling-line sync, 19 Mbps, 2 ms, edge core.")
add("URLLC","URLLC",2,26.0,6.0,"Smart-grid inverter control, 26 Mbps, 6 ms, firewall + vUPF.")
add("URLLC","URLLC",2,48.0,8.0,"Teleoperated excavator, 48 Mbps, 8 ms, edge user-plane.")
add("URLLC","URLLC",2,13.0,1.0,"Servo-drive coordination, 13 Mbps, 1 ms, firewall then vUPF at edge.")
add("URLLC","URLLC",2,38.0,10.0,"AR-guided maintenance with control loop, 38 Mbps, 10 ms, edge core.")
add("URLLC","URLLC",2,21.0,4.0,"Conveyor-sortation actuation, 21 Mbps, 4 ms, firewall + user-plane.")
add("URLLC","URLLC",2,29.0,5.0,"Robotic palletizer, 29 Mbps, 5 ms, edge vUPF.")
add("URLLC","URLLC",2,17.0,3.0,"CNC-machine tele-control, 17 Mbps, 3 ms, firewall to vUPF.")

# ---- mMTC x20 (delay 50-500, beta 1-10, chain IoTGateway[->Aggregator[->Analytics]]) ----
add("mMTC","mMTC",3,5.0,200.0,"City-wide smart-meter collection: ~5 Mbps aggregate, latency up to 200 ms, gateway, aggregation, then analytics in the cloud.")
add("mMTC","mMTC",2,2.0,300.0,"Agricultural soil-sensor mesh, 2 Mbps, 300 ms fine, IoT gateway then aggregator.")
add("mMTC","mMTC",3,8.0,150.0,"Industrial predictive-maintenance sensors, 8 Mbps, 150 ms, full gateway-aggregator-analytics chain.")
add("mMTC","mMTC",3,10.0,100.0,"Smart-building environmental telemetry, 10 Mbps, 100 ms, three-stage mMTC chain.")
add("mMTC","mMTC",2,1.0,500.0,"Remote wildlife trackers, 1 Mbps, 500 ms tolerable, gateway and aggregator.")
add("mMTC","mMTC",3,6.0,250.0,"Fleet-telematics ingestion, 6 Mbps, 250 ms, gateway, aggregate, analytics.")
add("mMTC","mMTC",2,3.0,400.0,"Parking-occupancy sensors, 3 Mbps, 400 ms, IoT gateway to aggregator.")
add("mMTC","mMTC",3,7.0,180.0,"Utility grid-edge monitoring, 7 Mbps, 180 ms, full three-function chain.")
add("mMTC","mMTC",3,9.0,120.0,"Cold-chain logistics sensors, 9 Mbps, 120 ms, gateway-aggregator-analytics.")
add("mMTC","mMTC",2,2.5,350.0,"Street-lighting controllers, 2.5 Mbps, 350 ms, gateway then aggregator.")
add("mMTC","mMTC",3,4.0,220.0,"Water-network leak sensors, 4 Mbps, 220 ms, three-stage chain.")
add("mMTC","mMTC",2,1.5,450.0,"Structural-health strain gauges, 1.5 Mbps, 450 ms, gateway + aggregator.")
add("mMTC","mMTC",3,10.0,90.0,"Warehouse asset-tracking tags, 10 Mbps, 90 ms, gateway, aggregate, analytics.")
add("mMTC","mMTC",3,5.5,200.0,"Environmental air-quality network, 5.5 Mbps, 200 ms, full mMTC chain.")
add("mMTC","mMTC",2,3.5,300.0,"Smart-bin fill-level sensors, 3.5 Mbps, 300 ms, gateway to aggregator.")
add("mMTC","mMTC",3,8.5,140.0,"Manufacturing vibration analytics, 8.5 Mbps, 140 ms, three functions.")
add("mMTC","mMTC",2,2.0,380.0,"Livestock health tags, 2 Mbps, 380 ms, IoT gateway and aggregator.")
add("mMTC","mMTC",3,6.5,160.0,"Pipeline pressure telemetry, 6.5 Mbps, 160 ms, gateway-aggregator-analytics.")
add("mMTC","mMTC",3,7.5,110.0,"Retail shelf-sensor network, 7.5 Mbps, 110 ms, full chain.")
add("mMTC","mMTC",2,1.0,500.0,"Seismic monitoring array, 1 Mbps, 500 ms, gateway then aggregator.")

# ---- V2X x10 (delay 5-20, beta 20-80, chain Firewall->V2XController[->vEPC]) ----
add("V2X","V2X",3,50.0,10.0,"Intersection collision-avoidance service: 50 Mbps, 10 ms, firewall, a V2X controller at the edge, and packet core.")
add("V2X","V2X",2,30.0,8.0,"Platooning coordination for trucks, 30 Mbps, 8 ms, firewall and V2X controller.")
add("V2X","V2X",3,70.0,12.0,"Highway hazard-warning broadcast, 70 Mbps, 12 ms, firewall + controller + vEPC.")
add("V2X","V2X",2,40.0,6.0,"Emergency-vehicle green-wave signaling, 40 Mbps, 6 ms, firewall to V2X controller.")
add("V2X","V2X",3,60.0,15.0,"Cooperative-perception sensor sharing, 60 Mbps, 15 ms, three-function V2X chain.")
add("V2X","V2X",2,25.0,7.0,"Vulnerable-road-user alerts, 25 Mbps, 7 ms, firewall and controller.")
add("V2X","V2X",3,80.0,18.0,"City-wide traffic-orchestration slice, 80 Mbps, 18 ms, firewall, controller, core.")
add("V2X","V2X",2,35.0,9.0,"Roadworks lane-merge assist, 35 Mbps, 9 ms, firewall to V2X controller.")
add("V2X","V2X",3,55.0,11.0,"Autonomous-shuttle route coordination, 55 Mbps, 11 ms, full V2X chain.")
add("V2X","V2X",2,45.0,5.0,"Rail-crossing warning, 45 Mbps, tight 5 ms, firewall and controller.")

# ---- XR x10 (delay 5-30, beta 100-500, chain Firewall->MediaProc[->CDN[->vEPC]]) ----
add("XR","XR",3,300.0,20.0,"Multiplayer AR game in a theme park: 300 Mbps, 20 ms, firewall, media processing, and a CDN edge.")
add("XR","XR",4,500.0,15.0,"Full VR telepresence suite, 500 Mbps, 15 ms, firewall, media proc, CDN, and packet core.")
add("XR","XR",2,150.0,25.0,"Lightweight AR maintenance overlay, 150 Mbps, 25 ms, firewall then media processing.")
add("XR","XR",3,400.0,18.0,"Cloud-rendered VR showroom, 400 Mbps, 18 ms, firewall, media proc, CDN.")
add("XR","XR",4,450.0,12.0,"Immersive multiplayer arena, 450 Mbps, 12 ms, all four XR functions.")
add("XR","XR",3,250.0,22.0,"Museum AR exhibit, 250 Mbps, 22 ms, firewall, media proc, CDN.")
add("XR","XR",2,120.0,28.0,"Assisted-assembly AR glasses, 120 Mbps, 28 ms, firewall to media processing.")
add("XR","XR",4,500.0,10.0,"Flagship VR concert, 500 Mbps, aggressive 10 ms, firewall, media proc, CDN, core.")
add("XR","XR",3,350.0,16.0,"Collaborative 3D design space, 350 Mbps, 16 ms, three-function XR chain.")
add("XR","XR",2,200.0,30.0,"Retail virtual try-on, 200 Mbps, 30 ms, firewall + media processing.")

# ---- Mixed / ambiguous x10 (gold = intended disambiguation + enumerated defensible set) ----
add("ambiguous","URLLC",2,20.0,3.0,"I need the network to react basically instantly for a robot control loop - a few Mbps is plenty.",
    "Latency-dominant, sub-ms-class control -> URLLC despite low stated rate; beta_in set to band low-mid (20 Mbps).",
    defensible=["URLLC"])
add("ambiguous","eMBB",3,300.0,45.0,"Give me a fast slice for lots of video to lots of people, don't care too much about delay.",
    "Throughput-dominant, delay-tolerant, many users -> eMBB full chain; delay mid-band (45 ms).",
    defensible=["eMBB"])
add("ambiguous","mMTC",3,5.0,250.0,"Thousands of tiny sensors trickling data all day; it can be slow but must scale massively.",
    "Massive low-rate device count, delay-tolerant -> mMTC; full gateway-aggregator-analytics chain.",
    defensible=["mMTC"])
add("ambiguous","V2X",3,50.0,10.0,"Cars talking to the roadside to avoid crashes, needs to be quick but it's outdoor vehicular.",
    "Vehicular safety, moderate rate + low delay -> V2X (V2XController is the signature VNF); URLLC also defensible on the latency-criticality reading.",
    defensible=["V2X", "URLLC"])
add("ambiguous","XR",3,300.0,20.0,"Headset experience with heavy rendering, both high bandwidth AND low latency.",
    "Joint high-throughput + low-delay + rendering -> XR (MediaProc); eMBB defensible if the translator lacks an XR concept and reads it as high-bw video.",
    defensible=["XR", "eMBB"])
add("ambiguous","URLLC",2,25.0,5.0,"Critical alarm signaling - small payloads but it absolutely cannot be late.",
    "Reliability/latency-critical, tiny payload -> URLLC; firewall + vUPF at edge.",
    defensible=["URLLC"])
add("ambiguous","eMBB",2,100.0,70.0,"Just normal internet for an office, nothing special.",
    "Generic broadband, no latency/scale signal -> default eMBB, short chain (firewall+CDN).",
    defensible=["eMBB"])
add("ambiguous","mMTC",2,3.0,400.0,"Battery-powered meters phoning home once in a while.",
    "Low-power, sporadic, low-rate -> mMTC; short gateway+aggregator chain.",
    defensible=["mMTC"])
add("ambiguous","V2X",2,35.0,8.0,"Roadside units warning nearby vehicles of hazards.",
    "Roadside-to-vehicle hazard warning -> V2X; URLLC also defensible on the safety-latency reading.",
    defensible=["V2X", "URLLC"])
add("ambiguous","XR",4,500.0,12.0,"The most demanding immersive experience you can build, spare nothing.",
    "Max-demand immersive -> XR full 4-function chain; eMBB defensible as a high-bw video reading.",
    defensible=["XR", "eMBB"])


def main():
    outdir = Path("data/benchmark_S")
    outdir.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (stratum, st, n, beta, delay, text, note, defensible) in enumerate(INTENTS):
        items.append({"id": f"S{i:03d}", "stratum": stratum, "intent": text,
                      "gold": gold_for(st, n, float(beta), float(delay), note, defensible)})
    # ── Band check (per senior request): for every edge whose VCR product != 1.0 (i.e. the
    # item is SUPPOSED to test VCR-awareness), confirm the naive VCR-ignoring answer (beta_in)
    # falls OUTSIDE the gold's +-15% bandwidth tolerance band. If it sneaks in, the item stops
    # testing VCR — flag it. Edges with product == 1.0 are legitimate flat-VCR controls.
    BW_TOL = 0.15
    print("===== BAND CHECK (bw-edge tolerance = +-15% of gold) =====")
    flagged = 0
    for it in items:
        g = it["gold"]; beta = g["beta_in"]
        prod = 1.0
        for k, e in enumerate(g["flow_edges"]):
            # reconstruct cumulative product up to this edge from gold vcr chain
            prod *= g["vnfs"][k]["vcr"]
            gold_bw = e["bandwidth_demand"]
            if abs(prod - 1.0) < 1e-9:
                continue  # flat-VCR control edge; naive == gold by design
            lo, hi = gold_bw * (1 - BW_TOL), gold_bw * (1 + BW_TOL)
            naive_inside = lo <= beta <= hi
            near_boundary = 0.85 <= prod <= 1.15
            if naive_inside or near_boundary:
                flagged += 1
                print(f"  FLAG {it['id']} {e['from']}->{e['to']}: prod={prod:.3f} gold={gold_bw} "
                      f"band=[{lo:.1f},{hi:.1f}] naive(beta_in)={beta} "
                      f"{'INSIDE' if naive_inside else 'near-boundary'}")
    print(f"  band-check: {flagged} VCR-testing edge(s) flagged "
          f"(0 = every VCR item excludes the naive answer)\n")

    # Freeze the EXACT file bytes (UTF-8) and hash THOSE, so the box verifies identically
    # by re-hashing the file. No canonical/indent ambiguity.
    content = json.dumps(items, indent=2, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    (outdir / "benchmark_S.json").write_bytes(content)
    (outdir / "MANIFEST.txt").write_text(
        f"benchmark_S_sha256={digest}\nn_items={len(items)}\n"
        + "hash_method=sha256(benchmark_S.json bytes, utf-8, indent=2, ensure_ascii=False)\n"
        + "strata=" + json.dumps({s: sum(1 for it in items if it['stratum'] == s)
                                  for s in ['eMBB','URLLC','mMTC','V2X','XR','ambiguous']}) + "\n",
        encoding="utf-8")
    # strata counts
    from collections import Counter
    cnt = Counter(it["stratum"] for it in items)
    print(f"benchmark_S_sha256 = {digest[:16]}  (full in MANIFEST.txt)")
    print(f"n_items = {len(items)}   strata = {dict(cnt)}")
    print(f"written -> {outdir}/benchmark_S.json  +  MANIFEST.txt")
    # 5 sample items, one per single-class stratum, for the checkpoint read
    print("\n===== 5 SAMPLE ITEMS (one per stratum) =====")
    seen = set()
    for it in items:
        s = it["stratum"]
        if s in seen or s == "ambiguous":
            continue
        seen.add(s)
        g = it["gold"]
        print(f"\n[{it['id']}] stratum={s}")
        print(f"  intent: {it['intent']}")
        print(f"  gold.slice_type = {g['slice_type']}   sfc = {g['sfc']}")
        print(f"  gold.qos = delay<={g['qos']['max_e2e_delay']}ms  min_throughput={g['qos']['min_throughput']}Mbps (=beta_in)")
        print(f"  gold.flow_edges (bw via Eq.3) = " +
              ", ".join(f"{e['from']}->{e['to']}:{e['bandwidth_demand']}" for e in g['flow_edges']))
        print(f"  gold.vnf vcr/tiers = " +
              " | ".join(f"{v['vnf_type']}(vcr={v['vcr']},tiers={v['permitted_tiers']})" for v in g['vnfs']))
        if len(seen) == 5:
            break
    # one ambiguous exemplar too
    amb = next(it for it in items if it["stratum"] == "ambiguous")
    print(f"\n[{amb['id']}] stratum=ambiguous (6th, disambiguation)")
    print(f"  intent: {amb['intent']}")
    print(f"  gold.slice_type = {amb['gold']['slice_type']}   sfc = {amb['gold']['sfc']}")
    print(f"  note: {amb['gold']['disambiguation_note']}")


if __name__ == "__main__":
    main()
