#!/usr/bin/env python3
"""
MWVRP Solver with road-connected routing + auto-tuning
- Greedy order assignment (simple but fast)
- Multi-warehouse allocation (up to N warehouses per order)
- Dijkstra shortest paths between waypoints
- One route per vehicle; starts/ends at home depot
- Auto-tuning loop tries multiple parameter sets and picks the best
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq, random


# ------------------- ENTRY POINT -------------------

def solver(env) -> Dict:
    """
    Entry point required by hackathon.
    Runs auto-tuning loop over parameter sets and picks best solution.
    """

    param_grid = [
        {"capacity_buffer": 0.95, "max_warehouses": 1, "order_strategy": "largest"},
        {"capacity_buffer": 0.90, "max_warehouses": 2, "order_strategy": "smallest"},
        {"capacity_buffer": 1.00, "max_warehouses": 2, "order_strategy": "random"},
    ]

    best_solution = None
    best_score = float("inf")

    for params in param_grid:
        env.reset_all_state()

        solution = base_solver(env,
                               capacity_buffer=params["capacity_buffer"],
                               max_warehouses=params["max_warehouses"],
                               order_strategy=params["order_strategy"])

        validation_result = env.validate_solution_complete(solution)

# If it’s a boolean:
        if isinstance(validation_result, bool):
          if not validation_result:
            continue

# If it’s a dict with an "is_valid" key:
        elif isinstance(validation_result, dict):
          if not validation_result.get("is_valid", True):
            continue


        cost = env.calculate_solution_cost(solution)
        fulfillment = env.get_solution_fulfillment_summary(solution)

        requested, delivered = 0, 0

# Case 1: top-level dict with "requested"/"delivered"
        if "requested" in fulfillment and "delivered" in fulfillment:
    # values are already dicts or ints
          if isinstance(fulfillment["requested"], dict):
            requested = sum(fulfillment["requested"].values())
          else:
            requested = int(fulfillment["requested"])
          if isinstance(fulfillment["delivered"], dict):
            delivered = sum(fulfillment["delivered"].values())
          else:
            delivered = int(fulfillment["delivered"])

# Case 2: per-order dict with ints
        else:
          for _oid, f in fulfillment.items():
        # f might itself be a dict or just an int
            if isinstance(f, dict):
              requested += sum(f.get("requested", {}).values()) if isinstance(f.get("requested"), dict) else int(f.get("requested", 0))
              delivered += sum(f.get("delivered", {}).values()) if isinstance(f.get("delivered"), dict) else int(f.get("delivered", 0))
            elif isinstance(f, int):
            # if it's just an int, treat it as requested count
              requested += f

        stats = env.get_solution_statistics(solution)
        requested = stats.get("total_items_requested", 0)
        delivered = stats.get("total_items_delivered", 0)
        fulfillment_rate = delivered / max(1, requested)

        # Hackathon scoring equation approximation (lower is better)
        score = cost + cost * (100 - (fulfillment_rate * 100))

        if score < best_score:
            best_score = score
            best_solution = solution

    return best_solution if best_solution else {"routes": []}


# ------------------- BASE SOLVER -------------------

def base_solver(env, capacity_buffer=0.95, max_warehouses=2, order_strategy="largest") -> Dict:
    """
    Greedy solver with road-connected routing.
    Parameters:
        capacity_buffer: fraction of vehicle capacity to allow
        max_warehouses: maximum warehouses to split an order across
        order_strategy: 'largest', 'smallest', or 'random'
    """

    orders = env.orders
    vehicles = {v.id: v for v in env.get_all_vehicles()}
    warehouses = env.warehouses
    skus = env.skus
    graph = env.get_road_network_data()
    adjacency = graph.get('adjacency_list', {})

    # Shadow inventory
    inventory = {wid: dict(wh.inventory) for wid, wh in warehouses.items()}

    # Track vehicle aggregate loads and assigned orders
    vehicle_loads = {vid: {'weight': 0.0, 'volume': 0.0, 'orders': []}
                     for vid in vehicles.keys()}

    # Order sorting strategy
    order_items = list(orders.items())
    if order_strategy == "largest":
        order_items.sort(
            key=lambda kv: sum(skus[s].weight * q for s, q in kv[1].requested_items.items()),
            reverse=True
        )
    elif order_strategy == "smallest":
        order_items.sort(
            key=lambda kv: sum(skus[s].weight * q for s, q in kv[1].requested_items.items())
        )
    elif order_strategy == "random":
        random.shuffle(order_items)

    # Assign orders greedily
    for order_id, order in order_items:
        order_weight = sum(skus[s].weight * q for s, q in order.requested_items.items())
        order_volume = sum(skus[s].volume * q for s, q in order.requested_items.items())

        allocation = _find_simple_allocation(order, inventory, warehouses, skus, max_warehouses)
        if not allocation:
            continue

        assigned_vehicle = None
        for vid, vehicle in vehicles.items():
            load = vehicle_loads[vid]
            if (load['weight'] + order_weight <= vehicle.capacity_weight * capacity_buffer and
                load['volume'] + order_volume <= vehicle.capacity_volume * capacity_buffer):
                assigned_vehicle = vid
                break

        if assigned_vehicle is None:
            continue

        vehicle_loads[assigned_vehicle]['weight'] += order_weight
        vehicle_loads[assigned_vehicle]['volume'] += order_volume
        vehicle_loads[assigned_vehicle]['orders'].append((order_id, allocation))

        # Update shadow inventory
        for wh_id, items in allocation:
            for sku_id, qty in items.items():
                inventory[wh_id][sku_id] -= qty

    # Build physically connected routes
    solution = {"routes": []}
    for vid, info in vehicle_loads.items():
        if not info['orders']:
            continue
        vehicle = vehicles[vid]
        route = _build_connected_route(env, vehicle, info['orders'], warehouses, orders, adjacency)
        if route:
            solution["routes"].append(route)

    return solution


# ------------------- HELPERS -------------------

def _find_simple_allocation(order, inventory, warehouses, skus, max_warehouses=2):
    """
    Quick allocation: try single warehouse first, then split (up to max_warehouses).
    Returns: [(warehouse_id, {sku_id: qty}), ...] covering full order; else [].
    """
    needed = dict(order.requested_items)

    # Try single warehouse
    for wh_id, inv in inventory.items():
        if all(inv.get(sku_id, 0) >= qty for sku_id, qty in needed.items()):
            return [(wh_id, needed)]

    # Split across up to max_warehouses
    allocation = []
    remaining = dict(needed)

    for wh_id, inv in inventory.items():
        if len(allocation) >= max_warehouses:
            break

        provided = {}
        to_delete = []
        for sku_id, qty_needed in list(remaining.items()):
            available = inv.get(sku_id, 0)
            if available <= 0:
                continue
            provide = min(available, qty_needed)
            if provide > 0:
                provided[sku_id] = provide
                new_qty = qty_needed - provide
                if new_qty <= 0:
                    to_delete.append(sku_id)
                else:
                    remaining[sku_id] = new_qty

        for sid in to_delete:
            del remaining[sid]

        if provided:
            allocation.append((wh_id, provided))

        if not remaining:
            return allocation

    return []


def _build_connected_route(env,
                           vehicle,
                           assigned_orders: List[Tuple[str, List[Tuple[str, Dict[str, int]]]]],
                           warehouses,
                           orders,
                           adjacency) -> Dict:
    """
    Construct steps:
    - Waypoints: home -> pickup warehouses -> order destinations -> home
    - Between waypoints: insert shortest path nodes via Dijkstra over adjacency_list
    - Operations only at actual warehouse/order nodes; travel steps are empty
    """

    # Resolve home node from vehicle home warehouse
    home_wh = warehouses[vehicle.home_warehouse_id]
    home_node = home_wh.location.id

    steps: List[Dict] = []

    def add_step(node_id: int):
        steps.append({'node_id': node_id, 'pickups': [], 'deliveries': [], 'unloads': []})

    def shortest_path(src: int, dst: int) -> Optional[List[int]]:
        if src == dst:
            return [src]
        heap = [(0.0, src)]
        dist = {src: 0.0}
        prev = {}
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d != dist.get(u, float('inf')):
                continue
            for v in adjacency.get(u, []):  # neighbors are node IDs
                w = env.get_distance(u, v)   # edge weight
                if w is None:
                    continue
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if dst not in dist:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    # Start at home
    current_node = home_node
    add_step(current_node)

    # Collect pickups grouped per warehouse
    pickup_by_wh = defaultdict(lambda: defaultdict(int))
    for order_id, allocation in assigned_orders:
        for wh_id, items in allocation:
            for sku_id, qty in items.items():
                pickup_by_wh[wh_id][sku_id] += qty

    # Visit warehouses for pickups
    for wh_id, sku_qty_map in pickup_by_wh.items():
        target_node = warehouses[wh_id].location.id
        if current_node != target_node:
            path = shortest_path(current_node, target_node)
            if not path:
                continue
            for node in path[1:]:
                add_step(node)
            current_node = target_node
        pickups = [{'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': qty}
                   for sku_id, qty in sku_qty_map.items()]
        steps[-1]['pickups'].extend(pickups)

    # Deliver to each order
    for order_id, allocation in assigned_orders:
        target_node = orders[order_id].destination.id
        if current_node != target_node:
            path = shortest_path(current_node, target_node)
            if not path:
                continue
            for node in path[1:]:
                add_step(node)
            current_node = target_node
        deliveries = []
        for wh_id, items in allocation:
            for sku_id, qty in items.items():
                deliveries.append({'order_id': order_id, 'sku_id': sku_id, 'quantity': qty})
        steps[-1]['deliveries'].extend(deliveries)

    # Return home
    if current_node != home_node:
        path = shortest_path(current_node, home_node)
        if path:
            for node in path[1:]:
                add_step(node)

    return {'vehicle_id': vehicle.id, 'steps': steps}
#from robin_logistics import LogisticsEnvironment
#env = LogisticsEnvironment()
#result = solver(env)
#print(f"Generated {len(result['routes'])} routes")