#!/usr/bin/env python3
"""
MWVRP Solver: 100% fulfillment first, minimal cost/distance, fewer vehicles
- Aggressive packing: fill existing vehicles before opening new ones
- Two-phase consolidation: close lightly loaded vehicles (cuts fixed costs)
- Distance-aware allocation (prefer 1-2 warehouses; expand only if required)
- Road-connected routing: Dijkstra with distance caching and connectivity guards
- Delivery sequencing: nearest-neighbor + cautious 2-opt (only for long routes)
- Order splitting across multiple vehicles when a single vehicle cannot fit
- Optional geographic clustering to reduce number of routes/vehicles when safe
- Auto-tuning picks the best valid solution with heavy penalty for under-fulfillment
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import heapq
import random


# ===================== ENTRY POINT =====================

def solver(env) -> Dict:
    """
    Auto-tunes conservative parameter sets that minimize vehicles while keeping fulfillment.
    """

    # Target small number of routes when feasible; always prioritize 100% fulfillment
    param_grid = [
        {"target_routes": 6, "capacity_buffer": 1.0, "max_warehouses": 2,
         "order_strategy": "nearest", "consolidate": True,
         "max_orders_per_vehicle": 24, "pack_threshold": 0.99},
        {"target_routes": 7, "capacity_buffer": 1.0, "max_warehouses": 2,
         "order_strategy": "largest", "consolidate": True,
         "max_orders_per_vehicle": 24, "pack_threshold": 0.99},
        # Non-clustered fallback
        {"target_routes": None, "capacity_buffer": 1.0, "max_warehouses": 2,
         "order_strategy": "nearest", "consolidate": True,
         "max_orders_per_vehicle": 25, "pack_threshold": 0.99},
    ]


    best_solution = None
    best_score = float("inf")

    for params in param_grid:
        env.reset_all_state()
        solution = base_solver(
            env,
            target_routes=params.get("target_routes"),
            capacity_buffer=params["capacity_buffer"],
            max_warehouses=params["max_warehouses"],
            order_strategy=params["order_strategy"],
            max_orders_per_vehicle=params["max_orders_per_vehicle"],
            pack_threshold=params["pack_threshold"],
            consolidate=params["consolidate"],
        )

        # Validate
        validation_result = env.validate_solution_complete(solution)
        if isinstance(validation_result, bool):
            if not validation_result:
                continue
        elif isinstance(validation_result, dict):
            if not validation_result.get("is_valid", True):
                continue

        # Metrics
        cost = env.calculate_solution_cost(solution)
        stats = env.get_solution_statistics(solution)
        requested = stats.get("total_items_requested", 0)
        delivered = stats.get("total_items_delivered", 0)
        fulfillment_rate = delivered / max(1, requested)

        # Score: fulfillment must be 100%, then minimize cost
        score = cost * (1 + max(0.0, 1.0 - fulfillment_rate) * 1000.0)

        if score < best_score:
            best_score = score
            best_solution = solution

    return best_solution if best_solution else {"routes": []}


# ===================== BASE SOLVER =====================

def base_solver(env,
                target_routes: Optional[int] = None,
                capacity_buffer: float = 1.00,
                max_warehouses: int = 2,
                order_strategy: str = "nearest",
                max_orders_per_vehicle: int = 18,
                pack_threshold: float = 0.95,
                consolidate: bool = True) -> Dict:
    """
    Vehicle-minimizing greedy solver with consolidation and optimized routing.
    """

    orders = env.orders
    warehouses = env.warehouses
    skus = env.skus

    # Vehicles sorted by economics (cheapest first)
    vehicles_list = env.get_all_vehicles()
    def _vehicle_total_cost_metric(v):
        fixed = getattr(v, "fixed_cost", 0.0)
        per_km = getattr(v, "cost_per_km", getattr(v, "variable_cost_per_km", 0.0))
        return fixed + per_km
    vehicles_list.sort(key=_vehicle_total_cost_metric)
    vehicles = {v.id: v for v in vehicles_list}

    # Road graph
    graph = env.get_road_network_data()
    adjacency = graph.get('adjacency_list', {})

    # Shadow inventory
    inventory = {wid: dict(wh.inventory) for wid, wh in warehouses.items()}

    # Track vehicle loads and assigned orders
    vehicle_loads = {v.id: {'weight': 0.0, 'volume': 0.0, 'orders': []} for v in vehicles_list}

    # Precompute reachability per vehicle to avoid assigning unreachable orders
    reachable_nodes_by_vehicle: Dict[str, Set[int]] = {}
    for v in vehicles_list:
        home = warehouses[v.home_warehouse_id].location.id
        reachable_nodes_by_vehicle[v.id] = _reachable_nodes(adjacency, home)

    # Build order sequence, optionally clustered to reduce number of routes
    clustered_order_lists: List[List[str]] = []
    all_order_ids = list(orders.keys())
    if not all_order_ids:
        return {"routes": []}

    if target_routes and isinstance(target_routes, int) and target_routes >= 2:
        clusters = kmeans_cluster_nodes({oid: orders[oid].destination.id for oid in all_order_ids},
                                        k=target_routes, env=env)
        def order_weight(oid: str) -> float:
            return sum(skus[s].weight * q for s, q in orders[oid].requested_items.items())

        for cluster in clusters:
            if order_strategy == "largest":
                clustered_order_lists.append(sorted(cluster, key=order_weight, reverse=True))
            elif order_strategy == "nearest":
                base_home_node = warehouses[vehicles_list[0].home_warehouse_id].location.id if vehicles_list else None
                clustered_order_lists.append(sorted(
                    cluster,
                    key=lambda oid: _safe_dist(env, base_home_node, orders[oid].destination.id) if base_home_node is not None else 0.0
                ))
            elif order_strategy == "random":
                c = list(cluster)
                random.shuffle(c)
                clustered_order_lists.append(c)
            else:
                clustered_order_lists.append(list(cluster))
    else:
        order_items = list(orders.items())
        if order_strategy == "largest":
            order_items.sort(
                key=lambda kv: sum(skus[s].weight * q for s, q in kv[1].requested_items.items()),
                reverse=True
            )
        elif order_strategy == "nearest":
            base_home_node = warehouses[vehicles_list[0].home_warehouse_id].location.id if vehicles_list else None
            order_items.sort(
                key=lambda kv: _safe_dist(env, base_home_node, kv[1].destination.id) if base_home_node is not None else 0.0
            )
        elif order_strategy == "random":
            random.shuffle(order_items)
        clustered_order_lists = [[oid for oid, _ in order_items]]

    # Assign: pack into existing vehicles before opening new ones, with safe splitting fallback
    for cluster_orders in clustered_order_lists:
        for order_id in cluster_orders:
            order = orders[order_id]
            order_weight = sum(skus[s].weight * q for s, q in order.requested_items.items())
            order_volume = sum(skus[s].volume * q for s, q in order.requested_items.items())

            # Allocation: prefer closest limited to few warehouses
            allocation = _find_allocation_prefer_close(order, inventory, warehouses, skus, env, max_warehouses)
            if not allocation:
                allocation = _find_simple_allocation(order, inventory, warehouses, skus, max_warehouses)
            # Last resort: allow more warehouses if absolutely necessary to fulfill
            if not allocation:
                allocation = _find_allocation_any(order, inventory, warehouses, env)
            if not allocation:
                # Cannot fulfill due to inventory shortage
                continue

            # Utilization helper
            def utilization(v_id):
                l = vehicle_loads[v_id]; v = vehicles[v_id]
                w_util = l['weight'] / max(1e-9, v.capacity_weight * capacity_buffer) if v.capacity_weight > 0 else 1.0
                v_util = l['volume'] / max(1e-9, v.capacity_volume * capacity_buffer) if v.capacity_volume > 0 else 1.0
                return max(w_util, v_util)

            # Vehicles already open, most utilized first
            open_vehicle_ids = [vid for vid in vehicle_loads if vehicle_loads[vid]['orders']]
            open_vehicle_ids.sort(key=lambda vid: utilization(vid), reverse=True)

            assigned_vehicle = None

            # Pass 1: pack full order into open vehicles (connectivity-aware)
            for vid in open_vehicle_ids:
                l = vehicle_loads[vid]
                v = vehicles[vid]
                # Respect order cap only if this order isn't already on the vehicle
                has_this_order = any(oid == order_id for oid, _ in l['orders'])
                if not has_this_order and len(l['orders']) >= max_orders_per_vehicle:
                    continue
                # Connectivity: vehicle must reach order destination
                if orders[order_id].destination.id not in reachable_nodes_by_vehicle[vid]:
                    continue
                if (l['weight'] + order_weight <= v.capacity_weight * capacity_buffer and
                    l['volume'] + order_volume <= v.capacity_volume * capacity_buffer):
                    assigned_vehicle = vid
                    break

            # Pass 2: open the cheapest feasible vehicle (connectivity-aware)
            if assigned_vehicle is None:
                for v in vehicles_list:
                    l = vehicle_loads[v.id]
                    # Vehicle must reach order destination
                    if orders[order_id].destination.id not in reachable_nodes_by_vehicle[v.id]:
                        continue
                    if len(l['orders']) >= max_orders_per_vehicle:
                        continue
                    if (l['weight'] + order_weight <= v.capacity_weight * capacity_buffer and
                        l['volume'] + order_volume <= v.capacity_volume * capacity_buffer):
                        assigned_vehicle = v.id
                        break

            if assigned_vehicle is not None:
                # Ensure all pickup warehouses are reachable by this vehicle; otherwise, fallback to split path
                all_reachable = True
                for wh_id, _ in allocation:
                    wh_node = warehouses[wh_id].location.id
                    if wh_node not in reachable_nodes_by_vehicle[assigned_vehicle]:
                        all_reachable = False
                        break
                if not all_reachable:
                    assigned_vehicle = None

            if assigned_vehicle is not None:
                # Accept full order on one vehicle
                vehicle_loads[assigned_vehicle]['weight'] += order_weight
                vehicle_loads[assigned_vehicle]['volume'] += order_volume
                vehicle_loads[assigned_vehicle]['orders'].append((order_id, allocation))
                # Update inventory for full allocation
                for wh_id, items in allocation:
                    for sku_id, qty in items.items():
                        inventory[wh_id][sku_id] -= qty
                continue

            # Pass 3: split across multiple vehicles safely (merge per-vehicle entry)
            remaining_items = dict(order.requested_items)
            success_split = _assign_order_split_across_vehicles(
                order_id=order_id,
                remaining_items=remaining_items,
                inventory=inventory,
                warehouses=warehouses,
                skus=skus,
                env=env,
                vehicles_list=vehicles_list,
                vehicle_loads=vehicle_loads,
                capacity_buffer=capacity_buffer,
                max_orders_per_vehicle=max_orders_per_vehicle,
                max_warehouses=max_warehouses,
                reachable_nodes_by_vehicle=reachable_nodes_by_vehicle,
                dest_node=orders[order_id].destination.id,
            )
            # If splitting failed, leave unassigned (will be penalized and not selected by auto-tuner)

    # Consolidate to reduce vehicles/routes
    if consolidate:
        vehicle_loads = consolidate_vehicles(vehicles_list, vehicle_loads, orders, skus, capacity_buffer)
        vehicle_loads = consolidate_small_fleets(vehicles_list, vehicle_loads, orders, skus,
                                                 capacity_buffer, max_orders_threshold=3)

    # Build connected routes
    solution = {"routes": []}
    for v in vehicles_list:
        info = vehicle_loads[v.id]
        if not info['orders']:
            continue
        route = _build_connected_route(env, v, info['orders'], warehouses, orders, adjacency)
        if route and route.get('steps'):
            solution["routes"].append(route)

    return solution


# ===================== HELPERS =====================

def _safe_dist(env, u: Optional[int], v: Optional[int]) -> float:
    if u is None or v is None:
        return float('inf')
    w = env.get_distance(u, v)
    return w if (w is not None) else float('inf')


def consolidate_vehicles(vehicles_list, vehicle_loads, orders, skus, capacity_buffer=1.0):
    """
    Move smallest orders from lightly loaded vehicles into other vehicles (if feasible)
    to reduce the number of active vehicles and cut fixed costs.
    """
    def util(v):
        l = vehicle_loads[v.id]
        w_util = l['weight'] / max(1e-9, v.capacity_weight * capacity_buffer) if v.capacity_weight > 0 else 1.0
        v_util = l['volume'] / max(1e-9, v.capacity_volume * capacity_buffer) if v.capacity_volume > 0 else 1.0
        return max(w_util, v_util)

    sorted_by_util = sorted(vehicles_list, key=util)  # donors first

    for donor in sorted_by_util:
        donor_load = vehicle_loads[donor.id]
        if not donor_load['orders']:
            continue

        donor_orders_sorted = sorted(
            donor_load['orders'],
            key=lambda ov: sum(skus[s].weight * q for s, q in orders[ov[0]].requested_items.items())
        )

        for (oid, alloc) in donor_orders_sorted:
            o = orders[oid]
            o_weight = sum(skus[s].weight * q for s, q in o.requested_items.items())
            o_volume = sum(skus[s].volume * q for s, q in o.requested_items.items())

            # Move into recipient if feasible
            for recipient in vehicles_list:
                if recipient.id == donor.id:
                    continue
                r_load = vehicle_loads[recipient.id]
                if (r_load['weight'] + o_weight <= recipient.capacity_weight * capacity_buffer and
                    r_load['volume'] + o_volume <= recipient.capacity_volume * capacity_buffer):
                    r_load['orders'].append((oid, alloc))
                    r_load['weight'] += o_weight
                    r_load['volume'] += o_volume
                    donor_load['orders'] = [x for x in donor_load['orders'] if x[0] != oid]
                    donor_load['weight'] -= o_weight
                    donor_load['volume'] -= o_volume
                    break  # next order

    return vehicle_loads


def consolidate_small_fleets(vehicles_list, vehicle_loads, orders, skus,
                             capacity_buffer=1.0, max_orders_threshold=3):
    """
    Close vehicles with few orders (<= threshold) by moving their orders to other vehicles.
    """
    donors = [v for v in vehicles_list if len(vehicle_loads[v.id]['orders']) <= max_orders_threshold
              and vehicle_loads[v.id]['orders']]

    for donor in donors:
        donor_load = vehicle_loads[donor.id]
        donor_orders = list(donor_load['orders'])  # copy

        for (oid, alloc) in donor_orders:
            o = orders[oid]
            o_weight = sum(skus[s].weight * q for s, q in o.requested_items.items())
            o_volume = sum(skus[s].volume * q for s, q in o.requested_items.items())

            for recipient in vehicles_list:
                if recipient.id == donor.id:
                    continue
                r_load = vehicle_loads[recipient.id]
                if (r_load['weight'] + o_weight <= recipient.capacity_weight * capacity_buffer and
                    r_load['volume'] + o_volume <= recipient.capacity_volume * capacity_buffer):
                    # move order
                    r_load['orders'].append((oid, alloc))
                    r_load['weight'] += o_weight
                    r_load['volume'] += o_volume
                    donor_load['orders'] = [x for x in donor_load['orders'] if x[0] != oid]
                    donor_load['weight'] -= o_weight
                    donor_load['volume'] -= o_volume
                    break  # next order

    return vehicle_loads


def _find_simple_allocation(order, inventory, warehouses, skus, max_warehouses=2):
    """
    Try single warehouse, else split across up to max_warehouses.
    """
    needed = dict(order.requested_items)

    # Single warehouse
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
            take = min(available, qty_needed)
            if take > 0:
                provided[sku_id] = take
                new_qty = qty_needed - take
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

    return allocation if not remaining else []


def _find_allocation_prefer_close(order, inventory, warehouses, skus, env, max_warehouses=2):
    """
    Prefer nearest single warehouse; else split across nearest warehouses (capped).
    """
    needed = dict(order.requested_items)
    customer_node = order.destination.id

    # Single warehouse candidates (closest first)
    single_wh = []
    for wh_id, inv in inventory.items():
        if all(inv.get(sid, 0) >= qty for sid, qty in needed.items()):
            wh_node = warehouses[wh_id].location.id
            dist = _safe_dist(env, wh_node, customer_node)
            single_wh.append((dist, wh_id))
    if single_wh:
        single_wh.sort()
        return [(single_wh[0][1], needed)]

    # Split across nearest warehouses (up to max_warehouses)
    wh_order = []
    for wh_id in inventory.keys():
        wh_node = warehouses[wh_id].location.id
        dist = _safe_dist(env, wh_node, customer_node)
        wh_order.append((dist, wh_id))
    wh_order.sort()

    allocation, remaining = [], dict(needed)
    for _, wh_id in wh_order:
        if len(allocation) >= max_warehouses:
            break
        inv = inventory[wh_id]
        provided, to_delete = {}, []
        for sku_id, qty_needed in list(remaining.items()):
            available = inv.get(sku_id, 0)
            if available <= 0:
                continue
            take = min(available, qty_needed)
            if take > 0:
                provided[sku_id] = take
                new_qty = qty_needed - take
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

    return allocation if not remaining else []


def _find_allocation_any(order, inventory, warehouses, env):
    """
    Last-resort allocator: split across any number of warehouses (closest-first) to fulfill order.
    """
    needed = dict(order.requested_items)
    customer_node = order.destination.id

    # Warehouses by proximity
    wh_order = []
    for wh_id, inv in inventory.items():
        wh_node = warehouses[wh_id].location.id
        dist = _safe_dist(env, wh_node, customer_node)
        wh_order.append((dist, wh_id))
    wh_order.sort()

    allocation, remaining = [], dict(needed)
    for _, wh_id in wh_order:
        inv = inventory[wh_id]
        provided = {}
        to_delete = []
        for sku_id, qty_needed in list(remaining.items()):
            available = inv.get(sku_id, 0)
            if available <= 0:
                continue
            take = min(available, qty_needed)
            if take > 0:
                provided[sku_id] = take
                new_qty = qty_needed - take
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

    return allocation if not remaining else []


def _reachable_nodes(adjacency: Dict[int, List[int]], start: int) -> Set[int]:
    visited: Set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        for v in adjacency.get(u, []):
            if v not in visited:
                stack.append(v)
    return visited


def _assign_order_split_across_vehicles(
    order_id: str,
    remaining_items: Dict[str, int],
    inventory: Dict[str, Dict[str, int]],
    warehouses,
    skus,
    env,
    vehicles_list,
    vehicle_loads,
    capacity_buffer: float,
    max_orders_per_vehicle: int,
    max_warehouses: int,
    reachable_nodes_by_vehicle: Dict[str, Set[int]],
    dest_node: int,
) -> bool:
    """
    Split an oversized order across multiple vehicles, merging per-vehicle entries to avoid duplicates.
    Returns True if the entire order was assigned, False otherwise.
    """

    def vehicle_can_take_any(vid: str) -> bool:
        load = vehicle_loads[vid]
        # Allow adding if this vehicle already has this order, else respect cap
        has_this_order = any(oid == order_id for oid, _ in load['orders'])
        return has_this_order or (len(load['orders']) < max_orders_per_vehicle)

    # First try open vehicles, then remaining vehicles
    candidate_vids = [v.id for v in vehicles_list if vehicle_loads[v.id]['orders']]
    candidate_vids += [v.id for v in vehicles_list if not vehicle_loads[v.id]['orders']]

    while remaining_items:
        progress = False
        for vid in candidate_vids:
            v = next(vv for vv in vehicles_list if vv.id == vid)
            # Connectivity: order destination must be reachable by this vehicle
            if dest_node not in reachable_nodes_by_vehicle[vid]:
                continue
            # Respect order cap
            if not vehicle_can_take_any(vid):
                continue
            # Build a partial chunk limited by vehicle remaining capacity and max_warehouses
            chunk = _allocate_partial_for_vehicle(
                order_id=order_id,
                remaining_items=remaining_items,
                inventory=inventory,
                warehouses=warehouses,
                skus=skus,
                env=env,
                vehicle=v,
                vehicle_load=vehicle_loads[vid],
                capacity_buffer=capacity_buffer,
                max_warehouses=max_warehouses,
                reachable_nodes=reachable_nodes_by_vehicle[vid],
                dest_node=dest_node,
            )
            if not chunk:
                continue

            # Merge chunk into vehicle's orders list (ensure single entry per order_id)
            load = vehicle_loads[vid]
            found_index = None
            for idx, (oid, _) in enumerate(load['orders']):
                if oid == order_id:
                    found_index = idx
                    break
            if found_index is None:
                load['orders'].append((order_id, chunk['allocation']))
            else:
                # Merge into existing allocation
                existing_alloc = load['orders'][found_index][1]
                merged = _merge_allocations(existing_alloc, chunk['allocation'])
                load['orders'][found_index] = (order_id, merged)

            # Update load totals
            load['weight'] += chunk['weight']
            load['volume'] += chunk['volume']

            # Update inventory and remaining items
            for wh_id, items in chunk['allocation']:
                for sku_id, qty in items.items():
                    inventory[wh_id][sku_id] -= qty
                    remaining_qty = remaining_items.get(sku_id, 0)
                    if remaining_qty > 0:
                        new_qty = remaining_qty - qty
                        if new_qty > 0:
                            remaining_items[sku_id] = new_qty
                        else:
                            del remaining_items[sku_id]

            progress = True
            if not remaining_items:
                break

        if not progress:
            break

    return len(remaining_items) == 0


def _allocate_partial_for_vehicle(
    order_id: str,
    remaining_items: Dict[str, int],
    inventory: Dict[str, Dict[str, int]],
    warehouses,
    skus,
    env,
    vehicle,
    vehicle_load,
    capacity_buffer: float,
    max_warehouses: int,
    reachable_nodes: Set[int],
    dest_node: int,
) -> Optional[Dict]:
    """
    Build a per-vehicle allocation chunk limited by remaining capacity and warehouses.
    Returns dict with keys: 'allocation' (list of (wh_id, items)), 'weight', 'volume'.
    Warehouses and order destination must be reachable from vehicle home.
    """
    # Remaining capacity
    rem_w = float('inf') if vehicle.capacity_weight <= 0 else vehicle.capacity_weight * capacity_buffer - vehicle_load['weight']
    rem_v = float('inf') if vehicle.capacity_volume <= 0 else vehicle.capacity_volume * capacity_buffer - vehicle_load['volume']
    if rem_w <= 1e-9 and rem_v <= 1e-9:
        return None

    # Build warehouse order by proximity to this order's destination, filtered by reachability
    wh_order = []
    for wh_id, inv in inventory.items():
        wh_node = warehouses[wh_id].location.id
        if wh_node not in reachable_nodes:
            continue
        d = _safe_dist(env, wh_node, dest_node)
        wh_order.append((d, wh_id))
    # If none reachable, cannot allocate
    if not wh_order:
        return None
    wh_order.sort()

    allocation: List[Tuple[str, Dict[str, int]]] = []
    used_wh = 0
    weight_sum = 0.0
    volume_sum = 0.0

    # Iterate warehouses by preference
    for _, wh_id in wh_order:
        if used_wh >= max_warehouses:
            break
        inv = inventory[wh_id]
        provided: Dict[str, int] = {}
        # Greedily allocate SKUs limited by remaining capacity
        for sku_id, qty_needed in list(remaining_items.items()):
            if qty_needed <= 0:
                continue
            available = inv.get(sku_id, 0)
            if available <= 0:
                continue
            w_unit = skus[sku_id].weight if skus[sku_id].weight > 0 else 0.0
            v_unit = skus[sku_id].volume if skus[sku_id].volume > 0 else 0.0

            # Compute capacity-limited max quantity
            max_by_weight = qty_needed
            if w_unit > 0 and rem_w < float('inf'):
                max_by_weight = min(max_by_weight, int(rem_w // w_unit) if w_unit > 0 else qty_needed)
            max_by_volume = qty_needed
            if v_unit > 0 and rem_v < float('inf'):
                max_by_volume = min(max_by_volume, int(rem_v // v_unit) if v_unit > 0 else qty_needed)

            can_take = min(qty_needed, available, max_by_weight, max_by_volume)
            if can_take <= 0:
                continue

            provided[sku_id] = can_take
            take_w = w_unit * can_take
            take_v = v_unit * can_take
            if rem_w < float('inf'):
                rem_w -= take_w
            if rem_v < float('inf'):
                rem_v -= take_v
            weight_sum += take_w
            volume_sum += take_v

            # If capacity nearly exhausted, stop
            if (rem_w <= 1e-6 and rem_v <= 1e-6):
                break

        if provided:
            allocation.append((wh_id, provided))
            used_wh += 1

        # If capacity is exhausted, stop
        if (rem_w <= 1e-6 and rem_v <= 1e-6):
            break

    if not allocation:
        return None

    return {"allocation": allocation, "weight": weight_sum, "volume": volume_sum}


def _merge_allocations(base_alloc: List[Tuple[str, Dict[str, int]]],
                       add_alloc: List[Tuple[str, Dict[str, int]]]) -> List[Tuple[str, Dict[str, int]]]:
    merged: Dict[str, Dict[str, int]] = {}
    for wh_id, items in base_alloc:
        merged.setdefault(wh_id, {})
        for sku_id, qty in items.items():
            merged[wh_id][sku_id] = merged[wh_id].get(sku_id, 0) + qty
    for wh_id, items in add_alloc:
        merged.setdefault(wh_id, {})
        for sku_id, qty in items.items():
            merged[wh_id][sku_id] = merged[wh_id].get(sku_id, 0) + qty
    return [(wh_id, items) for wh_id, items in merged.items()]


def kmeans_cluster_nodes(order_nodes: Dict[str, int], k: int, env) -> List[List[str]]:
    """
    Simple k-means/medoids on order destination nodes using pairwise distances (Lloyd's style).
    Returns up to k clusters of order IDs. If k >= number of orders, returns singleton clusters.
    """
    oids = list(order_nodes.keys())
    n = len(oids)
    if k is None or k <= 1 or k >= n:
        return [[oid] for oid in oids]

    # Initialize centers using evenly spaced samples
    step = max(1, n // k)
    centers = [order_nodes[oid] for oid in oids[::step][:k]]

    def dist(a_node: int, b_node: int) -> float:
        d = env.get_distance(a_node, b_node)
        return d if d is not None else float('inf')

    for _ in range(8):  # few iterations for speed
        clusters = [[] for _ in range(k)]
        # Assign
        for oid in oids:
            onode = order_nodes[oid]
            best_i = 0
            best_d = dist(onode, centers[0])
            for i in range(1, k):
                di = dist(onode, centers[i])
                if di < best_d:
                    best_i, best_d = i, di
            clusters[best_i].append(oid)

        # Recompute centers as medoids
        new_centers: List[int] = []
        for i in range(k):
            cluster = clusters[i]
            if not cluster:
                new_centers.append(centers[i])
                continue
            nodes = [order_nodes[oid] for oid in cluster]
            best_node, best_sum = nodes[0], float('inf')
            for nnode in nodes:
                s = 0.0
                for mnode in nodes:
                    s += dist(nnode, mnode)
                if s < best_sum:
                    best_sum = s
                    best_node = nnode
            new_centers.append(best_node)

        if all(new_centers[i] == centers[i] for i in range(k)):
            centers = new_centers
            break
        centers = new_centers

    # Final assignment
    clusters = [[] for _ in range(k)]
    for oid in oids:
        onode = order_nodes[oid]
        best_i = 0
        best_d = dist(onode, centers[0])
        for i in range(1, k):
            di = dist(onode, centers[i])
            if di < best_d:
                best_i, best_d = i, di
        clusters[best_i].append(oid)

    # Remove empties
    return [c for c in clusters if c]


def _build_connected_route(env,
                           vehicle,
                           assigned_orders: List[Tuple[str, List[Tuple[str, Dict[str, int]]]]],
                           warehouses,
                           orders,
                           adjacency) -> Dict:
    """
    Build route with safe connectivity and optimized delivery order.
    """

    home_node = warehouses[vehicle.home_warehouse_id].location.id
    steps: List[Dict] = []

    def add_step(node_id: int):
        steps.append({'node_id': node_id, 'pickups': [], 'deliveries': [], 'unloads': []})

    # Distance cache for Dijkstra
    dist_cache = {}
    def _edge_weight(u, v):
        key = (u, v)
        if key in dist_cache:
            return dist_cache[key]
        w = env.get_distance(u, v)
        dist_cache[key] = w
        return w

    def shortest_path(src: int, dst: int) -> Optional[List[int]]:
        if src == dst:
            return [src]
        heap = [(0.0, src)]
        dist = {src: 0.0}
        prev = {}
        visited = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            if u == dst:
                break

            for v in adjacency.get(u, []):
                w = _edge_weight(u, v)
                if w is None:
                    continue
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if dst not in dist:
            return None

        # Reconstruct path
        path = [dst]
        while path[-1] != src:
            if path[-1] not in prev:
                return None
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

    # Visit warehouses (skip unreachable legs)
    warehouse_list = list(pickup_by_wh.keys())
    picked_anything = False
    for wh_id in warehouse_list:
        target_node = warehouses[wh_id].location.id
        if current_node != target_node:
            path = shortest_path(current_node, target_node)
            if not path:
                continue
            for node in path[1:]:
                add_step(node)
            current_node = target_node

        pickups = [
            {'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': qty}
            for sku_id, qty in pickup_by_wh[wh_id].items()
        ]
        if pickups:
            steps[-1]['pickups'].extend(pickups)
            picked_anything = True

    # If no pickups, go home
    if not picked_anything:
        if current_node != home_node:
            path = shortest_path(current_node, home_node)
            if path:
                for node in path[1:]:
                    add_step(node)
        return {'vehicle_id': vehicle.id, 'steps': steps}

    # Optimize deliveries: NN + cautious 2-opt
    order_ids_optimized = _optimize_delivery_order_safe(assigned_orders, orders, warehouse_list, warehouses, home_node, env)

    # Deliveries
    alloc_map = {oid: alloc for oid, alloc in assigned_orders}
    assigned_orders_opt = [(oid, alloc_map.get(oid, [])) for oid in order_ids_optimized]

    for order_id, allocation in assigned_orders_opt:
        if not allocation:
            continue
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
        if deliveries:
            steps[-1]['deliveries'].extend(deliveries)

    # Return home
    if current_node != home_node:
        path = shortest_path(current_node, home_node)
        if path:
            for node in path[1:]:
                add_step(node)

    if not steps or steps[0]['node_id'] != home_node:
        return None

    return {'vehicle_id': vehicle.id, 'steps': steps}


def _optimize_delivery_order_safe(assigned_orders, orders, warehouse_list, warehouses, home_node, env):
    """
    Nearest-neighbor + single 2-opt improvement (only for routes >= 6 deliveries).
    """
    if len(assigned_orders) <= 1:
        return [oid for oid, _ in assigned_orders]

    start_node = warehouses[warehouse_list[-1]].location.id if warehouse_list else home_node

    # Pair distance cache
    pair_cache = {}
    def _pair_dist(a, b):
        key = (a, b)
        if key in pair_cache:
            return pair_cache[key]
        d = env.get_distance(a, b)
        pair_cache[key] = d if d is not None else float('inf')
        return pair_cache[key]

    # Nearest neighbor
    remaining = {oid for oid, _ in assigned_orders}
    current = start_node
    ordered = []
    while remaining:
        best_order = None
        best_dist = float('inf')
        for oid in remaining:
            order_node = orders[oid].destination.id
            d = _pair_dist(current, order_node)
            if d < best_dist:
                best_dist = d
                best_order = oid
        if best_order is None:
            ordered.extend([oid for oid, _ in assigned_orders if oid in remaining])
            break
        ordered.append(best_order)
        remaining.remove(best_order)
        current = orders[best_order].destination.id

    # Single 2-opt improvement (routes >= 6)
    if len(ordered) < 6:
        return ordered

    def route_distance(seq, start):
        dist = 0.0
        cur = start
        for oid in seq:
            dst = orders[oid].destination.id
            dist += _pair_dist(cur, dst)
            cur = dst
        return dist

    base = route_distance(ordered, start_node)
    improved_once = False
    for i in range(1, len(ordered) - 2):
        for j in range(i + 1, len(ordered)):
            candidate = ordered[:i] + ordered[i:j][::-1] + ordered[j:]
            cd = route_distance(candidate, start_node)
            if cd + 1e-6 < base:
                ordered = candidate
                base = cd
                improved_once = True
                break
        if improved_once:
            break

    return ordered
#from robin_logistics import LogisticsEnvironment
#env = LogisticsEnvironment()
#result = solver(env)
#print(f"Generated {len(result['routes'])} routes")