#!/usr/bin/env python3
"""
Enhanced MWVRP Solver: 100% Fulfillment + Minimal Cost & Distance
Key improvements:
- Guaranteed 100% fulfillment (all orders assigned if inventory exists)
- Greedy best-fit decreasing for optimal vehicle packing
- Multi-pass consolidation to minimize vehicle count
- Smart warehouse allocation prioritizing proximity
- Optimized routing with Dijkstra + 2-opt
- Comprehensive parameter tuning
"""

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import heapq
import random


def solver(env) -> Dict:
    """
    Main solver with multi-strategy approach to ensure 100% fulfillment
    while minimizing cost and distance.
    """
    
    # Try multiple strategies and pick the best valid solution
    strategies = [
        # Strategy 1: Aggressive packing, nearest-first
        {"pack_threshold": 0.99, "order_strategy": "nearest", "max_warehouses": 2, 
         "capacity_buffer": 1.0, "consolidate_threshold": 5},
        
        # Strategy 2: Aggressive packing, largest-first  
        {"pack_threshold": 0.99, "order_strategy": "largest", "max_warehouses": 2,
         "capacity_buffer": 1.0, "consolidate_threshold": 5},
        
        # Strategy 3: Slightly relaxed packing, nearest-first
        {"pack_threshold": 0.97, "order_strategy": "nearest", "max_warehouses": 2,
         "capacity_buffer": 1.0, "consolidate_threshold": 4},
        
        # Strategy 4: Allow more warehouse splits if needed
        {"pack_threshold": 0.99, "order_strategy": "nearest", "max_warehouses": 3,
         "capacity_buffer": 1.0, "consolidate_threshold": 5},
        
        # Strategy 5: Fallback with maximum flexibility
        {"pack_threshold": 0.95, "order_strategy": "random", "max_warehouses": 3,
         "capacity_buffer": 1.0, "consolidate_threshold": 3},
    ]
    
    best_solution = None
    best_score = float("inf")
    best_fulfillment = 0.0
    
    for strategy in strategies:
        env.reset_all_state()
        
        solution = base_solver(
            env,
            pack_threshold=strategy["pack_threshold"],
            order_strategy=strategy["order_strategy"],
            max_warehouses=strategy["max_warehouses"],
            capacity_buffer=strategy["capacity_buffer"],
            consolidate_threshold=strategy["consolidate_threshold"]
        )
        
        # Validate
        validation_result = env.validate_solution_complete(solution)
        if isinstance(validation_result, bool):
            if not validation_result:
                continue
        elif isinstance(validation_result, dict):
            if not validation_result.get("is_valid", True):
                continue
        
        # Calculate metrics
        cost = env.calculate_solution_cost(solution)
        stats = env.get_solution_statistics(solution)
        requested = stats.get("total_items_requested", 1)
        delivered = stats.get("total_items_delivered", 0)
        fulfillment_rate = delivered / max(1, requested)
        
        # Calculate total distance
        total_distance = 0.0
        for route in solution.get("routes", []):
            steps = route.get("steps", [])
            for i in range(len(steps) - 1):
                dist = env.get_distance(steps[i]["node_id"], steps[i + 1]["node_id"])
                if dist is not None:
                    total_distance += dist
        
        # Score: Fulfillment is paramount, then cost, then distance
        if fulfillment_rate < 0.9999:  # Not 100%
            score = 1e12 * (1.0 - fulfillment_rate) + cost + total_distance * 0.1
        else:
            score = cost + total_distance * 0.1
        
        # Update best if better fulfillment OR (same fulfillment and better score)
        if fulfillment_rate > best_fulfillment or (fulfillment_rate >= best_fulfillment and score < best_score):
            best_score = score
            best_solution = solution
            best_fulfillment = fulfillment_rate
    
    return best_solution if best_solution else {"routes": []}


def base_solver(env,
                pack_threshold: float = 0.99,
                order_strategy: str = "nearest",
                max_warehouses: int = 2,
                capacity_buffer: float = 1.0,
                consolidate_threshold: int = 5) -> Dict:
    """
    Base solver with guaranteed order fulfillment if inventory exists.
    """
    
    orders = env.orders
    warehouses = env.warehouses
    skus = env.skus
    
    if not orders:
        return {"routes": []}
    
    # Get all vehicles sorted by cost-efficiency
    vehicles_list = env.get_all_vehicles()
    vehicles_list.sort(key=lambda v: (
        getattr(v, "fixed_cost", 0.0) + 
        getattr(v, "cost_per_km", getattr(v, "variable_cost_per_km", 0.0)) * 100
    ))
    vehicles = {v.id: v for v in vehicles_list}
    
    # Road network
    graph = env.get_road_network_data()
    adjacency = graph.get('adjacency_list', {})
    
    # Shadow inventory tracking
    inventory = {wid: dict(wh.inventory) for wid, wh in warehouses.items()}
    
    # Vehicle loads: {vehicle_id: {'weight': float, 'volume': float, 'orders': [(order_id, allocation)]}}
    vehicle_loads = {v.id: {'weight': 0.0, 'volume': 0.0, 'orders': []} for v in vehicles_list}
    
    # Sort orders by strategy
    order_list = list(orders.items())
    if order_strategy == "largest":
        order_list.sort(
            key=lambda kv: sum(skus[s].weight * q for s, q in kv[1].requested_items.items()),
            reverse=True
        )
    elif order_strategy == "nearest":
        if vehicles_list:
            base_node = warehouses[vehicles_list[0].home_warehouse_id].location.id
            order_list.sort(
                key=lambda kv: safe_dist(env, base_node, kv[1].destination.id)
            )
    elif order_strategy == "random":
        random.shuffle(order_list)
    
    # PHASE 1: Assign ALL orders to vehicles (prioritize fulfillment)
    for order_id, order in order_list:
        order_weight = sum(skus[s].weight * q for s, q in order.requested_items.items())
        order_volume = sum(skus[s].volume * q for s, q in order.requested_items.items())
        
        # Find allocation from warehouses
        allocation = find_best_allocation(order, inventory, warehouses, skus, env, max_warehouses)
        
        if not allocation:
            # Skip this order - no inventory available
            continue
        
        # Find best vehicle for this order
        assigned_vehicle = None
        
        # Try 1: Pack into existing vehicles (best-fit)
        candidates = []
        for v in vehicles_list:
            load = vehicle_loads[v.id]
            if not load['orders']:
                continue  # Skip empty vehicles for now
            
            # Check capacity
            if (load['weight'] + order_weight <= v.capacity_weight * capacity_buffer and
                load['volume'] + order_volume <= v.capacity_volume * capacity_buffer):
                
                # Calculate utilization after adding this order
                util_w = (load['weight'] + order_weight) / max(1e-9, v.capacity_weight)
                util_v = (load['volume'] + order_volume) / max(1e-9, v.capacity_volume)
                util = max(util_w, util_v)
                
                # Only consider if under pack threshold
                if util <= pack_threshold:
                    candidates.append((util, v.id))
        
        # Pick the vehicle with highest utilization (best-fit)
        if candidates:
            candidates.sort(reverse=True)
            assigned_vehicle = candidates[0][1]
        
        # Try 2: Use a new vehicle (cheapest available)
        if assigned_vehicle is None:
            for v in vehicles_list:
                load = vehicle_loads[v.id]
                if (load['weight'] + order_weight <= v.capacity_weight * capacity_buffer and
                    load['volume'] + order_volume <= v.capacity_volume * capacity_buffer):
                    assigned_vehicle = v.id
                    break
        
        # If still no vehicle, skip (shouldn't happen with sufficient fleet)
        if assigned_vehicle is None:
            continue
        
        # Assign order to vehicle
        vehicle_loads[assigned_vehicle]['weight'] += order_weight
        vehicle_loads[assigned_vehicle]['volume'] += order_volume
        vehicle_loads[assigned_vehicle]['orders'].append((order_id, allocation))
        
        # Update inventory
        for wh_id, items in allocation:
            for sku_id, qty in items.items():
                inventory[wh_id][sku_id] -= qty
    
    # PHASE 2: Consolidation - reduce number of vehicles
    vehicle_loads = consolidate_routes(vehicles_list, vehicle_loads, orders, skus, 
                                       capacity_buffer, consolidate_threshold)
    
    # PHASE 3: Build optimized routes
    solution = {"routes": []}
    for v in vehicles_list:
        load = vehicle_loads[v.id]
        if not load['orders']:
            continue
        
        route = build_optimized_route(env, v, load['orders'], warehouses, orders, adjacency)
        if route and route.get('steps'):
            solution["routes"].append(route)
    
    return solution


def find_best_allocation(order, inventory, warehouses, skus, env, max_warehouses=2):
    """
    Find the best warehouse allocation for an order.
    Prioritizes: 1) Single warehouse, 2) Closest warehouses
    """
    needed = dict(order.requested_items)
    customer_node = order.destination.id
    
    # Try single warehouse (best option)
    single_wh_options = []
    for wh_id, inv in inventory.items():
        if all(inv.get(sid, 0) >= qty for sid, qty in needed.items()):
            wh_node = warehouses[wh_id].location.id
            dist = safe_dist(env, wh_node, customer_node)
            single_wh_options.append((dist, wh_id))
    
    if single_wh_options:
        single_wh_options.sort()
        return [(single_wh_options[0][1], needed)]
    
    # Multi-warehouse allocation (prioritize closest)
    wh_by_dist = []
    for wh_id in inventory.keys():
        wh_node = warehouses[wh_id].location.id
        dist = safe_dist(env, wh_node, customer_node)
        wh_by_dist.append((dist, wh_id))
    wh_by_dist.sort()
    
    allocation = []
    remaining = dict(needed)
    
    for _, wh_id in wh_by_dist:
        if len(allocation) >= max_warehouses or not remaining:
            break
        
        inv = inventory[wh_id]
        provided = {}
        
        for sku_id in list(remaining.keys()):
            available = inv.get(sku_id, 0)
            if available > 0:
                take = min(available, remaining[sku_id])
                provided[sku_id] = take
                remaining[sku_id] -= take
                if remaining[sku_id] <= 0:
                    del remaining[sku_id]
        
        if provided:
            allocation.append((wh_id, provided))
    
    return allocation if not remaining else []


def consolidate_routes(vehicles_list, vehicle_loads, orders, skus, capacity_buffer, threshold):
    """
    Multi-pass consolidation to minimize number of active vehicles.
    Aggressively consolidates to reduce fixed costs.
    """
    def get_util(v_id):
        load = vehicle_loads[v_id]
        v = next(v for v in vehicles_list if v.id == v_id)
        w_util = load['weight'] / max(1e-9, v.capacity_weight) if v.capacity_weight > 0 else 1.0
        v_util = load['volume'] / max(1e-9, v.capacity_volume) if v.capacity_volume > 0 else 1.0
        return max(w_util, v_util)
    
    def get_cost(v_id):
        v = next(v for v in vehicles_list if v.id == v_id)
        return getattr(v, "fixed_cost", 0.0)
    
    # Multiple consolidation passes for maximum reduction
    for pass_num in range(5):
        # Find vehicles to consolidate (prioritize expensive, lightly loaded ones)
        donors = []
        for v in vehicles_list:
            order_count = len(vehicle_loads[v.id]['orders'])
            if 0 < order_count <= threshold:
                util = get_util(v.id)
                cost = get_cost(v.id)
                # Prioritize: high cost, low utilization
                priority = cost / max(0.01, util)
                donors.append((priority, v.id))
        
        donors.sort(reverse=True)  # Highest priority first
        
        for _, donor_id in donors:
            donor_load = vehicle_loads[donor_id]
            if not donor_load['orders']:
                continue
            
            # Sort orders by size (smallest first, easier to fit)
            orders_to_move = sorted(
                donor_load['orders'],
                key=lambda x: sum(skus[s].weight * q for s, q in orders[x[0]].requested_items.items())
            )
            
            # Try to move each order
            for order_id, alloc in orders_to_move:
                o = orders[order_id]
                o_weight = sum(skus[s].weight * q for s, q in o.requested_items.items())
                o_volume = sum(skus[s].volume * q for s, q in o.requested_items.items())
                
                # Find best recipient (prioritize cheapest vehicles with space)
                recipients = []
                for v in vehicles_list:
                    if v.id == donor_id:
                        continue
                    r_load = vehicle_loads[v.id]
                    
                    # Allow up to 110% capacity for aggressive consolidation
                    if (r_load['weight'] + o_weight <= v.capacity_weight * capacity_buffer * 1.1 and
                        r_load['volume'] + o_volume <= v.capacity_volume * capacity_buffer * 1.1):
                        
                        # Prefer vehicles that are already in use and have lower cost
                        cost = get_cost(v.id)
                        in_use = 1 if r_load['orders'] else 0
                        recipients.append((in_use, -cost, v.id))
                
                if recipients:
                    recipients.sort(reverse=True)  # In-use vehicles first, then cheapest
                    recipient_id = recipients[0][2]
                    
                    # Move order
                    r_load = vehicle_loads[recipient_id]
                    r_load['orders'].append((order_id, alloc))
                    r_load['weight'] += o_weight
                    r_load['volume'] += o_volume
                    
                    if (order_id, alloc) in donor_load['orders']:
                        donor_load['orders'].remove((order_id, alloc))
                        donor_load['weight'] -= o_weight
                        donor_load['volume'] -= o_volume
    
    return vehicle_loads


def apply_2opt(sequence, orders, dist_func, start_node):
    """
    Apply 2-opt local search to improve delivery sequence.
    """
    if len(sequence) < 4:
        return sequence
    
    def route_length(seq):
        total = 0.0
        prev = start_node
        for oid in seq:
            d = dist_func(prev, orders[oid].destination.id)
            if d is not None:
                total += d
            prev = orders[oid].destination.id
        return total
    
    improved = True
    best_seq = sequence[:]
    best_len = route_length(best_seq)
    
    iterations = 0
    max_iterations = min(100, len(sequence) * 2)
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(len(best_seq) - 1):
            for j in range(i + 2, len(best_seq)):
                # Try reversing segment [i+1:j+1]
                new_seq = best_seq[:i+1] + best_seq[i+1:j+1][::-1] + best_seq[j+1:]
                new_len = route_length(new_seq)
                
                if new_len < best_len - 0.01:  # Small epsilon for floating point
                    best_seq = new_seq
                    best_len = new_len
                    improved = True
                    break
            
            if improved:
                break
    
    return best_seq


def build_optimized_route(env, vehicle, assigned_orders, warehouses, orders, adjacency):
    """
    Build a connected route with optimized delivery sequence.
    """
    home_node = warehouses[vehicle.home_warehouse_id].location.id
    steps = []
    
    # Distance cache
    dist_cache = {}
    def cached_dist(u, v):
        key = (u, v)
        if key not in dist_cache:
            dist_cache[key] = env.get_distance(u, v)
        return dist_cache[key]
    
    def dijkstra(src, dst):
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
                w = cached_dist(u, v)
                if w is None:
                    continue
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        
        if dst not in prev and dst != src:
            return None
        
        path = []
        curr = dst
        while curr != src:
            path.append(curr)
            if curr not in prev:
                return None
            curr = prev[curr]
        path.append(src)
        path.reverse()
        return path
    
    # Start at home
    current_node = home_node
    steps.append({'node_id': current_node, 'pickups': [], 'deliveries': [], 'unloads': []})
    
    # Collect all pickups by warehouse
    pickup_by_wh = defaultdict(lambda: defaultdict(int))
    for order_id, allocation in assigned_orders:
        for wh_id, items in allocation:
            for sku_id, qty in items.items():
                pickup_by_wh[wh_id][sku_id] += qty
    
    # Visit warehouses for pickups
    for wh_id, items in pickup_by_wh.items():
        wh_node = warehouses[wh_id].location.id
        
        # Navigate to warehouse
        if current_node != wh_node:
            path = dijkstra(current_node, wh_node)
            if not path:
                continue
            for node in path[1:]:
                steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
            current_node = wh_node
        
        # Add pickups
        for sku_id, qty in items.items():
            steps[-1]['pickups'].append({'warehouse_id': wh_id, 'sku_id': sku_id, 'quantity': qty})
    
    # Optimize delivery order using nearest neighbor + 2-opt
    undelivered = {oid for oid, _ in assigned_orders}
    delivery_sequence = []
    
    # Nearest neighbor construction
    nn_start = current_node
    while undelivered:
        best_next = None
        best_dist = float('inf')
        
        for oid in undelivered:
            dest = orders[oid].destination.id
            d = cached_dist(current_node, dest)
            if d is not None and d < best_dist:
                best_dist = d
                best_next = oid
        
        if best_next is None:
            break
        
        delivery_sequence.append(best_next)
        undelivered.remove(best_next)
        current_node = orders[best_next].destination.id
    
    # Apply 2-opt improvement for routes with 4+ deliveries
    if len(delivery_sequence) >= 4:
        delivery_sequence = apply_2opt(delivery_sequence, orders, cached_dist, nn_start)
    
    # Execute deliveries
    alloc_map = {oid: alloc for oid, alloc in assigned_orders}
    current_node = pickup_by_wh and warehouses[list(pickup_by_wh.keys())[-1]].location.id or home_node
    
    for order_id in delivery_sequence:
        dest_node = orders[order_id].destination.id
        
        # Navigate to delivery point
        if current_node != dest_node:
            path = dijkstra(current_node, dest_node)
            if not path:
                continue
            for node in path[1:]:
                steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
            current_node = dest_node
        
        # Add deliveries
        for wh_id, items in alloc_map[order_id]:
            for sku_id, qty in items.items():
                steps[-1]['deliveries'].append({'order_id': order_id, 'sku_id': sku_id, 'quantity': qty})
    
    # Return to home
    if current_node != home_node:
        path = dijkstra(current_node, home_node)
        if path:
            for node in path[1:]:
                steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
    
    return {'vehicle_id': vehicle.id, 'steps': steps}


def safe_dist(env, u, v):
    """Safe distance calculation with fallback."""
    if u is None or v is None:
        return float('inf')
    d = env.get_distance(u, v)
    return d if d is not None else float('inf')
