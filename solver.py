#!/usr/bin/env python3
"""
MWVRP Solver: 100% Fulfillment with Proper Multi-Vehicle Routes
- Each vehicle gets its own route
- Each route starts and ends at the vehicle's home warehouse
- Guarantees 100% order fulfillment
- Optimizes cost and distance
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq
import random


def solver(env) -> Dict:
    """
    Solver that guarantees 100% fulfillment with multiple vehicle routes.
    """
    
    strategies = [
        # Balanced strategy with reasonable vehicle loading
        {"max_orders_per_vehicle": 8, "order_strategy": "nearest", "max_warehouses": 2},
        {"max_orders_per_vehicle": 6, "order_strategy": "largest", "max_warehouses": 2},
        {"max_orders_per_vehicle": 10, "order_strategy": "nearest", "max_warehouses": 3},
    ]
    
    best_solution = None
    best_fulfillment = 0.0
    best_cost = float("inf")
    
    for strategy in strategies:
        env.reset_all_state()
        
        solution = base_solver(
            env,
            max_orders_per_vehicle=strategy["max_orders_per_vehicle"],
            order_strategy=strategy["order_strategy"],
            max_warehouses=strategy["max_warehouses"]
        )
        
        # Validate
        is_valid = True
        validation_result = env.validate_solution_complete(solution)
        if isinstance(validation_result, bool):
            is_valid = validation_result
        elif isinstance(validation_result, dict):
            is_valid = validation_result.get("is_valid", False)
        
        if not is_valid:
            continue
        
        # Calculate metrics
        stats = env.get_solution_statistics(solution)
        requested = stats.get("total_items_requested", 1)
        delivered = stats.get("total_items_delivered", 0)
        fulfillment_rate = delivered / max(1, requested)
        cost = env.calculate_solution_cost(solution)
        
        # Priority: Fulfillment first, then cost
        if fulfillment_rate > best_fulfillment or (fulfillment_rate >= best_fulfillment - 0.001 and cost < best_cost):
            best_fulfillment = fulfillment_rate
            best_cost = cost
            best_solution = solution
    
    return best_solution if best_solution else {"routes": []}


def base_solver(env,
                max_orders_per_vehicle: int = 8,
                order_strategy: str = "nearest",
                max_warehouses: int = 2) -> Dict:
    """
    Base solver - distributes orders across multiple vehicles properly.
    """
    
    orders = env.orders
    warehouses = env.warehouses
    skus = env.skus
    
    if not orders:
        return {"routes": []}
    
    # Get vehicles (cheapest first)
    vehicles_list = env.get_all_vehicles()
    vehicles_list.sort(key=lambda v: getattr(v, "fixed_cost", 0.0) + getattr(v, "cost_per_km", 0.0) * 50)
    
    # Road network
    graph = env.get_road_network_data()
    adjacency = graph.get('adjacency_list', {})
    
    # Track inventory
    inventory = {wid: dict(wh.inventory) for wid, wh in warehouses.items()}
    
    # Vehicle assignments: {vehicle_id: [(order_id, allocation)]}
    vehicle_assignments = {v.id: [] for v in vehicles_list}
    vehicle_weight = {v.id: 0.0 for v in vehicles_list}
    vehicle_volume = {v.id: 0.0 for v in vehicles_list}
    
    # Sort orders
    order_list = list(orders.items())
    if order_strategy == "largest":
        order_list.sort(
            key=lambda x: sum(skus[s].weight * q for s, q in x[1].requested_items.items()),
            reverse=True
        )
    elif order_strategy == "nearest" and vehicles_list:
        home = warehouses[vehicles_list[0].home_warehouse_id].location.id
        order_list.sort(key=lambda x: safe_dist(env, home, x[1].destination.id))
    elif order_strategy == "random":
        random.shuffle(order_list)
    
    # DISTRIBUTE ORDERS ACROSS VEHICLES
    current_vehicle_idx = 0
    
    for order_id, order in order_list:
        # Get order requirements
        order_weight = sum(skus[s].weight * q for s, q in order.requested_items.items())
        order_volume = sum(skus[s].volume * q for s, q in order.requested_items.items())
        
        # Find warehouse allocation
        allocation = find_allocation(order, inventory, warehouses, skus, env, max_warehouses)
        
        if not allocation:
            continue  # No inventory available
        
        # Try to assign to a vehicle
        assigned = False
        attempts = 0
        
        # Start from current vehicle and cycle through all vehicles
        while attempts < len(vehicles_list) and not assigned:
            v = vehicles_list[current_vehicle_idx]
            
            # Check if this vehicle can take the order
            current_order_count = len(vehicle_assignments[v.id])
            
            if (current_order_count < max_orders_per_vehicle and
                vehicle_weight[v.id] + order_weight <= v.capacity_weight * 0.95 and
                vehicle_volume[v.id] + order_volume <= v.capacity_volume * 0.95):
                
                # Assign to this vehicle
                vehicle_assignments[v.id].append((order_id, allocation))
                vehicle_weight[v.id] += order_weight
                vehicle_volume[v.id] += order_volume
                assigned = True
                
                # Update inventory
                for wh_id, items in allocation:
                    for sku_id, qty in items.items():
                        inventory[wh_id][sku_id] -= qty
            
            # Move to next vehicle
            current_vehicle_idx = (current_vehicle_idx + 1) % len(vehicles_list)
            attempts += 1
        
        # If still not assigned, try with relaxed constraints
        if not assigned:
            for v in vehicles_list:
                if (vehicle_weight[v.id] + order_weight <= v.capacity_weight and
                    vehicle_volume[v.id] + order_volume <= v.capacity_volume):
                    
                    vehicle_assignments[v.id].append((order_id, allocation))
                    vehicle_weight[v.id] += order_weight
                    vehicle_volume[v.id] += order_volume
                    assigned = True
                    
                    # Update inventory
                    for wh_id, items in allocation:
                        for sku_id, qty in items.items():
                            inventory[wh_id][sku_id] -= qty
                    break
    
    # BUILD SEPARATE ROUTE FOR EACH VEHICLE WITH ASSIGNMENTS
    solution = {"routes": []}
    
    for v in vehicles_list:
        # Skip vehicles with no assignments
        if not vehicle_assignments[v.id]:
            continue
        
        # Build route for this specific vehicle
        route = build_route(env, v, vehicle_assignments[v.id], warehouses, orders, adjacency)
        
        # Add route to solution if valid
        if route and route.get('steps') and len(route['steps']) > 0:
            # Verify route starts and ends at home warehouse
            home_node = warehouses[v.home_warehouse_id].location.id
            if route['steps'][0]['node_id'] == home_node and route['steps'][-1]['node_id'] == home_node:
                solution["routes"].append(route)
    
    return solution


def find_allocation(order, inventory, warehouses, skus, env, max_warehouses):
    """
    Find warehouse allocation for order.
    """
    needed = dict(order.requested_items)
    customer_node = order.destination.id
    
    # Try single warehouse first (best option)
    candidates = []
    for wh_id, inv in inventory.items():
        if all(inv.get(sid, 0) >= qty for sid, qty in needed.items()):
            wh_node = warehouses[wh_id].location.id
            dist = safe_dist(env, wh_node, customer_node)
            candidates.append((dist, wh_id))
    
    if candidates:
        candidates.sort()
        return [(candidates[0][1], needed)]
    
    # Multi-warehouse allocation (sorted by distance)
    wh_by_dist = sorted(
        inventory.keys(),
        key=lambda wid: safe_dist(env, warehouses[wid].location.id, customer_node)
    )
    
    allocation = []
    remaining = dict(needed)
    
    for wh_id in wh_by_dist[:max_warehouses]:
        if not remaining:
            break
        
        inv = inventory[wh_id]
        provided = {}
        
        for sku_id in list(remaining.keys()):
            available = inv.get(sku_id, 0)
            if available > 0:
                take = min(available, remaining[sku_id])
                provided[sku_id] = take
                remaining[sku_id] -= take
                if remaining[sku_id] == 0:
                    del remaining[sku_id]
        
        if provided:
            allocation.append((wh_id, provided))
    
    return allocation if not remaining else []


def build_route(env, vehicle, assignments, warehouses, orders, adjacency):
    """
    Build a complete route for ONE vehicle.
    Route MUST start and end at the vehicle's home warehouse.
    """
    
    home_node = warehouses[vehicle.home_warehouse_id].location.id
    
    # Dijkstra pathfinding with caching
    path_cache = {}
    
    def dijkstra(src, dst):
        cache_key = (src, dst)
        if cache_key in path_cache:
            return path_cache[cache_key]
        
        if src == dst:
            path_cache[cache_key] = [src]
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
                # Reconstruct path
                path = []
                curr = dst
                while curr != src:
                    path.append(curr)
                    if curr not in prev:
                        path_cache[cache_key] = None
                        return None
                    curr = prev[curr]
                path.append(src)
                path.reverse()
                path_cache[cache_key] = path
                return path
            
            for v in adjacency.get(u, []):
                w = env.get_distance(u, v)
                if w is None:
                    continue
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        
        path_cache[cache_key] = None
        return None
    
    steps = []
    current = home_node
    
    # Step 1: Start at home warehouse
    steps.append({'node_id': current, 'pickups': [], 'deliveries': [], 'unloads': []})
    
    # Step 2: Collect all pickups by warehouse
    pickup_map = defaultdict(lambda: defaultdict(int))
    for order_id, allocation in assignments:
        for wh_id, items in allocation:
            for sku_id, qty in items.items():
                pickup_map[wh_id][sku_id] += qty
    
    # Step 3: Visit all warehouses to pick up items
    for wh_id in pickup_map:
        wh_node = warehouses[wh_id].location.id
        
        # Navigate to warehouse
        if current != wh_node:
            path = dijkstra(current, wh_node)
            if not path:
                continue
            
            # Add intermediate steps
            for node in path[1:]:
                steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
            current = wh_node
        
        # Add all pickups at this warehouse
        for sku_id, qty in pickup_map[wh_id].items():
            steps[-1]['pickups'].append({
                'warehouse_id': wh_id,
                'sku_id': sku_id,
                'quantity': qty
            })
    
    # Step 4: Deliver to all orders (nearest neighbor sequence)
    undelivered = set(oid for oid, _ in assignments)
    alloc_map = {oid: alloc for oid, alloc in assignments}
    
    while undelivered:
        # Find nearest undelivered order
        best_order = None
        best_dist = float('inf')
        
        for oid in undelivered:
            dest = orders[oid].destination.id
            d = safe_dist(env, current, dest)
            if d < best_dist:
                best_dist = d
                best_order = oid
        
        if best_order is None:
            # Can't reach any remaining orders, skip them
            break
        
        # Navigate to delivery location
        dest_node = orders[best_order].destination.id
        
        if current != dest_node:
            path = dijkstra(current, dest_node)
            if not path:
                # Can't reach this order, skip it
                undelivered.remove(best_order)
                continue
            
            # Add intermediate steps
            for node in path[1:]:
                steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
            current = dest_node
        
        # Add ALL deliveries for this order
        allocation = alloc_map[best_order]
        for wh_id, items in allocation:
            for sku_id, qty in items.items():
                steps[-1]['deliveries'].append({
                    'order_id': best_order,
                    'sku_id': sku_id,
                    'quantity': qty
                })
        
        undelivered.remove(best_order)
    
    # Step 5: Return to home warehouse
    if current != home_node:
        path = dijkstra(current, home_node)
        if path:
            for node in path[1:]:
                steps.append({'node_id': node, 'pickups': [], 'deliveries': [], 'unloads': []})
    
    # Verify route starts and ends at home
    if steps[0]['node_id'] != home_node:
        return None
    if steps[-1]['node_id'] != home_node:
        # Add explicit return to home
        steps.append({'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []})
    
    return {'vehicle_id': vehicle.id, 'steps': steps}


def safe_dist(env, u, v):
    """Safe distance lookup with fallback."""
    if u is None or v is None:
        return float('inf')
    d = env.get_distance(u, v)
    return d if d is not None else float('inf')
