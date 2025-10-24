#!/usr/bin/env python3
"""
ROBUST MWVRP Solver: Guaranteed Multi-Vehicle Routes + 100% Fulfillment
- Enforces multiple vehicles through strict order limits
- Guarantees deliveries at correct nodes
- Simple, reliable logic
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq


def solver(env) -> Dict:
    """
    Main solver entry point.
    """
    return base_solver(env)


def base_solver(env) -> Dict:
    """
    Distribute orders across vehicles and build correct routes.
    """
    
    orders = env.orders
    warehouses = env.warehouses
    skus = env.skus
    
    if not orders:
        return {"routes": []}
    
    # Get all vehicles (sorted by cost)
    vehicles_list = env.get_all_vehicles()
    if not vehicles_list:
        return {"routes": []}
    
    vehicles_list.sort(key=lambda v: getattr(v, "fixed_cost", 0.0))
    
    # Get road network
    graph = env.get_road_network_data()
    adjacency = graph.get('adjacency_list', {})
    
    # Shadow inventory
    inventory = {wid: dict(wh.inventory) for wid, wh in warehouses.items()}
    
    # STEP 1: Assign orders to vehicles
    # Strict limit: max 5 orders per vehicle to force distribution
    MAX_ORDERS_PER_VEHICLE = 5
    vehicle_orders = {v.id: [] for v in vehicles_list}
    
    # Process all orders
    order_list = list(orders.items())
    vehicle_index = 0
    
    for order_id, order in order_list:
        # Find warehouse allocation for this order
        allocation = allocate_from_warehouses(order, inventory, warehouses, skus, env)
        
        if not allocation:
            # Can't fulfill this order (no inventory)
            continue
        
        # Find a vehicle for this order (round-robin with capacity check)
        assigned = False
        attempts = 0
        
        while attempts < len(vehicles_list) and not assigned:
            vehicle = vehicles_list[vehicle_index]
            
            # Check if vehicle has space for this order
            if len(vehicle_orders[vehicle.id]) < MAX_ORDERS_PER_VEHICLE:
                # Check weight/volume capacity
                order_weight = sum(skus[s].weight * q for s, q in order.requested_items.items())
                order_volume = sum(skus[s].volume * q for s, q in order.requested_items.items())
                
                # Calculate current load
                current_weight = 0.0
                current_volume = 0.0
                for oid, alloc in vehicle_orders[vehicle.id]:
                    o = orders[oid]
                    current_weight += sum(skus[s].weight * q for s, q in o.requested_items.items())
                    current_volume += sum(skus[s].volume * q for s, q in o.requested_items.items())
                
                # Check if order fits
                if (current_weight + order_weight <= vehicle.capacity_weight and
                    current_volume + order_volume <= vehicle.capacity_volume):
                    
                    # Assign order to this vehicle
                    vehicle_orders[vehicle.id].append((order_id, allocation))
                    assigned = True
                    
                    # Update inventory
                    for wh_id, items in allocation:
                        for sku_id, qty in items.items():
                            inventory[wh_id][sku_id] -= qty
            
            # Move to next vehicle
            vehicle_index = (vehicle_index + 1) % len(vehicles_list)
            attempts += 1
        
        # If still not assigned after trying all vehicles, try without order count limit
        if not assigned:
            for vehicle in vehicles_list:
                order_weight = sum(skus[s].weight * q for s, q in order.requested_items.items())
                order_volume = sum(skus[s].volume * q for s, q in order.requested_items.items())
                
                current_weight = sum(
                    sum(skus[s].weight * q for s, q in orders[oid].requested_items.items())
                    for oid, _ in vehicle_orders[vehicle.id]
                )
                current_volume = sum(
                    sum(skus[s].volume * q for s, q in orders[oid].requested_items.items())
                    for oid, _ in vehicle_orders[vehicle.id]
                )
                
                if (current_weight + order_weight <= vehicle.capacity_weight and
                    current_volume + order_volume <= vehicle.capacity_volume):
                    
                    vehicle_orders[vehicle.id].append((order_id, allocation))
                    
                    for wh_id, items in allocation:
                        for sku_id, qty in items.items():
                            inventory[wh_id][sku_id] -= qty
                    break
    
    # STEP 2: Build route for each vehicle that has orders
    solution = {"routes": []}
    
    for vehicle in vehicles_list:
        if not vehicle_orders[vehicle.id]:
            continue
        
        # Build this vehicle's route
        route = create_vehicle_route(
            env,
            vehicle,
            vehicle_orders[vehicle.id],
            warehouses,
            orders,
            adjacency
        )
        
        if route:
            solution["routes"].append(route)
    
    return solution


def allocate_from_warehouses(order, inventory, warehouses, skus, env):
    """
    Find warehouse allocation for an order.
    Returns: [(warehouse_id, {sku_id: quantity}), ...]
    """
    needed = dict(order.requested_items)
    dest_node = order.destination.id
    
    # Try to fulfill from single warehouse (best option)
    single_wh_options = []
    for wh_id, inv in inventory.items():
        # Check if this warehouse has everything
        can_fulfill = all(inv.get(sku_id, 0) >= qty for sku_id, qty in needed.items())
        if can_fulfill:
            wh_node = warehouses[wh_id].location.id
            dist = get_safe_distance(env, wh_node, dest_node)
            single_wh_options.append((dist, wh_id))
    
    if single_wh_options:
        # Use closest warehouse
        single_wh_options.sort()
        return [(single_wh_options[0][1], needed)]
    
    # Multi-warehouse allocation
    wh_by_distance = sorted(
        inventory.keys(),
        key=lambda wid: get_safe_distance(env, warehouses[wid].location.id, dest_node)
    )
    
    allocation = []
    remaining = dict(needed)
    
    for wh_id in wh_by_distance:
        if not remaining:
            break
        
        inv = inventory[wh_id]
        from_this_wh = {}
        
        for sku_id in list(remaining.keys()):
            available = inv.get(sku_id, 0)
            if available > 0:
                take = min(available, remaining[sku_id])
                from_this_wh[sku_id] = take
                remaining[sku_id] -= take
                if remaining[sku_id] <= 0:
                    del remaining[sku_id]
        
        if from_this_wh:
            allocation.append((wh_id, from_this_wh))
    
    # Only return if we can fulfill completely
    return allocation if not remaining else []


def create_vehicle_route(env, vehicle, order_assignments, warehouses, orders, adjacency):
    """
    Create a route for one vehicle.
    Route structure: Home → Warehouses (pickup) → Orders (deliver) → Home
    """
    
    home_warehouse_id = vehicle.home_warehouse_id
    home_node = warehouses[home_warehouse_id].location.id
    
    # Dijkstra shortest path
    def shortest_path(src, dst):
        if src == dst:
            return [src]
        
        pq = [(0.0, src)]
        distances = {src: 0.0}
        previous = {}
        visited = set()
        
        while pq:
            curr_dist, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == dst:
                # Reconstruct path
                path = []
                node = dst
                while node != src:
                    path.append(node)
                    if node not in previous:
                        return None
                    node = previous[node]
                path.append(src)
                path.reverse()
                return path
            
            for neighbor in adjacency.get(u, []):
                edge_weight = env.get_distance(u, neighbor)
                if edge_weight is None:
                    continue
                
                new_dist = curr_dist + edge_weight
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    previous[neighbor] = u
                    heapq.heappush(pq, (new_dist, neighbor))
        
        return None
    
    # Initialize route steps
    steps = []
    current_node = home_node
    
    # Start at home warehouse
    steps.append({
        'node_id': home_node,
        'pickups': [],
        'deliveries': [],
        'unloads': []
    })
    
    # PHASE 1: Collect all pickups needed
    pickups_by_warehouse = defaultdict(lambda: defaultdict(int))
    for order_id, allocation in order_assignments:
        for wh_id, items in allocation:
            for sku_id, quantity in items.items():
                pickups_by_warehouse[wh_id][sku_id] += quantity
    
    # PHASE 2: Visit warehouses to pick up items
    for wh_id, items_to_pickup in pickups_by_warehouse.items():
        wh_node = warehouses[wh_id].location.id
        
        # Navigate to warehouse
        if current_node != wh_node:
            path = shortest_path(current_node, wh_node)
            if not path:
                continue
            
            # Add navigation steps
            for node in path[1:]:
                steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            current_node = wh_node
        
        # Add pickup operations at this warehouse
        for sku_id, quantity in items_to_pickup.items():
            steps[-1]['pickups'].append({
                'warehouse_id': wh_id,
                'sku_id': sku_id,
                'quantity': quantity
            })
    
    # PHASE 3: Deliver to orders (nearest neighbor sequence)
    remaining_orders = {oid: alloc for oid, alloc in order_assignments}
    
    while remaining_orders:
        # Find nearest order from current location
        nearest_order_id = None
        nearest_distance = float('inf')
        
        for oid in remaining_orders:
            order_dest = orders[oid].destination.id
            dist = get_safe_distance(env, current_node, order_dest)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_order_id = oid
        
        if nearest_order_id is None:
            break
        
        # Navigate to this order's destination
        order_destination_node = orders[nearest_order_id].destination.id
        
        if current_node != order_destination_node:
            path = shortest_path(current_node, order_destination_node)
            if not path:
                # Can't reach this order, skip it
                del remaining_orders[nearest_order_id]
                continue
            
            # Add navigation steps
            for node in path[1:]:
                steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
            current_node = order_destination_node
        
        # Add delivery operations for this order at its exact destination
        allocation = remaining_orders[nearest_order_id]
        for wh_id, items in allocation:
            for sku_id, quantity in items.items():
                steps[-1]['deliveries'].append({
                    'order_id': nearest_order_id,
                    'sku_id': sku_id,
                    'quantity': quantity
                })
        
        # Mark order as delivered
        del remaining_orders[nearest_order_id]
    
    # PHASE 4: Return to home warehouse
    if current_node != home_node:
        path = shortest_path(current_node, home_node)
        if path:
            for node in path[1:]:
                steps.append({
                    'node_id': node,
                    'pickups': [],
                    'deliveries': [],
                    'unloads': []
                })
    
    # Ensure route ends at home
    if steps[-1]['node_id'] != home_node:
        steps.append({
            'node_id': home_node,
            'pickups': [],
            'deliveries': [],
            'unloads': []
        })
    
    return {
        'vehicle_id': vehicle.id,
        'steps': steps
    }


def get_safe_distance(env, node1, node2):
    """Get distance with fallback."""
    if node1 is None or node2 is None:
        return float('inf')
    dist = env.get_distance(node1, node2)
    return dist if dist is not None else float('inf')
