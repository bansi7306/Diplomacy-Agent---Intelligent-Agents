import time
import timeout_decorator
import random
import networkx as nx
from agent_baselines import Agent

'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file. If you do so, measure the
    running time of __init__, new_game, update_game, and get_actions yourself
    (for example, with time.perf_counter). This local workaround does not relax
    the one-second limit: it is a hard constraint and will be enforced
    independently during marking.
'''
from agent_baselines import Agent

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from agent_baselines.py first.
    
    You can add/override attributes and methods as needed.
    '''
    ARMY_GRAPH_DIAMETER = 10  # computed offline on the standard map's largest connected army component
    NAVY_GRAPH_DIAMETER = 13  # computed offline on the standard map's largest connected navy component

    @timeout_decorator.timeout(1)
    def __init__(self, agent_name='Group09Agent'):
        super().__init__(agent_name)
        self.map_graph_army = None
        self.map_graph_navy = None

        '''Implement your agent here.
        Opponent Model'''
        self.opponent_model = {}


    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_map_graphs()
        self.stranded_armies = set()   # armies with no useful land move and no convoy available this turn

        # The map never changes, so compute every shortest-path distance once per game
        self.army_distances = dict(nx.all_pairs_shortest_path_length(self.map_graph_army))
        self.navy_distances = dict(nx.all_pairs_shortest_path_length(self.map_graph_navy))

        # Convoy geography: separate landmasses (e.g. Britain vs the continent), sea zones that
        # bridge two landmasses, and the coastal provinces that touch a bridge sea (launch ports)
        components = [c for c in nx.connected_components(self.map_graph_army) if len(c) > 1]
        self.army_region = {node: i for i, comp in enumerate(components) for node in comp}
        self.bridge_seas = set()
        for sea in self.map_graph_navy.nodes:
            if self.game.map.area_type(sea) != 'WATER':
                continue
            regions = {self.army_region[c] for c in self.map_graph_navy.neighbors(sea) if c in self.army_region}
            if len(regions) >= 2:
                self.bridge_seas.add(sea)
        self.launch_ports = {c for sea in self.bridge_seas
                             for c in self.map_graph_navy.neighbors(sea) if c in self.army_region}
        self.armies_to_port = set()   # stranded armies walking to a launch port

        '''Implement your agent here.'''

        # Initialise opponent aggression for each new game
        self.opponent_model = {}

        for opponent in self.game.powers:
            if opponent != self.power_name:
                self.opponent_model[opponent] = {
                    'aggression': 0.0,
                    'relationship_score': 0.0,

                    'activity': 0.0,
                    'activity_history': [],
                    'hostility': 0.0
                }

        # System 2: persistent strategic target
        self.current_target = None
        self.current_targets = {}
        self.turns_since_target_progress = 0
        self.last_target_distance = None

    def update_opponent_activity(self, all_power_orders, unit_counts):

        for power_name, data in self.opponent_model.items():
            orders = all_power_orders.get(power_name, [])
            total_units = unit_counts.get(power_name, 0)

            if total_units == 0:
                turn_activity = 0.0

            else:
                move_orders = 0
                
                for order in orders:
                    parts = order.split()

                    if len(parts) >= 4 and parts[2] == "-":
                        move_orders += 1

                turn_activity = move_orders / total_units

            data["activity_history"].append(turn_activity)

            data["activity_history"] = data["activity_history"][-4:]
            data["activity"] = (
                sum(data["activity_history"])
                / len(data["activity_history"])
            )

    def update_opponent_hostility(self, all_power_orders, lost_centres, our_centres_before, our_units_before):

        our_locations_before = our_centres_before | our_units_before
        nearby_locations_before = set()

        for location in our_locations_before:
            adjacent_locations = self.game.map.abut_list(location)

            for adjacent in adjacent_locations:
                nearby_locations_before.add(adjacent[:3])

        for power_name, data in self.opponent_model.items():
            hostile_actions = 0.0
            orders = all_power_orders.get(power_name, [])

            for order in orders:
                parts = order.split()

                if len(parts) >= 4 and parts[2] == "-":
                    destination = parts[3][:3]

                    if destination in our_locations_before:
                        hostile_actions += 1.0

                    elif destination in nearby_locations_before:
                        hostile_actions += 0.25
                    

                elif len(parts) >= 5 and parts[2] == "S":
                    if "-" in parts:
                        destination = parts[-1][:3]

                        if destination in our_locations_before:
                            hostile_actions += 1.0

                        elif destination in nearby_locations_before:
                            hostile_actions += 0.25

            for centre in lost_centres:

                for unit in self.game.get_units(power_name):
                    unit_location = unit.split()[1][:3]

                    if unit_location == centre:
                        hostile_actions += 1.0
                        break

            old_hostility = data["hostility"]

            data["hostility"] = (
                0.7 * old_hostility
                + hostile_actions
            )

    def get_adaptive_opponent_threat(self, power_name):
        data = self.opponent_model.get(power_name)

        if data is None:
            return 0.0

        activity = data["activity"]
        hostility = data["hostility"]

        base_threat = 1.0
        return activity * (base_threat + hostility)            
    
    # Updates opponent aggression based on their observed orders
    def update_opponent_aggression(self, all_power_orders):

        for opponent, orders in all_power_orders.items():

            # Ignore our own orders
            if opponent == self.power_name:
                continue

            if opponent not in self.opponent_model:
                self.opponent_model[opponent] = {
                    'aggression': 0.0,
                    'relationship_score': 0.0
                }

            if not orders:
                continue

            attack_count = 0

            # Count how many orders involve attacking our units or centres
            own_units = self.game.get_units(self.power_name)
            own_centres = self.game.get_centers(self.power_name)

            own_locations = {
                unit.split()[1] for unit in own_units
            }

            own_territory = own_locations.union(own_centres)

            for order in orders:

                if ' - ' not in order:
                    continue

                parts = order.split()

                if '-' not in parts:
                    continue

                destination = parts[parts.index('-') + 1]

                if destination in own_territory:
                    attack_count += 1

            # Proportion of opponent orders that attack us
            attack_ratio = attack_count / len(orders)

            # Exponential moving average
            old_aggression = self.opponent_model[opponent]['aggression']

            new_aggression = (
                0.7 * old_aggression
                + 0.3 * attack_ratio
            )

            self.opponent_model[opponent]['aggression'] = new_aggression

    # Updates whether opponents are friendly, neutral or hostile
    # based on their observed movement-phase orders
    def update_opponent_relationships(self, all_power_orders):

        own_units = self.game.get_units(self.power_name)
        own_centres = self.game.get_centers(self.power_name)

        own_locations = {
            unit.split()[1] for unit in own_units
        }

        own_territory = own_locations.union(own_centres)

        for opponent, orders in all_power_orders.items():

            # Ignore our own orders
            if opponent == self.power_name:
                continue

            if opponent not in self.opponent_model:
                self.opponent_model[opponent] = {
                    'aggression': 0.0,
                    'relationship_score': 0.0
                }

            if not orders:
                continue

            hostile_count = 0
            friendly_count = 0

            for order in orders:

                parts = order.split()

                # Example: A BUR - PAR
                # Moving into our unit's location or supply centre
                if '-' in parts and 'S' not in parts:

                    dash_index = parts.index('-')

                    if dash_index + 1 < len(parts):
                        destination = parts[dash_index + 1]

                        if destination in own_territory:
                            hostile_count += 1

                # Example: A BUR S A PAR - PIC
                # Supporting one of our units
                elif 'S' in parts:

                    support_index = parts.index('S')

                    if support_index + 2 < len(parts):

                        supported_unit = ' '.join(
                            parts[support_index + 1:support_index + 3]
                        )

                        if supported_unit in own_units:
                            friendly_count += 1

            # Convert observed behaviour into a score from -1 to +1
            behaviour_score = (
                friendly_count - hostile_count
            ) / len(orders)

            # Gradually update the relationship using an
            # exponential moving average
            old_score = self.opponent_model[opponent].get('relationship_score', 0.0)

            new_score = (
                0.7 * old_score
                + 0.3 * behaviour_score
            )

            self.opponent_model[opponent]['relationship_score'] = new_score

            

    # Returns a dictionary of opponent relationship scores on a 0-10 scale
    # 0 = very friendly, 5 = neutral, 10 = very hostile
    def get_all_opponent_relationships(self):

        relationships = {}

        for opponent in self.game.powers:

            if opponent == self.power_name:
                continue

            relationship_score = self.opponent_model.get(
                opponent, {}
            ).get('relationship_score', 0.0)

            # Convert -1 to +1 into a 0 to 10 scale
            scaled_score = 5 - (5 * relationship_score)

            relationships[opponent] = round(scaled_score, 2)

        return relationships

    # Measures how much pressure an opponent places on our territory
    def get_opponent_pressure(self, opponent):

        enemy_units = self.game.get_units(opponent)

        own_units = self.game.get_units(self.power_name)
        own_centres = self.game.get_centers(self.power_name)

        own_locations = {
            unit.split()[1] for unit in own_units
        }

        own_territory = own_locations.union(own_centres)

        if not own_territory:
            return 0.0

        threatened_locations = set()

        for unit in enemy_units:

            parts = unit.split()

            unit_type = parts[0]
            location = parts[1]

            graph = (
                self.map_graph_army
                if unit_type == 'A'
                else self.map_graph_navy
            )

            if location not in graph:
                continue

            # Find locations the enemy unit can reach
            for neighbour in graph.neighbors(location):

                if neighbour in own_territory:
                    threatened_locations.add(neighbour)

        # Proportion of our territory threatened by this opponent
        pressure = (
            len(threatened_locations)
            / len(own_territory)
        )

        return pressure

    # Estimates an opponent's strength relative to all active powers
    def get_opponent_strength(self, opponent):

        opponent_units = len(
            self.game.get_units(opponent)
        )

        total_units = 0

        for power in self.game.powers:

            total_units += len(
                self.game.get_units(power)
            )

        if total_units == 0:
            return 0.0

        return opponent_units / total_units

    # Combines aggression, pressure and strength into a threat score
    def get_opponent_threat(self, opponent, weights):

        if self.TECHNIQUES['opponent_modelling']:
            return self.get_adaptive_opponent_threat(opponent)

        aggression = self.opponent_model.get(
            opponent, {}
        ).get('aggression', 0.0)

        pressure = self.get_opponent_pressure(
            opponent
        )

        strength = self.get_opponent_strength(
            opponent
        )

        threat = (
            weights['aggression_weight'] * aggression
            + weights['pressure_weight'] * pressure
            + weights['strength_weight'] * strength
        )

        return threat

    # Identifies which opponent occupies a location
    def get_location_opponent(self, location):

        for opponent in self.game.powers:

            if opponent == self.power_name:
                continue

            for unit in self.game.get_units(opponent):

                unit_location = unit.split()[1]

                if unit_location == location:
                    return opponent

        return None

    def get_centre_owner(self, location):
        location = location[:3]

        for power_name in self.game.powers:
            centres = self.game.get_centers(power_name)

            for centre in centres:
                if centre[:3] == location:
                    return power_name
        return None

    def is_truce_power(self, power_name):
        if power_name == self.power_name:
            return False
        data = self.opponent_model.get(power_name)

        if data is None:
            return False

        try:
            year = int(self.game.get_current_phase()[1:5])

            if year > 1912:
                return False
        except (ValueError, TypeError):
            pass

        activity_history = data["activity_history"]
        if len(activity_history) < 4:
            return False

        is_static = (
            len(activity_history) >= 4
            and all(activity == 0.0 for activity in activity_history)
        )
        if is_static:
            return False

        return data["hostility"] <= 0.15

    def get_attackable_centres(self, enemy_centres):
        if not self.TECHNIQUES['opponent_modelling']:
            return enemy_centres

        preferred_centres = []
        truce_centres = []

        for centre in enemy_centres:
            owner = self.get_centre_owner(centre)

            if owner is not None and self.is_truce_power(owner):
                truce_centres.append(centre)
            else:
                preferred_centres.append(centre)

        if preferred_centres:
            return preferred_centres

        return truce_centres

    # Calculates the highest opponent threat around a destination
    def get_destination_threat(self, destination, weights):

        highest_threat = 0.0

        for opponent in self.game.powers:

            if opponent == self.power_name:
                continue

            enemy_units = self.game.get_units(opponent)

            for unit in enemy_units:

                parts = unit.split()

                unit_type = parts[0]
                location = parts[1]

                graph = (
                    self.map_graph_army
                    if unit_type == 'A'
                    else self.map_graph_navy
                )

                if location not in graph:
                    continue

                # Check whether the opponent occupies or threatens the destination
                if (
                    location == destination
                    or destination in graph.neighbors(location)
                ):

                    #Threat per opponent only changes once per turn, so compute it the first time it's needed this turn and reuse it
                    if opponent not in self.opponent_threat_cache:
                        self.opponent_threat_cache[opponent] = self.get_opponent_threat(opponent, weights)
                    threat = self.opponent_threat_cache[opponent]

                    highest_threat = max(
                        highest_threat,
                        threat
                    )

                    break

        return highest_threat

    #Builds an army and navy adjacency graphs from the map data, it is used for distance scoring and resused from GreedyAgent
    def build_map_graphs(self):
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(self.game.map.loc_type.keys())

        for i in locations:
            if self.game.map.loc_type[i] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(i.upper())
            if self.game.map.loc_type[i] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(i.upper())

        locations = [i.upper() for i in locations]

        for i in locations:
            for j in locations:
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    #This function returns every supply center not currently controlled by us, this is essentially our target list for scoring moves
    def get_enemy_centres(self):
        enemy_centres = []
        for i in self.game.map.scs:
            #This is all the centers not controlled by us
            if i not in self.game.get_centers(self.power_name):
                enemy_centres.append(i)
        return enemy_centres

    # Distance from our closest unit to a centre (None if no unit can reach it)
    def distance_to_target(self, target):
        if target is None:
            return None
        best = None
        for unit in self.game.get_units(self.power_name):
            parts = unit.replace('*', '').split()
            distances = self.army_distances if parts[0] == 'A' else self.navy_distances
            d = distances.get(parts[1], {}).get(target)
            if d is not None and (best is None or d < best):
                best = d
        return best

    # Score a centre as a System 2 target: reachable soon, preferably undefended
    def target_score(self, centre, weights, occupied):
        d = self.distance_to_target(centre)
        if d is None:
            return None
        score = max(0, weights['distance_cap'] - d)
        if self.S2_PREFER_EMPTY and centre not in occupied:
            score += 2
        score -= weights['opponent_multiplier'] * self.get_destination_threat(centre, weights)
        return score

    # Moves between two centres (army route if there is one, else fleet route)
    def centre_gap(self, a, b):
        d = self.army_distances.get(a, {}).get(b)
        if d is None:
            d = self.navy_distances.get(a, {}).get(b)
        return d if d is not None else 99

    # System 2: pick the enemy centre our units can reach soonest, preferring undefended
    # ones, skipping truce partners while other targets exist (opponent modelling)
    def choose_strategic_target(self, enemy_centres, weights, exclude=None):
        candidates = [c for c in self.get_attackable_centres(enemy_centres) if c != exclude]
        if not candidates:
            candidates = self.get_attackable_centres(enemy_centres)
        occupied = {l[:3] for l in self.enemy_occupied}

        best_target = None
        best_score = -1000
        for centre in candidates:
            d = self.distance_to_target(centre)
            if d is None:
                continue
            score = max(0, weights['distance_cap'] - d)
            # An empty enemy centre falls to a single move; an occupied one needs a supported attack
            if self.S2_PREFER_EMPTY and centre not in occupied:
                score += 2
            score -= weights['opponent_multiplier'] * self.get_destination_threat(centre, weights)
            if score > best_score:
                best_score = score
                best_target = centre
        return best_target

    def update_strategic_target(self, enemy_centres, weights):
        needed = 18 - len(self.game.get_centers(self.power_name))
        endgame = self.S2_ENDGAME and 0 < needed <= self.ENDGAME_NEEDED

        # Endgame: every enemy centre is fair game; otherwise skip truce partners (opponent modelling)
        pool = set(enemy_centres) if endgame else set(self.get_attackable_centres(enemy_centres))
        occupied = {l[:3] for l in self.enemy_occupied}

        # 1) Drop targets we captured, or that left the pool (e.g. a truce formed)
        for t in list(self.current_targets):
            if t not in pool:
                del self.current_targets[t]

        # 2) Progress-aware patience per target; give up on targets we stopped closing in on
        dropped = set()
        for t, info in list(self.current_targets.items()):
            d = self.distance_to_target(t)
            if self.S2_PROGRESS_PATIENCE and d is not None and info['last_d'] is not None and d < info['last_d']:
                info['stale'] = 0
            else:
                info['stale'] += 1
            info['last_d'] = d
            if info['stale'] > self.TARGET_PATIENCE:
                del self.current_targets[t]
                dropped.add(t)

        # 3) How many targets we want
        if not self.S2_MULTI_TARGET:
            wanted = 1
        elif endgame:
            wanted = needed
        else:
            wanted = max(1, min(self.S2_MAX_FRONTS, len(self.game.get_units(self.power_name)) // 3))

        # 4) Fill up with the best remaining centres
        while len(self.current_targets) < wanted:
            best, best_score = None, None
            for c in sorted(pool):
                if c in self.current_targets or c in dropped:
                    continue
                # Outside the endgame, each target must be on a different front
                if not endgame and any(self.centre_gap(c, t) < self.S2_FRONT_SPACING for t in self.current_targets):
                    continue
                s = self.target_score(c, weights, occupied)
                if s is not None and (best_score is None or s > best_score):
                    best, best_score = c, s
            if best is None:
                break
            self.current_targets[best] = {'stale': 0, 'last_d': self.distance_to_target(best)}

        # Keep one "main" target for traces and older code that expect a single target
        def dist_or_far(t):
            d = self.distance_to_target(t)
            return 99 if d is None else d
        self.current_target = min(self.current_targets, key=dist_or_far) if self.current_targets else None

    #This bundles our own units and orderable locations into one dict for easy lookup
    def get_own_units_and_locations(self):
        return {
            'units': self.game.get_units(self.power_name),
            'orderable_locations': self.game.get_orderable_locations(self.power_name)
        }

    #This splits a locations legal orders into moves, holds and supports so that each move type can be scored differently
    def classify_orders(self, possible_orders):
        moves = []
        holds = []
        supports = []
        for order in possible_orders:
            #These are support moves
            if ' S ' in order:
                supports.append(order)
            elif self.is_move_order(order) and not order.endswith(' VIA'):
                moves.append(order)
            #And this is our holding moves
            elif order.endswith(' H'):
                holds.append(order)
        return moves, holds, supports

    #This function pulls the destinations location of out a move order string to make processing easier
    #'A PAR - BUR' turns into 'BUR'
    def get_move_destination(self, order):
        words = order.split(' ')
        dash_index = words.index('-')
        return words[dash_index + 1]

    def is_support_position(self, graph, destination, enemy_centres):
        if destination not in graph:
            return False
        occupied = {loc[:3] for loc in self.enemy_occupied}
        for neighbour in graph.neighbors(destination):
            n = neighbour[:3]
            if n in enemy_centres and n in occupied:
                return True
        return False

    def nearest_centre_distance(self, graph, node, centres):
        all_lengths = self.army_distances if graph is self.map_graph_army else self.navy_distances
        lengths = all_lengths.get(node, {})
        dists = [lengths[c] for c in centres if c in lengths]
        return min(dists) if dists else None

    #This fucntion scores a destination by how close it is to the nearest enemy center by using the map graph to find the shortest path
    def distance_score(self, graph, destination, enemy_centres, weights):
        if destination not in graph:
            return 0

        # Lookup into distances precomputed in new_game(), instead of a
        # full shortest-path search on every call
        all_lengths = self.army_distances if graph is self.map_graph_army else self.navy_distances
        lengths = all_lengths.get(destination, {})

        min_dist = 1000
        for centre in enemy_centres:
            dist = lengths.get(centre)
            if dist is not None and dist < min_dist:
                min_dist = dist

        if min_dist == 1000:
            return 0

        return max(0, weights['distance_cap'] - min_dist)

    #This checks if any other units that we control has a legal support order backing this specific move
    def is_move_supportable(self, move_order, all_possible_orders, own_orderable_locations):
        for other_loc in own_orderable_locations:
            for candidate in all_possible_orders.get(other_loc, []):
                if ' S ' in candidate and move_order in candidate:
                    return True
        return False
    
    #This function checks if an enemy unit is currently occupying the space of our intended move destination
    def is_contested(self, destination):
        return destination in self.enemy_occupied or destination[:3] in self.enemy_occupied

    #True only for genuine move orders like 'A PAR - BUR'. Supports and convoys also contain ' - ', so we check the third word instead.
    def is_move_order(self, order):
        parts = order.split()
        return len(parts) >= 4 and parts[2] == '-'

    #This combines our scoring factors into one final score, so dfistance, support and contested are combined into one score for the candidate order
    #This is a combination of Ben's strategy and Bansi's oppeonent model'''

        #Replaces moves that are guaranteed to fail because of our own units: moving into a spot one of our units stays in, or two of our units swapping places. Falls back to the next candidate, then hold.
    def remove_self_blocks(self, final_orders, top3_by_location):
        chosen_index = {loc: 0 for loc in final_orders}

        def dest_of(order):
            return self.get_move_destination(order)[:3] if self.is_move_order(order) else None

        for _ in range(5):
            changed = False
            staying = {loc[:3] for loc, o in final_orders.items() if not self.is_move_order(o)}
            moving = {loc[:3]: dest_of(o) for loc, o in final_orders.items() if self.is_move_order(o)}

            def move_priority(l):
                order_l = final_orders[l]
                is_supported = any(' S ' in o and o.endswith(order_l) for o in final_orders.values())
                cands = top3_by_location.get(l)
                score = next((s for s, o in cands if o == order_l), 1000) if cands else 1000
                return (is_supported, score, l)

            movers_by_dest = {}
            for l, o in final_orders.items():
                d = dest_of(o)
                if d is not None:
                    movers_by_dest.setdefault(d, []).append(l)
            duplicate_losers = set()
            for d, locs_here in movers_by_dest.items():
                if len(locs_here) > 1:
                    keep = max(locs_here, key=move_priority)
                    duplicate_losers.update(l for l in locs_here if l != keep)

            for loc, order in list(final_orders.items()):
                dest = dest_of(order)
                if dest is None:
                    continue

                into_staying_unit = dest in staying
                swap = moving.get(dest) == loc[:3] and loc[:3] > dest  # block only one side of a swap
                duplicate = loc in duplicate_losers

                if not (into_staying_unit or swap or duplicate):
                    continue

                # try the next candidate for this unit, otherwise hold
                replacement = f'{order[0]} {loc} H'
                candidates = top3_by_location.get(loc, [])
                idx = chosen_index[loc] + 1
                while idx < len(candidates):
                    alt = candidates[idx][1]
                    alt_dest = dest_of(alt)
                    if alt_dest is None or alt_dest not in staying:
                        replacement = alt
                        break
                    idx += 1
                chosen_index[loc] = idx

                final_orders[loc] = replacement
                changed = True

            if not changed:
                break

        return final_orders

    def score_order(self, is_hold, distance_score_val, is_supportable, is_contested_flag, weights, opponent_threat=0.0):

        if is_hold:
            return weights['hold_baseline']

        score = distance_score_val

        # Support only matters when the destination is defended - moving into
        # empty or friendly territory gains nothing from it
        # when the switch is off, fall back to the old behaviour: bonus for any supportable move
        if is_supportable and (is_contested_flag or not self.TECHNIQUES['contested_support_only']):
            score += weights['support_bonus']

        if is_contested_flag:
            score -= weights['contest_penalty']

        # Opponent modelling contribution
        score -= weights['opponent_multiplier'] * opponent_threat

        return score

    #This scores every candidate order at one location and returns the best one only for the movement phase
    def score_movement_location(self, loc, all_possible_orders, own_orderable_locations, enemy_centres, weights):
        possible_orders = all_possible_orders.get(loc, [])
        if not possible_orders:
            return []

        moves, holds, supports = self.classify_orders(possible_orders)

        scored_candidates = []
        attackable_centres = self.get_attackable_centres(enemy_centres)

        for move in moves:
            unit_type = move[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            destination = self.get_move_destination(move)

            dist_score = self.distance_score(graph, destination, attackable_centres, weights)

            # Progress scoring: reward moves that get closer to an enemy centre,
            # not moves that are merely close. Stops rear units shuffling sideways.
            if self.TECHNIQUES['progress_scoring']:
                cur_d = self.nearest_centre_distance(graph, loc, attackable_centres)
                new_d = self.nearest_centre_distance(graph, destination, attackable_centres)
                if cur_d is not None and new_d is not None:
                    if new_d < cur_d:
                        dist_score = max(dist_score, 1.5)   # real progress always beats holding (1.0)
                    elif destination[:3] not in attackable_centres and not self.is_support_position(graph, destination, attackable_centres):
                        dist_score = 0
            # Convoy positioning: a fleet moving into a bridge sea next to a stranded army
            # makes a convoy across to another landmass possible next turn
            if (self.TECHNIQUES['convoys'] and unit_type == 'F' and destination in self.bridge_seas
                    and any(self.navy_distances.get(destination, {}).get(a) == 1 for a in self.stranded_armies)):
                dist_score = max(dist_score, 2.0)

            # Stranded army walking to a launch port (e.g. LVP -> YOR) so it can be convoyed
            if self.TECHNIQUES['convoys'] and unit_type == 'A' and loc in self.armies_to_port:
                lengths_now = self.army_distances.get(loc, {})
                lengths_new = self.army_distances.get(destination, {})
                cur_p = min((lengths_now[p] for p in self.launch_ports if p in lengths_now), default=None)
                new_p = min((lengths_new[p] for p in self.launch_ports if p in lengths_new), default=None)
                if cur_p is not None and new_p is not None and new_p < cur_p:
                    dist_score = max(dist_score, 1.5)
            # System 2: bonus for progressing toward our committed
            # long-term target, on top of the general "closer to any
            # centre" signal
            if self.S2_TARGET_PULL and destination[:3] in self.current_targets:
                dist_score += 3

            supportable = self.is_move_supportable(move, all_possible_orders, own_orderable_locations)

            contested = self.is_contested(destination)

            # Get opponent threat associated with this destination
            opponent_threat = self.get_destination_threat(destination, weights) if self.TECHNIQUES['opponent_modelling'] else 0.0

            total_score = self.score_order(
                False,
                dist_score,
                supportable,
                contested,
                weights,
                opponent_threat
            )

            # Stage 5 implementation of the shallow lookahead
            if self.TECHNIQUES['lookahead'] and not contested and not supportable:
                if self.could_enemy_contest(destination, all_possible_orders):
                    total_score -= weights['lookahead_penalty']

            scored_candidates.append((total_score, move))

        # Centres only change owner at the end of Fall, so a unit sitting on a
        # centre we don't own must hold in Fall or it never captures it
        is_fall = self.game.get_current_phase().startswith('F')
        on_uncaptured_centre = loc[:3] in enemy_centres

        for hold in holds:
            total_score = self.score_order(True, 0, False, False, weights)
            if self.TECHNIQUES['fall_capture_hold'] and is_fall and on_uncaptured_centre:
                total_score = 100
            scored_candidates.append((total_score, hold))

        if not scored_candidates:
            return [(0, possible_orders[0])]

        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        return scored_candidates[:3]
    
    #This function returns the supply centres we currently control. It's used as the safety target for the retreat scoring
    def get_own_centres(self):
        return self.game.get_centers(self.power_name)

    #This function pulls the destination out of a retreat order string
    def get_retreat_destination(self, order):
        words = order.split(' ')
        r_index = words.index('R')
        return words[r_index + 1]

    #This function splits a locations legal orders into retreats and disbands
    def classify_retreat_orders(self, possible_orders):
        retreats = []
        disbands = []
        for order in possible_orders:
            if ' R ' in order:
                retreats.append(order)
            elif order.endswith(' D'):
                disbands.append(order)
        return retreats, disbands

    #This function then scores every retreat option by safety. This is done be evaluating the closeness to our own centres, and also avoiding contested spots and returns the best one.
    def score_retreat_location(self, loc, all_possible_orders, own_centres, weights):
        possible_orders = all_possible_orders.get(loc, [])
        if not possible_orders:
            return None

        retreats, disbands = self.classify_retreat_orders(possible_orders)

        if not retreats:
            return disbands[0] if disbands else possible_orders[0]

        scored_candidates = []
        for retreat in retreats:
            unit_type = retreat[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            destination = self.get_retreat_destination(retreat)

            safety_score = self.distance_score(graph, destination, own_centres, weights)
            contested = self.is_contested(destination)

            total_score = safety_score - (weights['contest_penalty'] if contested else 0)
            scored_candidates.append((total_score, retreat))

        max_score = max(s for s, o in scored_candidates)
        top_candidates = [o for s, o in scored_candidates if s == max_score]
        return sorted(top_candidates)[0]

    #This calculates how many units we're allowed to build this turn
    def get_required_builds(self):
        own_centres = self.get_own_centres()
        own_units = self.game.get_units(self.power_name)
        return max(0, len(own_centres) - len(own_units))

    #This function is for disbanding and finds out how many units we're forced to disband this turn
    def get_required_disbands(self):
        own_centres = self.get_own_centres()
        own_units = self.game.get_units(self.power_name)
        return max(0, len(own_units) - len(own_centres))

    #This function splits a locations legal orders into builds and disbands
    def classify_adjustment_orders(self, possible_orders):
        builds = []
        disbands = []
        for order in possible_orders:
            if order.endswith(' B'):
                builds.append(order)
            elif order.endswith(' D'):
                disbands.append(order)
        return builds, disbands

    #This function scores a home centres best build option by closeness to enemy territory, we want to build near where the fighting is.
    def score_build_location(self, loc, all_possible_orders, enemy_centres, weights):
        possible_orders = all_possible_orders.get(loc, [])
        builds, disbands = self.classify_adjustment_orders(possible_orders)

        if not builds:
            return None, -1

        best_order = None
        best_score = -1
        for build in builds:
            unit_type = build[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            score = self.distance_score(graph, loc, enemy_centres, weights)
            if score > best_score:
                best_score = score
                best_order = build

        return best_order, best_score

    #This function constructs a support order string for one unit that is backing up another unit's move
    def build_support_order(self, supporting_unit_type, supporting_loc, winning_move):
        return f'{supporting_unit_type} {supporting_loc} S {winning_move}'

    #This function then checks if a constructued support order is actually a legal move for the unit in question
    def is_support_legal(self, support_order, all_possible_orders, loc):
        return support_order in all_possible_orders.get(loc, [])

    #This function finds units that are targetting the same destination and resolves them by letting the winner keep its first choice move, checking if the loser can support it, and if not make the loser fall to its 2nd/3rd choice
    def resolve_collisions(self, top3_by_location, all_possible_orders):
        final_orders = {}
        chosen_index = {loc: 0 for loc in top3_by_location}

        def current_pick(loc):
            candidates = top3_by_location[loc]
            idx = chosen_index[loc]
            if idx >= len(candidates):
                return None
            return candidates[idx][1]

        def get_destination_if_move(order):
            if self.is_move_order(order):
                # Strip coast variants so 'BUL/EC' and 'BUL' count as the same province
                return self.get_move_destination(order)[:3]
            return None

        locations = list(top3_by_location.keys())

        #This group the current picks by destination to find collisions
        destination_map = {}
        for loc in locations:
            order = current_pick(loc)
            dest = get_destination_if_move(order) if order else None
            if dest:
                destination_map.setdefault(dest, []).append(loc)

        collided_locs = set()
        winners = {}
        for dest, locs_here in destination_map.items():
            if len(locs_here) > 1:
                #The winner is the highest score among current picks at this destination
                best_loc = max(locs_here, key=lambda l: top3_by_location[l][chosen_index[l]][0])
                winners[dest] = best_loc
                for l in locs_here:
                    if l != best_loc:
                        collided_locs.add(l)

        #We then resolve each collided (losing) unit
        for loc in collided_locs:
            candidates = top3_by_location[loc]
            order = candidates[chosen_index[loc]][1]
            dest = get_destination_if_move(order)
            winner_loc = winners[dest]
            winning_move = current_pick(winner_loc)
            unit_type = order[0]

            # Take the support straight from the engine's legal orders. Supports for coast
            # moves are listed without the coast ('... S F CON - BUL'), so accept either form
            wm = winning_move.split()
            coastless_move = ' '.join(wm[:3] + [wm[3][:3]]) if len(wm) >= 4 else winning_move
            matching = [o for o in all_possible_orders.get(loc, [])
                        if ' S ' in o and (o.endswith(' S ' + winning_move) or o.endswith(' S ' + coastless_move))]

            if matching:
                final_orders[loc] = matching[0]
            else:
                # fall to #2, then #3, then hold
                next_idx = chosen_index[loc] + 1
                if next_idx < len(candidates):
                    final_orders[loc] = candidates[next_idx][1]
                else:
                    final_orders[loc] = f'{unit_type} {loc} H'

        #Everyone not involved in a collision just takes their #1 pick
        for loc in locations:
            if loc not in final_orders:
                order = current_pick(loc)
                if order:
                    final_orders[loc] = order
                else:
                    # No scored candidate: hold, using the unit's real type (A/F), not the location's first letter
                    unit_type = next((u[0] for u in self.game.get_units(self.power_name)
                                      if u.split()[1] == loc), 'A')
                    final_orders[loc] = f'{unit_type} {loc} H'

        return final_orders

    #This function counts how many enemy units are adjacent to a location, it;s then used as a danger/exposure signal for retreating
    def count_adjacent_enemies(self, graph, loc):
        if loc not in graph:
            return 0
        count = 0
        for neighbour in graph.neighbors(loc):
            for power_name in self.game.powers.keys():
                if power_name == self.power_name:
                    continue
                for unit in self.game.get_units(power_name):
                    if unit.split(' ')[1] == neighbour:
                        count += 1
        return count

    #This scores a unit for disbanding, we want to disband the most dangerous/exposed units, these score the highest and get disbanded first. Distance to enemy territory is treated as the tie breaker.
    def score_disband_location(self, loc, all_possible_orders, enemy_centres, weights):
        possible_orders = all_possible_orders.get(loc, [])
        _, disbands = self.classify_adjustment_orders(possible_orders)

        if not disbands:
            return None, -1

        disband_order = disbands[0]
        unit_type = disband_order[0]
        graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy

        danger_score = self.count_adjacent_enemies(graph, loc)
        tiebreak_score = self.distance_score(graph, loc, enemy_centres, weights)

        total_score = (danger_score * 10) + (weights['distance_cap'] - tiebreak_score)

        return disband_order, total_score

    #This function checks if any enemy unit has a legal move into this destination for this turn, it's a one turn look ahead check
    def could_enemy_contest(self, destination, all_possible_orders):
        return destination in self.enemy_reachable or destination[:3] in self.enemy_reachable

    #This then looks up which power controls a given unit string so that we can judge their friendliness
    def get_unit_owner(self, unit):
        for power_name in self.game.powers.keys():
            if unit in self.game.get_units(power_name):
                return power_name
        return None

    #-----
    #The brain
    #-----

    def get_game_stage(self):
        current_phase = self.game.get_current_phase()
        year = int(current_phase[1:5])

        if year <= 1903:
            return 'EARLY'
        elif year <= 1910:
            return 'MID'
        else:
            return 'LATE'

    # Movement turns without capturing the strategic target before System 2 gives up on it
    TARGET_PATIENCE = 6
    # System 2 variant flags (for testing which parts help)
    S2_PREFER_EMPTY = True        # prefer enemy centres with no enemy unit on them
    S2_PROGRESS_PATIENCE = True   # only count turns where we got no closer to the target
    S2_TARGET_PULL = True         # +3 score for moving onto the target
    S2_MULTI_TARGET = True    # one target per front instead of one for the whole army
    S2_MAX_FRONTS = 3         # most targets at once outside the endgame
    S2_FRONT_SPACING = 3      # targets on different fronts must be at least this many moves apart
    S2_ENDGAME = True         # endgame mode: target the centres that get us to 18 (truces end)
    ENDGAME_NEEDED = 4        # endgame starts when we need this many centres or fewer to win

    TECHNIQUES = {
        'supported_attacks': True,
        'collision_resolution': True,
        'lookahead': True,
        'strategic_target': True,
        'fall_capture_hold': True,
        'contested_support_only': True,
        'self_block_removal': True,
        'opponent_modelling': True,
        'progress_scoring': True,
        'support_reconciliation': True,
        'convoys': True,
    }

    STAGE_WEIGHTS = {
        'EARLY': {
            'distance_cap': 5,
            'support_bonus': 2,
            'contest_penalty': 3,
            'hold_baseline': 1.0,
            'lookahead_penalty': 2,
            'aggression_weight': 0.4,
            'pressure_weight': 0.4,
            'strength_weight': 0.2,
            'opponent_multiplier': 0.0,
        },
        'MID': {
            'distance_cap': 5,
            'support_bonus': 2,
            'contest_penalty': 3,
            'hold_baseline': 1.0,
            'lookahead_penalty': 2,
            'aggression_weight': 0.4,
            'pressure_weight': 0.4,
            'strength_weight': 0.2,
            'opponent_multiplier': 0.0,
        },
        'LATE': {
            'distance_cap': 5,
            'support_bonus': 2,
            'contest_penalty': 3,
            'hold_baseline': 1.0,
            'lookahead_penalty': 2,
            'aggression_weight': 0.4,
            'pressure_weight': 0.4,
            'strength_weight': 0.2,
            'opponent_multiplier': 0.0,
        },
    }

    def get_active_weights(self):
        stage = self.get_game_stage()
        return self.STAGE_WEIGHTS[stage]

        #Builds the set of every location currently occupied by an enemy unit, computed once per turn so is_contested is a fast lookup
    def build_enemy_occupied(self):
        occupied = set()
        for power_name in self.game.powers.keys():
            if power_name == self.power_name:
                continue
            for unit in self.game.get_units(power_name):
                loc = unit.split(' ')[1]
                occupied.add(loc)
                occupied.add(loc[:3])   # also the bare province, so 'BUL/EC' matches 'BUL'
        return occupied

    #Builds the set of every destination any enemy unit can move into this turn, computed once per turn so the lookahead check is a fast lookup
    def build_enemy_reachable(self, all_possible_orders):
        unit_owner = {}
        for power_name in self.game.powers.keys():
            if power_name == self.power_name:
                continue
            for unit in self.game.get_units(power_name):
                unit_owner[unit] = power_name

        reachable = set()
        for loc, orders in all_possible_orders.items():
            for order in orders:
                if ' - ' not in order:
                    continue
                unit = ' '.join(order.split(' ')[:2])
                if unit in unit_owner:
                    dest = self.get_move_destination(order)
                    reachable.add(dest)
                    reachable.add(dest[:3])   # bare province too, so coasts match
        return reachable

    def find_supportable_attacks(self, orderable_locations, all_possible_orders):
        candidates = []

        enemy_centres = set(self.get_enemy_centres())
        is_fall = self.game.get_current_phase().startswith('F')

        for attacker_loc in orderable_locations:
            # A unit on an uncaptured centre in Fall must stay to capture it.
            # It can still act as a supporter, since supporting doesn't move it.
            if self.TECHNIQUES['fall_capture_hold'] and is_fall and attacker_loc[:3] in enemy_centres:
                continue

            possible_orders = all_possible_orders.get(attacker_loc, [])
            moves, _, _ = self.classify_orders(possible_orders)

            for move in moves:
                destination = self.get_move_destination(move)

                if not self.is_contested(destination):
                    continue

                for supporter_loc in orderable_locations:
                    if supporter_loc == attacker_loc:
                        continue

                    # Use the engine's own support order for this supporter, so the
                    # supporter's unit type (A/F) is always correct
                    supporter_possible_orders = all_possible_orders.get(supporter_loc, [])
                    matching = [o for o in supporter_possible_orders if ' S ' in o and o.endswith(' S ' + move)]
                    if not matching:
                        continue
                    support_order = matching[0]
                    candidates.append((attacker_loc, supporter_loc, move, support_order))

        return candidates

    def select_committed_attacks(self, candidates):
        enemy_centres = set(self.get_enemy_centres())
        attackable_centres = set(
            self.get_attackable_centres(enemy_centres)
        )

        def priority(candidate):
            attacker_loc, supporter_loc, move, support_order = candidate
            destination = self.get_move_destination(move)
            if destination[:3] in self.current_targets:
                rank = 0
            elif destination in attackable_centres:
                rank = 1
            elif destination in enemy_centres:
                rank = 3
            else:
                rank = 2
            return (rank, attacker_loc, supporter_loc, move)

        candidates = sorted(candidates, key=priority)

        committed_units = set()
        locked_orders = {}

        for attacker_loc, supporter_loc, move, support_order in candidates:
            if attacker_loc in committed_units or supporter_loc in committed_units:
                continue

            locked_orders[attacker_loc] = move
            locked_orders[supporter_loc] = support_order
            committed_units.add(attacker_loc)
            committed_units.add(supporter_loc)

        return locked_orders

    @timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    def update_game(self, all_power_orders):

        was_movement_phase = self.game.get_current_phase().endswith("M")

        our_centres_before = set()
        our_units_before = set()
        if was_movement_phase:
            our_centres_before = set(
                centre[:3] for centre in self.game.get_centers(self.power_name)
            )

        
        
            our_units_before = set(
                unit.split()[1][:3]
                for unit in self.game.get_units(self.power_name)
            )

        unit_counts = {}

        if was_movement_phase:
            for power_name in self.opponent_model:
                unit_counts[power_name] = len(
                    self.game.get_units(power_name)
                )

        # Observe opponents before the game state changes
        if self.game.phase_type == 'M':
            self.update_opponent_aggression(all_power_orders)
            self.update_opponent_relationships(all_power_orders)

        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

        

        if was_movement_phase:
            

            our_centres_after = set(
                centre[:3] for centre in self.game.get_centers(self.power_name)
            )
            lost_centres = our_centres_before - our_centres_after

            self.update_opponent_hostility(
                all_power_orders,
                lost_centres,
                our_centres_before,
                our_units_before
            )
            

            self.update_opponent_activity(
                all_power_orders,
                unit_counts
            )
            

    def reconcile_supports(self, final_orders, top3_by_location, all_possible_orders):
        order_at = {loc[:3]: order for loc, order in final_orders.items()}

        def dest_of(order):
            return self.get_move_destination(order)[:3] if self.is_move_order(order) else None

        for loc, order in list(final_orders.items()):
            parts = order.split()
            if len(parts) < 5 or parts[2] != 'S':
                continue
            supported_unit = f'{parts[3]} {parts[4]}'
            supported_order = order_at.get(parts[4][:3])
            if supported_order is None or not supported_order.startswith(supported_unit):
                continue  # supporting a foreign unit: leave it alone

            supports_move = len(parts) >= 7 and parts[5] == '-'
            if supports_move and dest_of(supported_order) == parts[6][:3]:
                continue  # still valid
            if not supports_move and not self.is_move_order(supported_order):
                continue  # still valid

            legal = all_possible_orders.get(loc, [])
            unit = f'{parts[0]} {parts[1]}'
            replacement = None

            # 1) Support whatever the unit is actually doing now: its new move, or its hold
            if self.is_move_order(supported_order):
                candidate = f'{unit} S {supported_order}'
            else:
                candidate = f'{unit} S {supported_unit}'
            if candidate in legal:
                replacement = candidate

            # 2) Otherwise take this unit's own best move that doesn't clash with our other orders
            if replacement is None:
                taken = {dest_of(o) for o in final_orders.values() if dest_of(o)}
                staying = {l[:3] for l, o in final_orders.items() if not self.is_move_order(o) and l != loc}
                for _, alt in top3_by_location.get(loc, []):
                    d = dest_of(alt)
                    if d is None or (d not in taken and d not in staying):
                        replacement = alt
                        break

            # 3) Otherwise hold
            final_orders[loc] = replacement or f'{unit} H'
            order_at[loc[:3]] = final_orders[loc]

        return final_orders

        # Convoys: move armies that have no useful land move (e.g. England's island army) by sea.
    # Single-fleet convoys only: one of our fleets, in a sea zone touching both the army and
    # the destination, carries it. The army's VIA move and the fleet's convoy order are
    # locked together so later steps can't break the pair. Armies on a landmass with no enemy
    # centres walk to a launch port first; stranded armies with no convoy yet are remembered,
    # so a fleet can move into position (see score_movement_location).
    def plan_convoys(self, orderable_locations, all_possible_orders, locked_orders, enemy_centres):
        convoy_orders = {}
        self.stranded_armies = set()
        self.armies_to_port = set()
        used = set(locked_orders)
        unit_type_at = {u.split()[1]: u[0] for u in self.game.get_units(self.power_name)}
        own_provinces = {l[:3] for l in orderable_locations}
        enemy_held = {l[:3] for l in self.enemy_occupied}
        taken_dests = set()
        is_fall = self.game.get_current_phase().startswith('F')

        for army_loc in orderable_locations:
            if army_loc in used or unit_type_at.get(army_loc) != 'A':
                continue
            # An army sitting on an uncaptured centre in Fall must stay to capture it
            if is_fall and army_loc[:3] in enemy_centres:
                continue
            orders = all_possible_orders.get(army_loc, [])

            # Only stranded armies: no land move gets closer to an enemy centre
            cur_d = self.nearest_centre_distance(self.map_graph_army, army_loc, enemy_centres)
            land_progress = False
            for o in orders:
                if self.is_move_order(o) and not o.endswith(' VIA'):
                    new_d = self.nearest_centre_distance(self.map_graph_army, self.get_move_destination(o), enemy_centres)
                    if cur_d is not None and new_d is not None and new_d < cur_d:
                        land_progress = True
                        break
            if land_progress:
                continue

            # On a landmass with no enemy centres left (e.g. Britain): walk to a launch port first
            if cur_d is None and army_loc not in self.launch_ports:
                self.armies_to_port.add(army_loc)
                continue

            best = None   # (score, via_move, fleet_loc, convoy_order, dest)
            for via in [o for o in orders if o.endswith(' VIA')]:
                dest = self.get_move_destination(via)[:3]
                if dest in own_provinces or dest in enemy_held or dest in taken_dests:
                    continue
                # Score the landing spot: empty enemy centre first, otherwise closeness to enemy centres
                if dest in enemy_centres:
                    score = 10
                else:
                    d = self.nearest_centre_distance(self.map_graph_army, dest, enemy_centres)
                    score = 0 if d is None else max(0, 5 - d)
                if score <= 0:
                    continue

                # Find one of our free fleets next to both the army and the destination
                move_core = via[:-len(' VIA')]
                for fleet_loc in orderable_locations:
                    if fleet_loc in used or unit_type_at.get(fleet_loc) != 'F':
                        continue
                    navy = self.navy_distances.get(fleet_loc, {})
                    if navy.get(army_loc) != 1 or navy.get(dest) != 1:
                        continue
                    convoy = [o for o in all_possible_orders.get(fleet_loc, [])
                              if ' C ' in o and o.endswith(' C ' + move_core)]
                    if convoy and (best is None or score > best[0]):
                        best = (score, via, fleet_loc, convoy[0], dest)
                        break

            if best:
                _, via, fleet_loc, convoy_order, dest = best
                convoy_orders[army_loc] = via
                convoy_orders[fleet_loc] = convoy_order
                used.update([army_loc, fleet_loc])
                taken_dests.add(dest)
            else:
                # Stranded with no convoy available yet: a fleet should move into position
                self.stranded_armies.add(army_loc)

        return convoy_orders

    #This is called every turn and returns our full list of orders
    @timeout_decorator.timeout(1)
    def get_actions(self):

        '''Implement your agent here.'''

        weights = self.get_active_weights()

        # Per-turn caches - rebuilt every turn from the current board, never carried over
        self.opponent_threat_cache = {}
        self.enemy_occupied = self.build_enemy_occupied()

        own_info = self.get_own_units_and_locations()
        orderable_locations = own_info['orderable_locations']

        if not orderable_locations:
            return []
 
        all_possible_orders = self.game.get_all_possible_orders()
        self.enemy_reachable = self.build_enemy_reachable(all_possible_orders)
 
        if self.game.phase_type == 'R':
            own_centres = self.get_own_centres()
            power_orders = []
            for loc in orderable_locations:
                best_order = self.score_retreat_location(loc, all_possible_orders, own_centres, weights)
                if best_order:
                    power_orders.append(best_order)
            return power_orders

        if self.game.phase_type == 'A':
            enemy_centres = self.get_enemy_centres()
            required_builds = self.get_required_builds()
            required_disbands = self.get_required_disbands()

            power_orders = []

            if required_builds > 0:
                scored_builds = []
                for loc in orderable_locations:
                    build_order, score = self.score_build_location(loc, all_possible_orders, enemy_centres, weights)
                    if build_order:
                        scored_builds.append((score, build_order))
                scored_builds.sort(reverse=True, key=lambda x: x[0])
                power_orders = [order for score, order in scored_builds[:required_builds]]

            elif required_disbands > 0:
                scored_disbands = []
                for loc in orderable_locations:
                    disband_order, score = self.score_disband_location(loc, all_possible_orders, enemy_centres, weights)
                    if disband_order:
                        scored_disbands.append((score, disband_order))
                scored_disbands.sort(reverse=True, key=lambda x: x[0])
                power_orders = [order for score, order in scored_disbands[:required_disbands]]

            return power_orders
 
        enemy_centres = self.get_enemy_centres()
        if self.TECHNIQUES['strategic_target']:
            self.update_strategic_target(enemy_centres, weights)

        # Deliberate supported-attack pre-pass:

        # Deliberate supported-attack pre-pass: lock in units that can
        # guarantee-win a contested centre via a planned attack+support pair,
        # before normal per-unit scoring runs.
        if self.TECHNIQUES['supported_attacks']:
            attack_candidates = self.find_supportable_attacks(orderable_locations, all_possible_orders)
            locked_orders = self.select_committed_attacks(attack_candidates)
        else:
            locked_orders = {}

        # Convoys: lock stranded armies and their carrying fleet together, before normal scoring
        if self.TECHNIQUES['convoys']:
            locked_orders.update(self.plan_convoys(orderable_locations, all_possible_orders, locked_orders, enemy_centres))

        remaining_locations = [loc for loc in orderable_locations if loc not in locked_orders]

        top3_by_location = {}
        for loc in remaining_locations:
            top3_by_location[loc] = self.score_movement_location(
                loc, all_possible_orders, orderable_locations, enemy_centres, weights
            )

        if self.TECHNIQUES['collision_resolution']:
            final_orders_dict = self.resolve_collisions(top3_by_location, all_possible_orders)
        else:
            # every unit just takes its own top-scored order
            final_orders_dict = {loc: cands[0][1] for loc, cands in top3_by_location.items() if cands}
        final_orders_dict.update(locked_orders)
        if self.TECHNIQUES['self_block_removal']:
            final_orders_dict = self.remove_self_blocks(final_orders_dict, top3_by_location)

        if self.TECHNIQUES['support_reconciliation']:
            final_orders_dict = self.reconcile_supports(final_orders_dict, top3_by_location, all_possible_orders)

        power_orders = list(final_orders_dict.values())
        return power_orders
