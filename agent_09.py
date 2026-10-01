import timeout_decorator
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

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from agent_baselines.py first.
    
    You can add/override attributes and methods as needed.
    '''

    #---
    #Configuration
    #---

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
    DISTANCE_CAP = 5
    SUPPORT_BONUS = 2
    CONTEST_PENALTY = 3
    HOLD_BASELINE = 1.0
    LOOKAHEAD_PENALTY = 2
    OPPONENT_MULTIPLIER = 0.0
    AGGRESSION_WEIGHT = 0.4
    PRESSURE_WEIGHT = 0.4
    STRENGTH_WEIGHT = 0.2
    TARGET_PATIENCE = 6
    S2_PREFER_EMPTY = True
    S2_PROGRESS_PATIENCE = True
    S2_TARGET_PULL = True
    S2_MULTI_TARGET = True
    S2_MAX_FRONTS = 3
    S2_FRONT_SPACING = 3
    S2_ENDGAME = True
    ENDGAME_NEEDED = 4

    #---
    #Setup
    #---

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
        self.stranded_armies = set()

        self.army_distances = dict(nx.all_pairs_shortest_path_length(self.map_graph_army))
        self.navy_distances = dict(nx.all_pairs_shortest_path_length(self.map_graph_navy))

        components = [c for c in nx.connected_components(self.map_graph_army) if len(c) > 1]
        self.army_region = {node: i for i, comp in enumerate(components) for node in comp}
        self.bridge_seas = set()
        for sea in self.map_graph_navy.nodes:
            if self.game.map.area_type(sea) != 'WATER':
                continue
            regions = {self.army_region[c] for c in self.map_graph_navy.neighbors(sea) if c in self.army_region}
            if len(regions) >= 2:
                self.bridge_seas.add(sea)
        self.launch_ports = {c for sea in self.bridge_seas for c in self.map_graph_navy.neighbors(sea) if c in self.army_region}
        self.armies_to_port = set()

        '''Implement your agent here.'''

        self.opponent_model = {}

        for opponent in self.game.powers:
            if opponent != self.power_name:
                self.opponent_model[opponent] = {'aggression': 0.0, 'relationship_score': 0.0, 'activity': 0.0, 'activity_history': [], 'hostility': 0.0}

        self.current_target = None
        self.current_targets = {}
        self.last_target_distance = None

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

    #---
    #Game Engine Update
    #---

    @timeout_decorator.timeout(1)
    def update_game(self, all_power_orders):
        was_movement_phase = self.game.get_current_phase().endswith("M")

        our_centres_before = set()
        our_units_before = set()
        if was_movement_phase:
            our_centres_before = set(centre[:3] for centre in self.game.get_centers(self.power_name))

            our_units_before = set(unit.split()[1][:3] for unit in self.game.get_units(self.power_name))

        unit_counts = {}

        if was_movement_phase:
            for power_name in self.opponent_model:
                unit_counts[power_name] = len(self.game.get_units(power_name))

        if self.game.phase_type == 'M':
            self.update_opponent_aggression(all_power_orders)
            self.update_opponent_relationships(all_power_orders)

        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

        if was_movement_phase:
            our_centres_after = set(centre[:3] for centre in self.game.get_centers(self.power_name))
            lost_centres = our_centres_before - our_centres_after

            self.update_opponent_hostility(all_power_orders, lost_centres, our_centres_before, our_units_before)

            self.update_opponent_activity(all_power_orders, unit_counts)

    #---
    #Board Helpers
    #---

    def get_own_units_and_locations(self):
        return {'units': self.game.get_units(self.power_name), 'orderable_locations': self.game.get_orderable_locations(self.power_name)}

    def get_own_centres(self):
        return self.game.get_centers(self.power_name)

    def get_enemy_centres(self):
        enemy_centres = []
        for i in self.game.map.scs:
            if i not in self.game.get_centers(self.power_name):
                enemy_centres.append(i)
        return enemy_centres

    def classify_orders(self, possible_orders):
        moves = []
        holds = []
        supports = []
        for order in possible_orders:
            if ' S ' in order:
                supports.append(order)
            elif self.is_move_order(order) and not order.endswith(' VIA'):
                moves.append(order)
            elif order.endswith(' H'):
                holds.append(order)
        return moves, holds, supports

    def get_move_destination(self, order):
        words = order.split(' ')
        dash_index = words.index('-')
        return words[dash_index + 1]

    def is_move_order(self, order):
        parts = order.split()
        return len(parts) >= 4 and parts[2] == '-'

    def build_enemy_occupied(self):
        occupied = set()
        for power_name in self.game.powers.keys():
            if power_name == self.power_name:
                continue
            for unit in self.game.get_units(power_name):
                loc = unit.split(' ')[1]
                occupied.add(loc)
                occupied.add(loc[:3])
        return occupied

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
                    reachable.add(dest[:3])
        return reachable

    def is_contested(self, destination):
        return destination in self.enemy_occupied or destination[:3] in self.enemy_occupied

    def could_enemy_contest(self, destination, all_possible_orders):
        return destination in self.enemy_reachable or destination[:3] in self.enemy_reachable

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

    def nearest_centre_distance(self, graph, node, centres):
        all_lengths = self.army_distances if graph is self.map_graph_army else self.navy_distances
        lengths = all_lengths.get(node, {})
        dists = [lengths[c] for c in centres if c in lengths]
        return min(dists) if dists else None

    #---
    #Basic Technique: Greedy Heuristic Move Scoring
    #---

    def distance_score(self, graph, destination, enemy_centres):
        if destination not in graph:
            return 0

        all_lengths = self.army_distances if graph is self.map_graph_army else self.navy_distances
        lengths = all_lengths.get(destination, {})

        min_dist = 1000
        for centre in enemy_centres:
            dist = lengths.get(centre)
            if dist is not None and dist < min_dist:
                min_dist = dist

        if min_dist == 1000:
            return 0

        return max(0, self.DISTANCE_CAP - min_dist)

    def is_move_supportable(self, move_order, all_possible_orders, own_orderable_locations):
        for other_loc in own_orderable_locations:
            for candidate in all_possible_orders.get(other_loc, []):
                if ' S ' in candidate and move_order in candidate:
                    return True
        return False

    def score_order(self, is_hold, distance_score_val, is_supportable, is_contested_flag, opponent_threat=0.0):
        if is_hold:
            return self.HOLD_BASELINE

        score = distance_score_val

        if is_supportable and (is_contested_flag or not self.TECHNIQUES['contested_support_only']):
            score += self.SUPPORT_BONUS

        if is_contested_flag:
            score -= self.CONTEST_PENALTY

        score -= self.OPPONENT_MULTIPLIER * opponent_threat

        return score

    def score_movement_location(self, loc, all_possible_orders, own_orderable_locations, enemy_centres):
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

            dist_score = self.distance_score(graph, destination, attackable_centres)

            if self.TECHNIQUES['progress_scoring']:
                cur_d = self.nearest_centre_distance(graph, loc, attackable_centres)
                new_d = self.nearest_centre_distance(graph, destination, attackable_centres)
                if cur_d is not None and new_d is not None:
                    if new_d < cur_d:
                        dist_score = max(dist_score, 1.5)
                    elif destination[:3] not in attackable_centres and not self.is_support_position(graph, destination, attackable_centres):
                        dist_score = 0
            if (self.TECHNIQUES['convoys'] and unit_type == 'F' and destination in self.bridge_seas and any(self.navy_distances.get(destination, {}).get(a) == 1 for a in self.stranded_armies)):
                dist_score = max(dist_score, 2.0)

            if self.TECHNIQUES['convoys'] and unit_type == 'A' and loc in self.armies_to_port:
                lengths_now = self.army_distances.get(loc, {})
                lengths_new = self.army_distances.get(destination, {})
                cur_p = min((lengths_now[p] for p in self.launch_ports if p in lengths_now), default=None)
                new_p = min((lengths_new[p] for p in self.launch_ports if p in lengths_new), default=None)
                if cur_p is not None and new_p is not None and new_p < cur_p:
                    dist_score = max(dist_score, 1.5)
            if self.S2_TARGET_PULL and destination[:3] in self.current_targets:
                dist_score += 3

            supportable = self.is_move_supportable(move, all_possible_orders, own_orderable_locations)

            contested = self.is_contested(destination)

            opponent_threat = self.get_destination_threat(destination) if self.TECHNIQUES['opponent_modelling'] else 0.0

            total_score = self.score_order(False, dist_score, supportable, contested, opponent_threat)

            if self.TECHNIQUES['lookahead'] and not contested and not supportable:
                if self.could_enemy_contest(destination, all_possible_orders):
                    total_score -= self.LOOKAHEAD_PENALTY

            scored_candidates.append((total_score, move))

        is_fall = self.game.get_current_phase().startswith('F')
        on_uncaptured_centre = loc[:3] in enemy_centres

        for hold in holds:
            total_score = self.score_order(True, 0, False, False)
            if self.TECHNIQUES['fall_capture_hold'] and is_fall and on_uncaptured_centre:
                total_score = 100
            scored_candidates.append((total_score, hold))

        if not scored_candidates:
            return [(0, possible_orders[0])]

        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        return scored_candidates[:3]

    #---
    #Unit Coordination: Supported Attacks
    #---

    def find_supportable_attacks(self, orderable_locations, all_possible_orders):
        candidates = []

        enemy_centres = set(self.get_enemy_centres())
        is_fall = self.game.get_current_phase().startswith('F')

        for attacker_loc in orderable_locations:
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

                    supporter_possible_orders = all_possible_orders.get(supporter_loc, [])
                    matching = [o for o in supporter_possible_orders if ' S ' in o and o.endswith(' S ' + move)]
                    if not matching:
                        continue
                    support_order = matching[0]
                    candidates.append((attacker_loc, supporter_loc, move, support_order))

        return candidates

    def select_committed_attacks(self, candidates):
        enemy_centres = set(self.get_enemy_centres())
        attackable_centres = set(self.get_attackable_centres(enemy_centres))

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

    #---
    #Unit Coordination: Collision Resolution
    #---

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
                return self.get_move_destination(order)[:3]
            return None

        locations = list(top3_by_location.keys())

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
                best_loc = max(locs_here, key=lambda l: top3_by_location[l][chosen_index[l]][0])
                winners[dest] = best_loc
                for l in locs_here:
                    if l != best_loc:
                        collided_locs.add(l)

        for loc in collided_locs:
            candidates = top3_by_location[loc]
            order = candidates[chosen_index[loc]][1]
            dest = get_destination_if_move(order)
            winner_loc = winners[dest]
            winning_move = current_pick(winner_loc)
            unit_type = order[0]

            wm = winning_move.split()
            coastless_move = ' '.join(wm[:3] + [wm[3][:3]]) if len(wm) >= 4 else winning_move
            matching = [o for o in all_possible_orders.get(loc, []) if ' S ' in o and (o.endswith(' S ' + winning_move) or o.endswith(' S ' + coastless_move))]

            if matching:
                final_orders[loc] = matching[0]
            else:
                next_idx = chosen_index[loc] + 1
                if next_idx < len(candidates):
                    final_orders[loc] = candidates[next_idx][1]
                else:
                    final_orders[loc] = f'{unit_type} {loc} H'

        for loc in locations:
            if loc not in final_orders:
                order = current_pick(loc)
                if order:
                    final_orders[loc] = order
                else:
                    unit_type = next((u[0] for u in self.game.get_units(self.power_name) if u.split()[1] == loc), 'A')
                    final_orders[loc] = f'{unit_type} {loc} H'

        return final_orders

    #---
    #Unit Coordination: Order Refinement
    #---

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
                swap = moving.get(dest) == loc[:3] and loc[:3] > dest
                duplicate = loc in duplicate_losers

                if not (into_staying_unit or swap or duplicate):
                    continue

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
                continue

            supports_move = len(parts) >= 7 and parts[5] == '-'
            if supports_move and dest_of(supported_order) == parts[6][:3]:
                continue
            if not supports_move and not self.is_move_order(supported_order):
                continue

            legal = all_possible_orders.get(loc, [])
            unit = f'{parts[0]} {parts[1]}'
            replacement = None

            if self.is_move_order(supported_order):
                candidate = f'{unit} S {supported_order}'
            else:
                candidate = f'{unit} S {supported_unit}'
            if candidate in legal:
                replacement = candidate

            if replacement is None:
                taken = {dest_of(o) for o in final_orders.values() if dest_of(o)}
                staying = {l[:3] for l, o in final_orders.items() if not self.is_move_order(o) and l != loc}
                for _, alt in top3_by_location.get(loc, []):
                    d = dest_of(alt)
                    if d is None or (d not in taken and d not in staying):
                        replacement = alt
                        break

            final_orders[loc] = replacement or f'{unit} H'
            order_at[loc[:3]] = final_orders[loc]

        return final_orders

    #---
    #Unit Coordination: Convoys
    #---

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
            if is_fall and army_loc[:3] in enemy_centres:
                continue
            orders = all_possible_orders.get(army_loc, [])

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

            if cur_d is None and army_loc not in self.launch_ports:
                self.armies_to_port.add(army_loc)
                continue

            best = None
            for via in [o for o in orders if o.endswith(' VIA')]:
                dest = self.get_move_destination(via)[:3]
                if dest in own_provinces or dest in enemy_held or dest in taken_dests:
                    continue
                if dest in enemy_centres:
                    score = 10
                else:
                    d = self.nearest_centre_distance(self.map_graph_army, dest, enemy_centres)
                    score = 0 if d is None else max(0, 5 - d)
                if score <= 0:
                    continue

                move_core = via[:-len(' VIA')]
                for fleet_loc in orderable_locations:
                    if fleet_loc in used or unit_type_at.get(fleet_loc) != 'F':
                        continue
                    navy = self.navy_distances.get(fleet_loc, {})
                    if navy.get(army_loc) != 1 or navy.get(dest) != 1:
                        continue
                    convoy = [o for o in all_possible_orders.get(fleet_loc, []) if ' C ' in o and o.endswith(' C ' + move_core)]
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
                self.stranded_armies.add(army_loc)

        return convoy_orders

    #---
    #Strategic Movement: Progress Scoring
    #---

    def is_support_position(self, graph, destination, enemy_centres):
        if destination not in graph:
            return False
        occupied = {loc[:3] for loc in self.enemy_occupied}
        for neighbour in graph.neighbors(destination):
            n = neighbour[:3]
            if n in enemy_centres and n in occupied:
                return True
        return False

    #---
    #Strategic Movement: Strategic Targeting (System 2)
    #---

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

    def target_score(self, centre, occupied):
        d = self.distance_to_target(centre)
        if d is None:
            return None
        score = max(0, self.DISTANCE_CAP - d)
        if self.S2_PREFER_EMPTY and centre not in occupied:
            score += 2
        score -= self.OPPONENT_MULTIPLIER * self.get_destination_threat(centre)
        return score

    def centre_gap(self, a, b):
        d = self.army_distances.get(a, {}).get(b)
        if d is None:
            d = self.navy_distances.get(a, {}).get(b)
        return d if d is not None else 99

    def update_strategic_target(self, enemy_centres):
        needed = 18 - len(self.game.get_centers(self.power_name))
        endgame = self.S2_ENDGAME and 0 < needed <= self.ENDGAME_NEEDED

        pool = set(enemy_centres) if endgame else set(self.get_attackable_centres(enemy_centres))
        occupied = {l[:3] for l in self.enemy_occupied}

        for t in list(self.current_targets):
            if t not in pool:
                del self.current_targets[t]

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

        if not self.S2_MULTI_TARGET:
            wanted = 1
        elif endgame:
            wanted = needed
        else:
            wanted = max(1, min(self.S2_MAX_FRONTS, len(self.game.get_units(self.power_name)) // 3))

        while len(self.current_targets) < wanted:
            best, best_score = None, None
            for c in sorted(pool):
                if c in self.current_targets or c in dropped:
                    continue
                if not endgame and any(self.centre_gap(c, t) < self.S2_FRONT_SPACING for t in self.current_targets):
                    continue
                s = self.target_score(c, occupied)
                if s is not None and (best_score is None or s > best_score):
                    best, best_score = c, s
            if best is None:
                break
            self.current_targets[best] = {'stale': 0, 'last_d': self.distance_to_target(best)}

        def dist_or_far(t):
            d = self.distance_to_target(t)
            return 99 if d is None else d
        self.current_target = min(self.current_targets, key=dist_or_far) if self.current_targets else None

    #---
    #Opponent Modelling
    #---

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
            data["activity"] = (sum(data["activity_history"]) / len(data["activity_history"]))

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

            data["hostility"] = (0.7 * old_hostility + hostile_actions)

    def get_adaptive_opponent_threat(self, power_name):
        data = self.opponent_model.get(power_name)

        if data is None:
            return 0.0

        activity = data["activity"]
        hostility = data["hostility"]

        base_threat = 1.0
        return activity * (base_threat + hostility)

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

        is_static = (len(activity_history) >= 4 and all(activity == 0.0 for activity in activity_history))
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

    def update_opponent_aggression(self, all_power_orders):
        for opponent, orders in all_power_orders.items():
            if opponent == self.power_name:
                continue

            if opponent not in self.opponent_model:
                self.opponent_model[opponent] = {'aggression': 0.0, 'relationship_score': 0.0}

            if not orders:
                continue

            attack_count = 0

            own_units = self.game.get_units(self.power_name)
            own_centres = self.game.get_centers(self.power_name)

            own_locations = {unit.split()[1] for unit in own_units}

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

            attack_ratio = attack_count / len(orders)

            old_aggression = self.opponent_model[opponent]['aggression']

            new_aggression = (0.7 * old_aggression + 0.3 * attack_ratio)

            self.opponent_model[opponent]['aggression'] = new_aggression

    def update_opponent_relationships(self, all_power_orders):
        own_units = self.game.get_units(self.power_name)
        own_centres = self.game.get_centers(self.power_name)

        own_locations = {unit.split()[1] for unit in own_units}

        own_territory = own_locations.union(own_centres)

        for opponent, orders in all_power_orders.items():
            if opponent == self.power_name:
                continue

            if opponent not in self.opponent_model:
                self.opponent_model[opponent] = {'aggression': 0.0, 'relationship_score': 0.0}

            if not orders:
                continue

            hostile_count = 0
            friendly_count = 0

            for order in orders:
                parts = order.split()

                if '-' in parts and 'S' not in parts:
                    dash_index = parts.index('-')

                    if dash_index + 1 < len(parts):
                        destination = parts[dash_index + 1]

                        if destination in own_territory:
                            hostile_count += 1

                elif 'S' in parts:
                    support_index = parts.index('S')

                    if support_index + 2 < len(parts):
                        supported_unit = ' '.join(parts[support_index + 1:support_index + 3])

                        if supported_unit in own_units:
                            friendly_count += 1

            behaviour_score = (friendly_count - hostile_count) / len(orders)

            old_score = self.opponent_model[opponent].get('relationship_score', 0.0)

            new_score = (0.7 * old_score + 0.3 * behaviour_score)

            self.opponent_model[opponent]['relationship_score'] = new_score

    def get_all_opponent_relationships(self):
        relationships = {}

        for opponent in self.game.powers:
            if opponent == self.power_name:
                continue

            relationship_score = self.opponent_model.get(opponent, {}).get('relationship_score', 0.0)

            scaled_score = 5 - (5 * relationship_score)

            relationships[opponent] = round(scaled_score, 2)

        return relationships

    def get_opponent_pressure(self, opponent):
        enemy_units = self.game.get_units(opponent)

        own_units = self.game.get_units(self.power_name)
        own_centres = self.game.get_centers(self.power_name)

        own_locations = {unit.split()[1] for unit in own_units}

        own_territory = own_locations.union(own_centres)

        if not own_territory:
            return 0.0

        threatened_locations = set()

        for unit in enemy_units:
            parts = unit.split()

            unit_type = parts[0]
            location = parts[1]

            graph = (self.map_graph_army if unit_type == 'A' else self.map_graph_navy)

            if location not in graph:
                continue

            for neighbour in graph.neighbors(location):
                if neighbour in own_territory:
                    threatened_locations.add(neighbour)

        pressure = (len(threatened_locations) / len(own_territory))

        return pressure

    def get_opponent_strength(self, opponent):
        opponent_units = len(self.game.get_units(opponent))

        total_units = 0

        for power in self.game.powers:
            total_units += len(self.game.get_units(power))

        if total_units == 0:
            return 0.0

        return opponent_units / total_units

    def get_opponent_threat(self, opponent):
        if self.TECHNIQUES['opponent_modelling']:
            return self.get_adaptive_opponent_threat(opponent)

        aggression = self.opponent_model.get(opponent, {}).get('aggression', 0.0)

        pressure = self.get_opponent_pressure(opponent)

        strength = self.get_opponent_strength(opponent)

        threat = (self.AGGRESSION_WEIGHT * aggression + self.PRESSURE_WEIGHT * pressure + self.STRENGTH_WEIGHT * strength)

        return threat

    def get_destination_threat(self, destination):
        highest_threat = 0.0

        for opponent in self.game.powers:
            if opponent == self.power_name:
                continue

            enemy_units = self.game.get_units(opponent)

            for unit in enemy_units:
                parts = unit.split()

                unit_type = parts[0]
                location = parts[1]

                graph = (self.map_graph_army if unit_type == 'A' else self.map_graph_navy)

                if location not in graph:
                    continue

                if (location == destination or destination in graph.neighbors(location)):
                    if opponent not in self.opponent_threat_cache:
                        self.opponent_threat_cache[opponent] = self.get_opponent_threat(opponent)
                    threat = self.opponent_threat_cache[opponent]

                    highest_threat = max(highest_threat, threat)

                    break

        return highest_threat

    #---
    #Retreats
    #---

    def get_retreat_destination(self, order):
        words = order.split(' ')
        r_index = words.index('R')
        return words[r_index + 1]

    def classify_retreat_orders(self, possible_orders):
        retreats = []
        disbands = []
        for order in possible_orders:
            if ' R ' in order:
                retreats.append(order)
            elif order.endswith(' D'):
                disbands.append(order)
        return retreats, disbands

    def score_retreat_location(self, loc, all_possible_orders, own_centres):
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

            safety_score = self.distance_score(graph, destination, own_centres)
            contested = self.is_contested(destination)

            total_score = safety_score - (self.CONTEST_PENALTY if contested else 0)
            scored_candidates.append((total_score, retreat))

        max_score = max(s for s, o in scored_candidates)
        top_candidates = [o for s, o in scored_candidates if s == max_score]
        return sorted(top_candidates)[0]

    #---
    #Builds and Disbands
    #---

    def get_required_builds(self):
        own_centres = self.get_own_centres()
        own_units = self.game.get_units(self.power_name)
        return max(0, len(own_centres) - len(own_units))

    def get_required_disbands(self):
        own_centres = self.get_own_centres()
        own_units = self.game.get_units(self.power_name)
        return max(0, len(own_units) - len(own_centres))

    def classify_adjustment_orders(self, possible_orders):
        builds = []
        disbands = []
        for order in possible_orders:
            if order.endswith(' B'):
                builds.append(order)
            elif order.endswith(' D'):
                disbands.append(order)
        return builds, disbands

    def score_build_location(self, loc, all_possible_orders, enemy_centres):
        possible_orders = all_possible_orders.get(loc, [])
        builds, disbands = self.classify_adjustment_orders(possible_orders)

        if not builds:
            return None, -1

        best_order = None
        best_score = -1
        for build in builds:
            unit_type = build[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            score = self.distance_score(graph, loc, enemy_centres)
            if score > best_score:
                best_score = score
                best_order = build

        return best_order, best_score

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

    def score_disband_location(self, loc, all_possible_orders, enemy_centres):
        possible_orders = all_possible_orders.get(loc, [])
        _, disbands = self.classify_adjustment_orders(possible_orders)

        if not disbands:
            return None, -1

        disband_order = disbands[0]
        unit_type = disband_order[0]
        graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy

        danger_score = self.count_adjacent_enemies(graph, loc)
        tiebreak_score = self.distance_score(graph, loc, enemy_centres)

        total_score = (danger_score * 10) + (self.DISTANCE_CAP - tiebreak_score)

        return disband_order, total_score

    #---
    #Turn Orders
    #---

    @timeout_decorator.timeout(1)
    def get_actions(self):
        '''Implement your agent here.'''


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
                best_order = self.score_retreat_location(loc, all_possible_orders, own_centres)
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
                    build_order, score = self.score_build_location(loc, all_possible_orders, enemy_centres)
                    if build_order:
                        scored_builds.append((score, build_order))
                scored_builds.sort(reverse=True, key=lambda x: x[0])
                power_orders = [order for score, order in scored_builds[:required_builds]]

            elif required_disbands > 0:
                scored_disbands = []
                for loc in orderable_locations:
                    disband_order, score = self.score_disband_location(loc, all_possible_orders, enemy_centres)
                    if disband_order:
                        scored_disbands.append((score, disband_order))
                scored_disbands.sort(reverse=True, key=lambda x: x[0])
                power_orders = [order for score, order in scored_disbands[:required_disbands]]

            return power_orders

        enemy_centres = self.get_enemy_centres()
        if self.TECHNIQUES['strategic_target']:
            self.update_strategic_target(enemy_centres)

        if self.TECHNIQUES['supported_attacks']:
            attack_candidates = self.find_supportable_attacks(orderable_locations, all_possible_orders)
            locked_orders = self.select_committed_attacks(attack_candidates)
        else:
            locked_orders = {}

        if self.TECHNIQUES['convoys']:
            locked_orders.update(self.plan_convoys(orderable_locations, all_possible_orders, locked_orders, enemy_centres))

        remaining_locations = [loc for loc in orderable_locations if loc not in locked_orders]

        top3_by_location = {}
        for loc in remaining_locations:
            top3_by_location[loc] = self.score_movement_location(loc, all_possible_orders, orderable_locations, enemy_centres)

        if self.TECHNIQUES['collision_resolution']:
            final_orders_dict = self.resolve_collisions(top3_by_location, all_possible_orders)
        else:
            final_orders_dict = {loc: cands[0][1] for loc, cands in top3_by_location.items() if cands}
        final_orders_dict.update(locked_orders)
        if self.TECHNIQUES['self_block_removal']:
            final_orders_dict = self.remove_self_blocks(final_orders_dict, top3_by_location)

        if self.TECHNIQUES['support_reconciliation']:
            final_orders_dict = self.reconcile_supports(final_orders_dict, top3_by_location, all_possible_orders)

        power_orders = list(final_orders_dict.values())
        return power_orders