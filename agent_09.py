import timeout_decorator
import networkx as nx
from agent_baselines import Agent

class StudentAgent(Agent):
    #---
    #Configuration
    #---

    #This dict stores the on/off switch for every technique so each one can be tested on its own
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

    #This sets how far away a center can be and still be added to the units score to decide its move 
    DISTANCE_CAP = 5

    #This is the support bonus that is given to a units move when another unit nearby can support its own move
    SUPPORT_BONUS = 2

    #This is the penalty we inflict on a units move for when it wwants to move into an area an enemy is already on
    CONTEST_PENALTY = 3

    #This is the score every hold order gets
    HOLD_BASELINE = 1.0

    #This is the penalty for moving into an empty province that an enemy could also move into this turn
    LOOKAHEAD_PENALTY = 2

    #This is how much an opponent's threat lowers a move's score
    OPPONENT_MULTIPLIER = 0.0

    #This decides how many turns we keep a target without getting closer to it before giving up on it
    TARGET_PATIENCE = 6

    #This switch makes the agent brain prefer centres with no enemy unit on them cause free centres are 100% upside
    AGENT_BRAIN_PREFER_EMPTY = True

    #This switch will resets a target's patience whenever we get closer to it
    AGENT_BRAIN_PROGRESS_PATIENCE = True

    #This switch gives extra score to moves that go straight into a target
    AGENT_BRAIN_TARGET_PULL = True

    #This switch lets the agent brain keep more than one target at a time
    AGENT_BRAIN_MULTI_TARGET = True

    #This is the most targets the agent brain will keep at once
    AGENT_BRAIN_MAX_FRONTS = 3

    #This is how many provinces apart two targets have to be
    AGENT_BRAIN_FRONT_SPACING = 3

    #This switch turns on the endgame push when we are close to winning
    AGENT_BRAIN_ENDGAME = True

    #This is how many centres away from winning we have to be for the endgame push to start aka the backstabbing
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

        #This dict stores what we learn about each opponent
        self.opponent_model = {}

    #This function runs once at the start of a game and works out everything about the map that we will need later
    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_map_graphs()

        #This set stores the armies that have no land route to an enemy centre
        self.stranded_armies = set()

        #These two dicts store the shortest distance between every pair of provinces, one for armies and one for fleets
        self.army_distances = dict(nx.all_pairs_shortest_path_length(self.map_graph_army))
        self.navy_distances = dict(nx.all_pairs_shortest_path_length(self.map_graph_navy))

        #This list stores each separate landmass that an army can walk around
        components = [component for component in nx.connected_components(self.map_graph_army) if len(component) > 1]

        #This dict stores which landmass each province belongs to
        self.army_region = {node: region_index for region_index, component in enumerate(components) for node in component}

        #This set stores the seas that touch two or more landmasses, which are the seas a convoy can cross
        self.bridge_seas = set()

        #This loop checks every sea and keeps the ones that join different landmasses
        for sea in self.map_graph_navy.nodes:
            if self.game.map.area_type(sea) != 'WATER':
                continue
            regions = {self.army_region[coast] for coast in self.map_graph_navy.neighbors(sea) if coast in self.army_region}
            if len(regions) >= 2:
                self.bridge_seas.add(sea)

        #This set stores the coastal provinces next to a bridge sea which is where an army can be picked up
        self.launch_ports = {coast for sea in self.bridge_seas for coast in self.map_graph_navy.neighbors(sea) if coast in self.army_region}

        #This set stores the armies that need to walk to a launch port
        self.armies_to_port = set()

        #This dict stores the activity and hostility we track for each opponent
        self.opponent_model = {}

        for opponent in self.game.powers:
            if opponent != self.power_name:
                self.opponent_model[opponent] = {'activity': 0.0, 'activity_history': [], 'hostility': 0.0}

        #This dict stores the centres the agent brain is currently going after
        self.current_targets = {}

    #This function builds two graphs of the map, one for where armies can move and one for where fleets can move
    def build_map_graphs(self):
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        #This list stores every location on the map
        locations = list(self.game.map.loc_type.keys())

        #This loop adds each location to the army graph, the fleet graph, or both
        for location in locations:
            if self.game.map.loc_type[location] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(location.upper())
            if self.game.map.loc_type[location] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(location.upper())

        locations = [location.upper() for location in locations]

        #This loop joins two locations whenever a unit can move between them
        for location in locations:
            for neighbour in locations:
                if self.game.map.abuts('A', location, '-', neighbour):
                    self.map_graph_army.add_edge(location, neighbour)
                if self.game.map.abuts('F', location, '-', neighbour):
                    self.map_graph_navy.add_edge(location, neighbour)

    #---
    #Game Engine Update
    #---

    #This function applies everyone's orders to our copy of the game and records how each opponent behaved
    @timeout_decorator.timeout(1)
    def update_game(self, all_power_orders):
        #This remembers whether the phase being processed is a movement phase
        was_movement_phase = self.game.get_current_phase().endswith("M")

        #These two sets store our centres and unit positions from before the orders are processed
        our_centres_before = set()
        our_units_before = set()
        if was_movement_phase:
            our_centres_before = set(centre[:3] for centre in self.game.get_centers(self.power_name))
            our_units_before = set(unit.split()[1][:3] for unit in self.game.get_units(self.power_name))

        #This dict stores how many units each opponent had before the orders are processed
        unit_counts = {}

        if was_movement_phase:
            for power_name in self.opponent_model:
                unit_counts[power_name] = len(self.game.get_units(power_name))

        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

        #This part works out which centres we lost and updates what we know about each opponent
        if was_movement_phase:
            our_centres_after = set(centre[:3] for centre in self.game.get_centers(self.power_name))
            lost_centres = our_centres_before - our_centres_after

            self.update_opponent_hostility(all_power_orders, lost_centres, our_centres_before, our_units_before)

            self.update_opponent_activity(all_power_orders, unit_counts)

    #---
    #Board Helpers
    #---

    #This function gets our units and the locations we can give orders to
    def get_own_units_and_locations(self):
        return {'units': self.game.get_units(self.power_name), 'orderable_locations': self.game.get_orderable_locations(self.power_name)}

    #This function gets the supply centres we own
    def get_own_centres(self):
        return self.game.get_centers(self.power_name)

    #This function finds every supply centre we do not own
    def get_enemy_centres(self):
        #This is the list that stores the centres we do not own
        enemy_centres = []
        for centre in self.game.map.scs:
            if centre not in self.game.get_centers(self.power_name):
                enemy_centres.append(centre)

        #This then returns the list of centres we dont own
        return enemy_centres

    #This function sorts a unit's legal orders into moves and holds and leaves out supports and convoy moves
    def classify_orders(self, possible_orders):
        #These two lists store the move orders and the hold orders
        moves = []
        holds = []
        for order in possible_orders:
            if ' S ' in order:
                continue
            elif self.is_move_order(order) and not order.endswith(' VIA'):
                moves.append(order)
            elif order.endswith(' H'):
                holds.append(order)

        #This then returns the moves and the holds
        return moves, holds

    #This function pulls the destination out of a move order
    def get_move_destination(self, order):
        words = order.split(' ')
        dash_index = words.index('-')

        #This then returns the word after the dash, which is the destination
        return words[dash_index + 1]

    #This function checks whether an order is a move
    def is_move_order(self, order):
        parts = order.split()

        #This then returns True if the order has a dash in the move position
        return len(parts) >= 4 and parts[2] == '-'

    #This function finds every province an enemy unit is standing on
    def build_enemy_occupied(self):
        #This set stores the enemy occupied locations, with and without their coast
        occupied = set()
        for power_name in self.game.powers.keys():
            if power_name == self.power_name:
                continue
            for unit in self.game.get_units(power_name):
                location = unit.split(' ')[1]
                occupied.add(location)
                occupied.add(location[:3])

        #This then returns the set of enemy occupied locations
        return occupied

    #This function finds every province an enemy unit could move into this turn
    def build_enemy_reachable(self, all_possible_orders):
        #This dict stores which power owns each enemy unit
        unit_owner = {}
        for power_name in self.game.powers.keys():
            if power_name == self.power_name:
                continue
            for unit in self.game.get_units(power_name):
                unit_owner[unit] = power_name

        #This set stores every location an enemy move could end in, with and without its coast
        reachable = set()
        for location, orders in all_possible_orders.items():
            for order in orders:
                if ' - ' not in order:
                    continue
                unit = ' '.join(order.split(' ')[:2])
                if unit in unit_owner:
                    destination = self.get_move_destination(order)
                    reachable.add(destination)
                    reachable.add(destination[:3])

        #This then returns the set of locations an enemy could reach
        return reachable

    #This function checks whether an enemy unit is standing on a destination
    def is_contested(self, destination):
        return destination in self.enemy_occupied or destination[:3] in self.enemy_occupied

    #This function checks whether an enemy unit could move into a destination this turn
    def could_enemy_contest(self, destination):
        return destination in self.enemy_reachable or destination[:3] in self.enemy_reachable

    #This function finds which power owns a supply centre
    def get_centre_owner(self, location):
        location = location[:3]

        for power_name in self.game.powers:
            centres = self.game.get_centers(power_name)

            for centre in centres:
                if centre[:3] == location:
                    #This returns the power that owns the centre
                    return power_name

        #This then returns nothing if no power owns the centre
        return None

    #This function works out how far a location is from the closest centre in a list
    def nearest_centre_distance(self, graph, node, centres):
        all_lengths = self.army_distances if graph is self.map_graph_army else self.navy_distances
        lengths = all_lengths.get(node, {})

        #This list stores the distance to every centre that can be reached
        distances = [lengths[centre] for centre in centres if centre in lengths]

        #This then returns the shortest distance, or nothing if no centre can be reached
        return min(distances) if distances else None

    #---
    #Basic Technique: Greedy Heuristic Move Scoring
    #---

    #This function scores a destination by how close it is to the nearest enemy centre
    def distance_score(self, graph, destination, enemy_centres):
        if destination not in graph:
            #This returns zero if this type of unit cannot be on the destination
            return 0

        all_lengths = self.army_distances if graph is self.map_graph_army else self.navy_distances
        lengths = all_lengths.get(destination, {})

        min_distance = 1000

        #This loop finds the distance to the closest enemy centre
        for centre in enemy_centres:
            distance = lengths.get(centre)
            if distance is not None and distance < min_distance:
                min_distance = distance

        if min_distance == 1000:
            #This returns zero if no enemy centre can be reached
            return 0

        #This then returns a higher score the closer the destination is, and never less than zero
        return max(0, self.DISTANCE_CAP - min_distance)

    #This function checks whether another of our units has a legal order that supports a move
    def is_move_supportable(self, move_order, all_possible_orders, own_orderable_locations):
        for other_location in own_orderable_locations:
            for candidate in all_possible_orders.get(other_location, []):
                if ' S ' in candidate and move_order in candidate:
                    #This returns True as soon as one supporting order is found
                    return True

        #This then returns False if no unit can support the move
        return False

    #This function turns what we know about one order into a single score
    def score_order(self, is_hold, distance_score_val, is_supportable, is_contested_flag, opponent_threat=0.0):
        if is_hold:
            #This returns the flat hold score if the order is a hold
            return self.HOLD_BASELINE

        score = distance_score_val

        #This adds the support bonus, and only when the destination is contested if that switch is on
        if is_supportable and (is_contested_flag or not self.TECHNIQUES['contested_support_only']):
            score += self.SUPPORT_BONUS

        #This takes off the penalty for moving onto an enemy unit
        if is_contested_flag:
            score -= self.CONTEST_PENALTY

        #This takes off the opponent threat, scaled by the multiplier
        score -= self.OPPONENT_MULTIPLIER * opponent_threat

        #This then returns the final score for the order
        return score

    #This function scores every legal order for one unit and keeps its three best
    def score_movement_location(self, location, all_possible_orders, own_orderable_locations, enemy_centres):
        possible_orders = all_possible_orders.get(location, [])
        if not possible_orders:
            #This returns nothing if the unit has no legal orders
            return []

        moves, holds = self.classify_orders(possible_orders)

        #This list stores each order together with its score
        scored_candidates = []

        #This list stores the enemy centres we are allowed to attack, after any truces are taken out
        attackable_centres = self.get_attackable_centres(enemy_centres)

        #This loop scores every move the unit could make
        for move in moves:
            unit_type = move[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            destination = self.get_move_destination(move)

            #This is the basic score, based on how close the move gets to an enemy centre
            move_distance_score = self.distance_score(graph, destination, attackable_centres)

            #This part is progress scoring, a move that gets closer always beats holding and a move that does not get closer scores nothing unless it sets up a support
            if self.TECHNIQUES['progress_scoring']:
                current_distance = self.nearest_centre_distance(graph, location, attackable_centres)
                new_distance = self.nearest_centre_distance(graph, destination, attackable_centres)
                if current_distance is not None and new_distance is not None:
                    if new_distance < current_distance:
                        move_distance_score = max(move_distance_score, 1.5)
                    elif destination[:3] not in attackable_centres and not self.is_support_position(graph, destination, attackable_centres):
                        move_distance_score = 0

            #This part sends a fleet into a sea where it can convoy a stranded army
            if (self.TECHNIQUES['convoys'] and unit_type == 'F' and destination in self.bridge_seas and any(self.navy_distances.get(destination, {}).get(army) == 1 for army in self.stranded_armies)):
                move_distance_score = max(move_distance_score, 2.0)

            #This part walks an army towards a port where a fleet can pick it up
            if self.TECHNIQUES['convoys'] and unit_type == 'A' and location in self.armies_to_port:
                lengths_now = self.army_distances.get(location, {})
                lengths_new = self.army_distances.get(destination, {})
                current_port_distance = min((lengths_now[port] for port in self.launch_ports if port in lengths_now), default=None)
                new_port_distance = min((lengths_new[port] for port in self.launch_ports if port in lengths_new), default=None)
                if current_port_distance is not None and new_port_distance is not None and new_port_distance < current_port_distance:
                    move_distance_score = max(move_distance_score, 1.5)

            #This part gives extra score to a move that goes straight into one of our targets
            if self.AGENT_BRAIN_TARGET_PULL and destination[:3] in self.current_targets:
                move_distance_score += 3

            #This checks whether one of our other units could support the move
            supportable = self.is_move_supportable(move, all_possible_orders, own_orderable_locations)

            #This checks whether an enemy unit is standing on the destination
            contested = self.is_contested(destination)

            #This looks up how threatening the opponents around the destination are
            opponent_threat = self.get_destination_threat(destination) if self.TECHNIQUES['opponent_modelling'] else 0.0

            #This combines everything into the score for the move
            total_score = self.score_order(False, move_distance_score, supportable, contested, opponent_threat)

            #This part is the lookahead, it takes off score if an enemy could move into the same empty province
            if self.TECHNIQUES['lookahead'] and not contested and not supportable:
                if self.could_enemy_contest(destination):
                    total_score -= self.LOOKAHEAD_PENALTY

            scored_candidates.append((total_score, move))

        #This part is the Fall capture hold, a unit standing on a centre we have not captured yet stays there in Fall so that we take it
        is_fall = self.game.get_current_phase().startswith('F')
        on_uncaptured_centre = location[:3] in enemy_centres

        #This loop scores the hold order
        for hold in holds:
            total_score = self.score_order(True, 0, False, False)
            if self.TECHNIQUES['fall_capture_hold'] and is_fall and on_uncaptured_centre:
                total_score = 100
            scored_candidates.append((total_score, hold))

        if not scored_candidates:
            #This returns the first legal order if nothing could be scored
            return [(0, possible_orders[0])]

        scored_candidates.sort(reverse=True, key=lambda scored_candidate: scored_candidate[0])

        #This then returns the three best orders, highest score first
        return scored_candidates[:3]

    #---
    #Unit Coordination: Supported Attacks
    #---

    #This function finds every pair of our units where one can attack an enemy held province and the other can support it
    def find_supportable_attacks(self, orderable_locations, all_possible_orders):
        #This list stores each possible attack as the attacker, the supporter, the move and the support order
        candidates = []

        enemy_centres = set(self.get_enemy_centres())
        is_fall = self.game.get_current_phase().startswith('F')

        #This loop tries every one of our units as the attacker
        for attacker_location in orderable_locations:
            #This skips a unit that is holding a centre for the Fall capture
            if self.TECHNIQUES['fall_capture_hold'] and is_fall and attacker_location[:3] in enemy_centres:
                continue

            possible_orders = all_possible_orders.get(attacker_location, [])
            moves, _ = self.classify_orders(possible_orders)

            for move in moves:
                destination = self.get_move_destination(move)

                #This skips moves into empty provinces because they do not need support
                if not self.is_contested(destination):
                    continue

                #This loop looks for another unit that can legally support the move
                for supporter_location in orderable_locations:
                    if supporter_location == attacker_location:
                        continue

                    supporter_possible_orders = all_possible_orders.get(supporter_location, [])

                    #This list stores the supporter's legal orders that support exactly this move
                    matching = [order for order in supporter_possible_orders if ' S ' in order and order.endswith(' S ' + move)]
                    if not matching:
                        continue
                    support_order = matching[0]
                    candidates.append((attacker_location, supporter_location, move, support_order))

        #This then returns every attack and supporter pair that was found
        return candidates

    #This function picks which supported attacks to lock in, making sure each unit is only used once
    def select_committed_attacks(self, candidates):
        enemy_centres = set(self.get_enemy_centres())
        attackable_centres = set(self.get_attackable_centres(enemy_centres))

        #This function ranks an attack so that attacks on our targets come first
        def priority(candidate):
            attacker_location, supporter_location, move, support_order = candidate
            destination = self.get_move_destination(move)
            if destination[:3] in self.current_targets:
                rank = 0
            elif destination in attackable_centres:
                rank = 1
            elif destination in enemy_centres:
                rank = 3
            else:
                rank = 2

            #This then returns the rank and the order details, so ties are always broken the same way
            return (rank, attacker_location, supporter_location, move)

        candidates = sorted(candidates, key=priority)

        #This set stores the units that already have a locked order
        committed_units = set()

        #This dict stores the locked order for each committed unit
        locked_orders = {}

        #This loop locks each attack in ranked order and skips any that needs a unit already in use
        for attacker_location, supporter_location, move, support_order in candidates:
            if attacker_location in committed_units or supporter_location in committed_units:
                continue

            locked_orders[attacker_location] = move
            locked_orders[supporter_location] = support_order
            committed_units.add(attacker_location)
            committed_units.add(supporter_location)

        #This then returns the locked attack and support orders
        return locked_orders

    #---
    #Unit Coordination: Collision Resolution
    #---

    #This function makes sure two of our units do not move into the same province, the loser supports the winner or takes its next best order
    def resolve_collisions(self, top3_by_location, all_possible_orders):
        #This dict stores the final order for each location
        final_orders = {}

        #This dict stores which of its three options each location is currently using
        chosen_index = {location: 0 for location in top3_by_location}

        #This function gets the order a location is currently using
        def current_pick(location):
            candidates = top3_by_location[location]
            index = chosen_index[location]
            if index >= len(candidates):
                #This returns nothing if the location has run out of options
                return None

            #This then returns the order at the chosen position
            return candidates[index][1]

        #This function gets the province a move ends in, with any coast removed
        def get_destination_if_move(order):
            if self.is_move_order(order):
                #This returns the destination if the order is a move
                return self.get_move_destination(order)[:3]

            #This then returns nothing if the order is not a move
            return None

        locations = list(top3_by_location.keys())

        #This dict stores which of our locations are trying to move into each destination
        destination_map = {}
        for location in locations:
            order = current_pick(location)
            destination = get_destination_if_move(order) if order else None
            if destination:
                destination_map.setdefault(destination, []).append(location)

        #This set stores the locations that lost a clash
        collided_locations = set()

        #This dict stores the winning location for each contested destination
        winners = {}

        #This loop finds each clash and lets the highest scoring move win
        for destination, locations_here in destination_map.items():
            if len(locations_here) > 1:
                best_location = max(locations_here, key=lambda contender: top3_by_location[contender][chosen_index[contender]][0])
                winners[destination] = best_location
                for clashing_location in locations_here:
                    if clashing_location != best_location:
                        collided_locations.add(clashing_location)

        #This loop gives every losing unit a new order
        for location in collided_locations:
            candidates = top3_by_location[location]
            order = candidates[chosen_index[location]][1]
            destination = get_destination_if_move(order)
            winner_location = winners[destination]
            winning_move = current_pick(winner_location)
            unit_type = order[0]

            winning_move_parts = winning_move.split()

            #This is the winning move with its coast removed, because support orders are written without the coast
            coastless_move = ' '.join(winning_move_parts[:3] + [winning_move_parts[3][:3]]) if len(winning_move_parts) >= 4 else winning_move

            #This list stores the loser's legal orders that support the winning move
            matching = [order for order in all_possible_orders.get(location, []) if ' S ' in order and (order.endswith(' S ' + winning_move) or order.endswith(' S ' + coastless_move))]

            #This makes the loser support the winner if it can, otherwise take its next option, otherwise hold
            if matching:
                final_orders[location] = matching[0]
            else:
                next_index = chosen_index[location] + 1
                if next_index < len(candidates):
                    final_orders[location] = candidates[next_index][1]
                else:
                    final_orders[location] = f'{unit_type} {location} H'

        #This loop gives every other location its first choice
        for location in locations:
            if location not in final_orders:
                order = current_pick(location)
                if order:
                    final_orders[location] = order
                else:
                    unit_type = next((unit[0] for unit in self.game.get_units(self.power_name) if unit.split()[1] == location), 'A')
                    final_orders[location] = f'{unit_type} {location} H'

        #This then returns the final order for every location
        return final_orders

    #---
    #Unit Coordination: Order Refinement
    #---

    #This function makes sure none of our units blocks another, by fixing moves into our own staying units, swaps and duplicate destinations
    def remove_self_blocks(self, final_orders, top3_by_location):
        #This dict stores which fallback option each location has reached
        chosen_index = {location: 0 for location in final_orders}

        #This function gets the province a move ends in, or nothing if the order is not a move
        def destination_of(order):
            return self.get_move_destination(order)[:3] if self.is_move_order(order) else None

        #This loop repeats the checks until no order changes, up to five times
        for _ in range(5):
            changed = False

            #This set stores the provinces where our unit is not moving
            staying = {location[:3] for location, order in final_orders.items() if not self.is_move_order(order)}

            #This dict stores where each of our moving units is going
            moving = {location[:3]: destination_of(order) for location, order in final_orders.items() if self.is_move_order(order)}

            #This function ranks a move so that a supported move wins a tie, and then the higher score
            def move_priority(mover_location):
                current_order = final_orders[mover_location]
                is_supported = any(' S ' in order and order.endswith(current_order) for order in final_orders.values())
                scored_options = top3_by_location.get(mover_location)
                score = next((option_score for option_score, option_order in scored_options if option_order == current_order), 1000) if scored_options else 1000

                #This then returns whether the move is supported, its score and its location
                return (is_supported, score, mover_location)

            #This dict stores which of our units are moving into each destination
            movers_by_destination = {}
            for mover_location, mover_order in final_orders.items():
                mover_destination = destination_of(mover_order)
                if mover_destination is not None:
                    movers_by_destination.setdefault(mover_destination, []).append(mover_location)

            #This set stores the units that lose when two of ours want the same destination
            duplicate_losers = set()
            for mover_destination, locations_here in movers_by_destination.items():
                if len(locations_here) > 1:
                    keep = max(locations_here, key=move_priority)
                    duplicate_losers.update(other_location for other_location in locations_here if other_location != keep)

            #This loop replaces every blocked move with the next option that is not blocked
            for location, order in list(final_orders.items()):
                destination = destination_of(order)
                if destination is None:
                    continue

                into_staying_unit = destination in staying
                swap = moving.get(destination) == location[:3] and location[:3] > destination
                duplicate = location in duplicate_losers

                #This skips the move if nothing is wrong with it
                if not (into_staying_unit or swap or duplicate):
                    continue

                #This is the fallback, the unit holds if none of its other options work
                replacement = f'{order[0]} {location} H'
                candidates = top3_by_location.get(location, [])
                index = chosen_index[location] + 1
                while index < len(candidates):
                    alternative = candidates[index][1]
                    alternative_destination = destination_of(alternative)
                    if alternative_destination is None or alternative_destination not in staying:
                        replacement = alternative
                        break
                    index += 1
                chosen_index[location] = index

                final_orders[location] = replacement
                changed = True

            #This stops repeating once a full pass changes nothing
            if not changed:
                break

        #This then returns the orders with the self blocks removed
        return final_orders

    #This function makes sure every support order matches what the supported unit is actually doing
    def reconcile_supports(self, final_orders, top3_by_location, all_possible_orders):
        #This dict stores the final order of the unit in each province
        order_at = {location[:3]: order for location, order in final_orders.items()}

        #This function gets the province a move ends in, or nothing if the order is not a move
        def destination_of(order):
            return self.get_move_destination(order)[:3] if self.is_move_order(order) else None

        #This loop checks every support order we are giving to one of our own units
        for location, order in list(final_orders.items()):
            parts = order.split()

            #This skips any order that is not a support
            if len(parts) < 5 or parts[2] != 'S':
                continue
            supported_unit = f'{parts[3]} {parts[4]}'
            supported_order = order_at.get(parts[4][:3])

            #This skips supports of units that are not ours
            if supported_order is None or not supported_order.startswith(supported_unit):
                continue

            supports_move = len(parts) >= 7 and parts[5] == '-'

            #This skips the support if the supported unit really is making that move
            if supports_move and destination_of(supported_order) == parts[6][:3]:
                continue

            #This skips the support if the supported unit really is holding
            if not supports_move and not self.is_move_order(supported_order):
                continue

            legal = all_possible_orders.get(location, [])
            unit = f'{parts[0]} {parts[1]}'
            replacement = None

            #This tries to point the support at what the unit is really doing
            if self.is_move_order(supported_order):
                candidate = f'{unit} S {supported_order}'
            else:
                candidate = f'{unit} S {supported_unit}'
            if candidate in legal:
                replacement = candidate

            #This part picks the next best order that does not clash, if the support could not be fixed
            if replacement is None:
                taken = {destination_of(order) for order in final_orders.values() if destination_of(order)}
                staying = {other_location[:3] for other_location, other_order in final_orders.items() if not self.is_move_order(other_order) and other_location != location}
                for _, alternative in top3_by_location.get(location, []):
                    alternative_destination = destination_of(alternative)
                    if alternative_destination is None or (alternative_destination not in taken and alternative_destination not in staying):
                        replacement = alternative
                        break

            #This uses the replacement, or holds if nothing else works
            final_orders[location] = replacement or f'{unit} H'
            order_at[location[:3]] = final_orders[location]

        #This then returns the orders with every support matching a real move or hold
        return final_orders

    #---
    #Unit Coordination: Convoys
    #---

    #This function finds armies with no land route to an enemy centre and gives them a convoy if one of our fleets is in place
    def plan_convoys(self, orderable_locations, all_possible_orders, locked_orders, enemy_centres):
        #This dict stores the convoy orders for the armies and fleets involved
        convoy_orders = {}
        self.stranded_armies = set()
        self.armies_to_port = set()

        #This set stores the units that already have an order
        used = set(locked_orders)

        #This dict stores whether the unit at each location is an army or a fleet
        unit_type_at = {unit.split()[1]: unit[0] for unit in self.game.get_units(self.power_name)}

        #These two sets store the provinces we hold and the provinces enemy units hold
        own_provinces = {location[:3] for location in orderable_locations}
        enemy_held = {location[:3] for location in self.enemy_occupied}

        #This set stores the landing provinces already claimed by another convoy
        taken_destinations = set()
        is_fall = self.game.get_current_phase().startswith('F')

        #This loop checks every army that does not have an order yet
        for army_location in orderable_locations:
            if army_location in used or unit_type_at.get(army_location) != 'A':
                continue

            #This skips an army that is holding a centre for the Fall capture
            if is_fall and army_location[:3] in enemy_centres:
                continue
            orders = all_possible_orders.get(army_location, [])

            #This part checks whether the army can get closer to an enemy centre by land
            current_distance = self.nearest_centre_distance(self.map_graph_army, army_location, enemy_centres)
            land_progress = False
            for order in orders:
                if self.is_move_order(order) and not order.endswith(' VIA'):
                    new_distance = self.nearest_centre_distance(self.map_graph_army, self.get_move_destination(order), enemy_centres)
                    if current_distance is not None and new_distance is not None and new_distance < current_distance:
                        land_progress = True
                        break

            #This skips the army if it can make progress by land
            if land_progress:
                continue

            #This marks the army to walk to a port if its landmass has no enemy centres and it is not at a port yet
            if current_distance is None and army_location not in self.launch_ports:
                self.armies_to_port.add(army_location)
                continue

            #This part scores every convoy the army could take and keeps the best one
            best = None
            for via in [order for order in orders if order.endswith(' VIA')]:
                destination = self.get_move_destination(via)[:3]

                #This skips a landing on our own unit, on an enemy unit, or on a province another convoy is using
                if destination in own_provinces or destination in enemy_held or destination in taken_destinations:
                    continue

                #This scores the landing, an enemy centre is best and otherwise closer to one is better
                if destination in enemy_centres:
                    score = 10
                else:
                    landing_distance = self.nearest_centre_distance(self.map_graph_army, destination, enemy_centres)
                    score = 0 if landing_distance is None else max(0, 5 - landing_distance)
                if score <= 0:
                    continue

                move_core = via[:-len(' VIA')]

                #This loop looks for a free fleet that is next to both the army and the landing province
                for fleet_location in orderable_locations:
                    if fleet_location in used or unit_type_at.get(fleet_location) != 'F':
                        continue
                    navy = self.navy_distances.get(fleet_location, {})
                    if navy.get(army_location) != 1 or navy.get(destination) != 1:
                        continue

                    #This list stores the fleet's legal orders that convoy exactly this move
                    convoy = [order for order in all_possible_orders.get(fleet_location, []) if ' C ' in order and order.endswith(' C ' + move_core)]
                    if convoy and (best is None or score > best[0]):
                        best = (score, via, fleet_location, convoy[0], destination)
                        break

            #This locks in the army's move and the fleet's convoy order, or records the army as stranded if no convoy exists
            if best:
                _, via, fleet_location, convoy_order, destination = best
                convoy_orders[army_location] = via
                convoy_orders[fleet_location] = convoy_order
                used.update([army_location, fleet_location])
                taken_destinations.add(destination)
            else:
                self.stranded_armies.add(army_location)

        #This then returns the convoy orders to lock in
        return convoy_orders

    #---
    #Strategic Movement: Progress Scoring
    #---

    #This function checks whether a destination is next to an enemy centre that an enemy unit is standing on
    def is_support_position(self, graph, destination, enemy_centres):
        if destination not in graph:
            #This returns False if this type of unit cannot be on the destination
            return False

        #This set stores the provinces enemy units are standing on
        occupied = {location[:3] for location in self.enemy_occupied}
        for neighbour in graph.neighbors(destination):
            neighbour_province = neighbour[:3]
            if neighbour_province in enemy_centres and neighbour_province in occupied:
                #This returns True as soon as one occupied enemy centre is next to the destination
                return True

        #This then returns False if the destination cannot help an attack
        return False

    #---
    #Strategic Movement: Strategic Targeting (Agent Brain)
    #---

    #This function works out how far our closest unit is from a target
    def distance_to_target(self, target):
        if target is None:
            #This returns nothing if there is no target
            return None
        best = None

        #This loop finds the smallest distance from any of our units to the target
        for unit in self.game.get_units(self.power_name):
            parts = unit.replace('*', '').split()
            distances = self.army_distances if parts[0] == 'A' else self.navy_distances
            unit_distance = distances.get(parts[1], {}).get(target)
            if unit_distance is not None and (best is None or unit_distance < best):
                best = unit_distance

        #This then returns the distance of our closest unit
        return best

    #This function scores a centre as a possible target, closer and empty centres score higher
    def target_score(self, centre, occupied):
        distance = self.distance_to_target(centre)
        if distance is None:
            #This returns nothing if none of our units can reach the centre
            return None
        score = max(0, self.DISTANCE_CAP - distance)

        #This adds a bonus if no enemy unit is standing on the centre
        if self.AGENT_BRAIN_PREFER_EMPTY and centre not in occupied:
            score += 2

        #This takes off the opponent threat around the centre, scaled by the multiplier
        score -= self.OPPONENT_MULTIPLIER * self.get_destination_threat(centre)

        #This then returns the target score for the centre
        return score

    #This function works out how far apart two centres are, by land first and then by sea
    def centre_gap(self, first_centre, second_centre):
        distance = self.army_distances.get(first_centre, {}).get(second_centre)
        if distance is None:
            distance = self.navy_distances.get(first_centre, {}).get(second_centre)

        #This then returns the distance, or a large number if the centres are not connected
        return distance if distance is not None else 99

    #This function is the agent brain, it keeps a list of target centres across turns, drops the ones we are not getting closer to and picks new ones
    def update_strategic_target(self, enemy_centres):
        #This is how many more centres we need to win
        needed = 18 - len(self.game.get_centers(self.power_name))

        #This is True when we are close enough to winning for the endgame push
        endgame = self.AGENT_BRAIN_ENDGAME and 0 < needed <= self.ENDGAME_NEEDED

        #This set stores the centres we can choose targets from, every enemy centre in the endgame and otherwise only the attackable ones
        pool = set(enemy_centres) if endgame else set(self.get_attackable_centres(enemy_centres))

        #This set stores the provinces enemy units are standing on
        occupied = {location[:3] for location in self.enemy_occupied}

        #This loop removes targets we have captured or are no longer allowed to attack
        for target in list(self.current_targets):
            if target not in pool:
                del self.current_targets[target]

        #This set stores the targets we gave up on this turn, so they are not picked again straight away
        dropped = set()

        #This loop updates the patience of each target and drops any we have stopped getting closer to
        for target, info in list(self.current_targets.items()):
            distance = self.distance_to_target(target)
            if self.AGENT_BRAIN_PROGRESS_PATIENCE and distance is not None and info['last_distance'] is not None and distance < info['last_distance']:
                info['stale'] = 0
            else:
                info['stale'] += 1
            info['last_distance'] = distance
            if info['stale'] > self.TARGET_PATIENCE:
                del self.current_targets[target]
                dropped.add(target)

        #This part works out how many targets we want, one for every three units normally and exactly the number we need in the endgame
        if not self.AGENT_BRAIN_MULTI_TARGET:
            wanted = 1
        elif endgame:
            wanted = needed
        else:
            wanted = max(1, min(self.AGENT_BRAIN_MAX_FRONTS, len(self.game.get_units(self.power_name)) // 3))

        #This loop adds the best scoring centre as a new target until we have enough
        while len(self.current_targets) < wanted:
            best, best_score = None, None
            for centre in sorted(pool):
                if centre in self.current_targets or centre in dropped:
                    continue

                #This skips a centre that is too close to a target we already have
                if not endgame and any(self.centre_gap(centre, chosen_target) < self.AGENT_BRAIN_FRONT_SPACING for chosen_target in self.current_targets):
                    continue
                score = self.target_score(centre, occupied)
                if score is not None and (best_score is None or score > best_score):
                    best, best_score = centre, score
            if best is None:
                break

            #This stores the new target with fresh patience and its current distance
            self.current_targets[best] = {'stale': 0, 'last_distance': self.distance_to_target(best)}

    #---
    #Opponent Modelling
    #---

    #This function records how much of each opponent's army moved this turn and keeps an average over the last four turns
    def update_opponent_activity(self, all_power_orders, unit_counts):
        #This loop works out the activity of each opponent
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

            #This stores the turn's activity, keeps only the last four turns and averages them
            data["activity_history"].append(turn_activity)

            data["activity_history"] = data["activity_history"][-4:]
            data["activity"] = (sum(data["activity_history"]) / len(data["activity_history"]))

    #This function records how hostile each opponent was towards us this turn
    def update_opponent_hostility(self, all_power_orders, lost_centres, our_centres_before, our_units_before):
        #This set stores our centres and unit positions from before the turn
        our_locations_before = our_centres_before | our_units_before

        #This set stores every province next to one of ours
        nearby_locations_before = set()

        for location in our_locations_before:
            adjacent_locations = self.game.map.abut_list(location)

            for adjacent in adjacent_locations:
                nearby_locations_before.add(adjacent[:3])

        #This loop adds up the hostile actions of each opponent
        for power_name, data in self.opponent_model.items():
            hostile_actions = 0.0
            orders = all_power_orders.get(power_name, [])

            for order in orders:
                parts = order.split()

                #This counts a move into one of our provinces as fully hostile and a move next to us as partly hostile
                if len(parts) >= 4 and parts[2] == "-":
                    destination = parts[3][:3]

                    if destination in our_locations_before:
                        hostile_actions += 1.0

                    elif destination in nearby_locations_before:
                        hostile_actions += 0.25

                #This counts a support for a move into or next to our provinces in the same way
                elif len(parts) >= 5 and parts[2] == "S":
                    if "-" in parts:
                        destination = parts[-1][:3]

                        if destination in our_locations_before:
                            hostile_actions += 1.0

                        elif destination in nearby_locations_before:
                            hostile_actions += 0.25

            #This loop counts each centre the opponent took from us
            for centre in lost_centres:
                for unit in self.game.get_units(power_name):
                    unit_location = unit.split()[1][:3]

                    if unit_location == centre:
                        hostile_actions += 1.0
                        break

            #This fades the old hostility and adds this turn's hostile actions
            old_hostility = data["hostility"]

            data["hostility"] = (0.7 * old_hostility + hostile_actions)

    #This function turns an opponent's activity and hostility into one threat value
    def get_adaptive_opponent_threat(self, power_name):
        data = self.opponent_model.get(power_name)

        if data is None:
            #This returns zero if we have no record of the power
            return 0.0

        activity = data["activity"]
        hostility = data["hostility"]

        base_threat = 1.0

        #This then returns the threat, which is higher for opponents that are both active and hostile
        return activity * (base_threat + hostility)

    #This function decides whether we treat a power as being in a truce with us
    def is_truce_power(self, power_name):
        if power_name == self.power_name:
            #This returns False for our own power
            return False
        data = self.opponent_model.get(power_name)

        if data is None:
            #This returns False if we have no record of the power
            return False

        try:
            year = int(self.game.get_current_phase()[1:5])

            if year > 1912:
                #This returns False late in the game, when all truces end
                return False
        except (ValueError, TypeError):
            pass

        activity_history = data["activity_history"]
        if len(activity_history) < 4:
            #This returns False until we have watched the power for four turns
            return False

        #This checks whether the power has not moved at all
        is_static = (len(activity_history) >= 4 and all(activity == 0.0 for activity in activity_history))
        if is_static:
            #This returns False for a power that never moves, because it is not choosing to leave us alone
            return False

        #This then returns True if the power has shown almost no hostility towards us
        return data["hostility"] <= 0.15

    #This function takes the centres of truce powers out of the list of centres we can attack
    def get_attackable_centres(self, enemy_centres):
        if not self.TECHNIQUES['opponent_modelling']:
            #This returns every enemy centre if opponent modelling is switched off
            return enemy_centres

        #These two lists store the centres we can attack and the centres owned by truce powers
        preferred_centres = []
        truce_centres = []

        for centre in enemy_centres:
            owner = self.get_centre_owner(centre)

            if owner is not None and self.is_truce_power(owner):
                truce_centres.append(centre)
            else:
                preferred_centres.append(centre)

        if preferred_centres:
            #This returns the centres we can attack if there are any
            return preferred_centres

        #This then returns the truce centres if there is nothing else left to attack
        return truce_centres

    #This function gets the threat value of one opponent
    def get_opponent_threat(self, opponent):
        if not self.TECHNIQUES['opponent_modelling']:
            #This returns zero if opponent modelling is switched off
            return 0.0

        #This then returns the threat from the opponent model
        return self.get_adaptive_opponent_threat(opponent)

    #This function finds the highest threat among the opponents that have a unit on or next to a destination
    def get_destination_threat(self, destination):
        highest_threat = 0.0

        #This loop checks every opponent's units
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

                #This uses the opponent's threat if one of its units is on or next to the destination, and stores it so it is only worked out once a turn
                if (location == destination or destination in graph.neighbors(location)):
                    if opponent not in self.opponent_threat_cache:
                        self.opponent_threat_cache[opponent] = self.get_opponent_threat(opponent)
                    threat = self.opponent_threat_cache[opponent]

                    highest_threat = max(highest_threat, threat)

                    break

        #This then returns the highest threat that was found
        return highest_threat

    #---
    #Retreats
    #---

    #This function pulls the destination out of a retreat order
    def get_retreat_destination(self, order):
        words = order.split(' ')
        r_index = words.index('R')

        #This then returns the word after the R, which is the destination
        return words[r_index + 1]

    #This function sorts a dislodged unit's orders into retreats and disbands
    def classify_retreat_orders(self, possible_orders):
        #These two lists store the retreat orders and the disband orders
        retreats = []
        disbands = []
        for order in possible_orders:
            if ' R ' in order:
                retreats.append(order)
            elif order.endswith(' D'):
                disbands.append(order)

        #This then returns the retreats and the disbands
        return retreats, disbands

    #This function picks the best retreat for a dislodged unit, which is the one closest to our own centres
    def score_retreat_location(self, location, all_possible_orders, own_centres):
        possible_orders = all_possible_orders.get(location, [])
        if not possible_orders:
            #This returns nothing if the unit has no legal orders
            return None

        retreats, disbands = self.classify_retreat_orders(possible_orders)

        if not retreats:
            #This returns a disband if the unit has nowhere to retreat to
            return disbands[0] if disbands else possible_orders[0]

        #This list stores each retreat together with its score
        scored_candidates = []

        #This loop scores each retreat by how close it is to our own centres, with a penalty if an enemy unit is there
        for retreat in retreats:
            unit_type = retreat[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            destination = self.get_retreat_destination(retreat)

            safety_score = self.distance_score(graph, destination, own_centres)
            contested = self.is_contested(destination)

            total_score = safety_score - (self.CONTEST_PENALTY if contested else 0)
            scored_candidates.append((total_score, retreat))

        max_score = max(score for score, order in scored_candidates)

        #This list stores the retreats that share the best score
        top_candidates = [order for score, order in scored_candidates if score == max_score]

        #This then returns the best retreat, using alphabetical order to break ties
        return sorted(top_candidates)[0]

    #---
    #Builds and Disbands
    #---

    #This function works out how many units we are allowed to build
    def get_required_builds(self):
        own_centres = self.get_own_centres()
        own_units = self.game.get_units(self.power_name)

        #This then returns how many more centres we own than units
        return max(0, len(own_centres) - len(own_units))

    #This function works out how many units we have to remove
    def get_required_disbands(self):
        own_centres = self.get_own_centres()
        own_units = self.game.get_units(self.power_name)

        #This then returns how many more units we have than centres
        return max(0, len(own_units) - len(own_centres))

    #This function sorts adjustment orders into builds and disbands
    def classify_adjustment_orders(self, possible_orders):
        #These two lists store the build orders and the disband orders
        builds = []
        disbands = []
        for order in possible_orders:
            if order.endswith(' B'):
                builds.append(order)
            elif order.endswith(' D'):
                disbands.append(order)

        #This then returns the builds and the disbands
        return builds, disbands

    #This function picks the best unit to build at a home centre, based on how close it would be to enemy centres
    def score_build_location(self, location, all_possible_orders, enemy_centres):
        possible_orders = all_possible_orders.get(location, [])
        builds, disbands = self.classify_adjustment_orders(possible_orders)

        if not builds:
            #This returns nothing if we cannot build here
            return None, -1

        best_order = None
        best_score = -1

        #This loop scores each unit type we could build and keeps the best
        for build in builds:
            unit_type = build[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            score = self.distance_score(graph, location, enemy_centres)
            if score > best_score:
                best_score = score
                best_order = build

        #This then returns the best build order and its score
        return best_order, best_score

    #This function counts the enemy units next to a location
    def count_adjacent_enemies(self, graph, location):
        if location not in graph:
            #This returns zero if this type of unit cannot be on the location
            return 0
        count = 0

        #This loop counts every enemy unit standing on a neighbouring province
        for neighbour in graph.neighbors(location):
            for power_name in self.game.powers.keys():
                if power_name == self.power_name:
                    continue
                for unit in self.game.get_units(power_name):
                    if unit.split(' ')[1] == neighbour:
                        count += 1

        #This then returns the number of enemy units next to the location
        return count

    #This function scores how good a unit is to remove, units with more enemies around them are removed first
    def score_disband_location(self, location, all_possible_orders, enemy_centres):
        possible_orders = all_possible_orders.get(location, [])
        _, disbands = self.classify_adjustment_orders(possible_orders)

        if not disbands:
            #This returns nothing if the unit cannot be disbanded
            return None, -1

        disband_order = disbands[0]
        unit_type = disband_order[0]
        graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy

        #This is how many enemy units are next to the unit
        danger_score = self.count_adjacent_enemies(graph, location)

        #This is used to break ties, based on how close the unit is to enemy centres
        tiebreak_score = self.distance_score(graph, location, enemy_centres)

        total_score = (danger_score * 10) + (self.DISTANCE_CAP - tiebreak_score)

        #This then returns the disband order and its score
        return disband_order, total_score

    #---
    #Turn Orders
    #---

    #This function is called every phase and returns our orders, it runs each technique in order for movement phases and uses the simple scorers for retreats and builds
    @timeout_decorator.timeout(1)
    def get_actions(self):
        '''Implement your agent here.'''

        #This dict stores each opponent's threat so it is only worked out once a turn
        self.opponent_threat_cache = {}

        #This set stores every province an enemy unit is standing on
        self.enemy_occupied = self.build_enemy_occupied()

        own_info = self.get_own_units_and_locations()
        orderable_locations = own_info['orderable_locations']

        if not orderable_locations:
            #This returns no orders if we have no units to order
            return []

        #This dict stores every legal order for every location on the board
        all_possible_orders = self.game.get_all_possible_orders()

        #This set stores every province an enemy could move into this turn
        self.enemy_reachable = self.build_enemy_reachable(all_possible_orders)

        #This part handles the retreat phase, each dislodged unit takes its best retreat
        if self.game.phase_type == 'R':
            own_centres = self.get_own_centres()
            power_orders = []
            for location in orderable_locations:
                best_order = self.score_retreat_location(location, all_possible_orders, own_centres)
                if best_order:
                    power_orders.append(best_order)

            #This then returns the retreat orders
            return power_orders

        #This part handles the adjustment phase, where we build or disband units
        if self.game.phase_type == 'A':
            enemy_centres = self.get_enemy_centres()
            required_builds = self.get_required_builds()
            required_disbands = self.get_required_disbands()

            power_orders = []

            #This builds at the best scoring home centres, or disbands the units with the highest disband scores, whichever the game asks for
            if required_builds > 0:
                scored_builds = []
                for location in orderable_locations:
                    build_order, score = self.score_build_location(location, all_possible_orders, enemy_centres)
                    if build_order:
                        scored_builds.append((score, build_order))
                scored_builds.sort(reverse=True, key=lambda scored_build: scored_build[0])
                power_orders = [order for score, order in scored_builds[:required_builds]]

            elif required_disbands > 0:
                scored_disbands = []
                for location in orderable_locations:
                    disband_order, score = self.score_disband_location(location, all_possible_orders, enemy_centres)
                    if disband_order:
                        scored_disbands.append((score, disband_order))
                scored_disbands.sort(reverse=True, key=lambda scored_disband: scored_disband[0])
                power_orders = [order for score, order in scored_disbands[:required_disbands]]

            #This then returns the build or disband orders
            return power_orders

        #This part handles the movement phase, starting with the list of centres we do not own
        enemy_centres = self.get_enemy_centres()

        #This step lets the agent brain update its targets, if it is switched on
        if self.TECHNIQUES['strategic_target']:
            self.update_strategic_target(enemy_centres)

        #This step finds and locks in the supported attacks, if they are switched on
        if self.TECHNIQUES['supported_attacks']:
            attack_candidates = self.find_supportable_attacks(orderable_locations, all_possible_orders)
            locked_orders = self.select_committed_attacks(attack_candidates)
        else:
            locked_orders = {}

        #This step locks in any convoys, if they are switched on
        if self.TECHNIQUES['convoys']:
            locked_orders.update(self.plan_convoys(orderable_locations, all_possible_orders, locked_orders, enemy_centres))

        #This list stores the units that do not have a locked order yet
        remaining_locations = [location for location in orderable_locations if location not in locked_orders]

        #This dict stores the three best orders for each remaining unit
        top3_by_location = {}
        for location in remaining_locations:
            top3_by_location[location] = self.score_movement_location(location, all_possible_orders, orderable_locations, enemy_centres)

        #This step fixes units that chose the same destination if it is switched on, otherwise every unit takes its first choice
        if self.TECHNIQUES['collision_resolution']:
            final_orders_dict = self.resolve_collisions(top3_by_location, all_possible_orders)
        else:
            final_orders_dict = {location: candidates[0][1] for location, candidates in top3_by_location.items() if candidates}

        #This adds the locked attack and convoy orders to the final orders
        final_orders_dict.update(locked_orders)

        #This step removes moves that block our own units, if it is switched on
        if self.TECHNIQUES['self_block_removal']:
            final_orders_dict = self.remove_self_blocks(final_orders_dict, top3_by_location)

        #This step fixes supports that no longer match what the supported unit is doing, if it is switched on
        if self.TECHNIQUES['support_reconciliation']:
            final_orders_dict = self.reconcile_supports(final_orders_dict, top3_by_location, all_possible_orders)

        power_orders = list(final_orders_dict.values())

        #This then returns the final list of orders
        return power_orders