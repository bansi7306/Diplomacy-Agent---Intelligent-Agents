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

        '''Implement your agent here.'''

        # Initialise opponent aggression for each new game
        self.opponent_model = {}

        for opponent in self.game.powers:
            if opponent != self.power_name:
                self.opponent_model[opponent] = {
                    'aggression': 0.0
                }

    # Updates opponent aggression based on their observed orders
    def update_opponent_aggression(self, all_power_orders):

        for opponent, orders in all_power_orders.items():

            # Ignore our own orders
            if opponent == self.power_name:
                continue

            if opponent not in self.opponent_model:
                self.opponent_model[opponent] = {
                    'aggression': 0.0
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
    def get_opponent_threat(self, opponent):

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
            0.4 * aggression
            + 0.4 * pressure
            + 0.2 * strength
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

    # Calculates the highest opponent threat around a destination
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

                    threat = self.get_opponent_threat(opponent)

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
            #These are the movement moves
            elif ' - ' in order:
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

    #This fucntion scores a destination by how close it is to the nearest enemy center by using the map graph to find the shortest path
    def distance_score(self, graph, destination, enemy_centres):
        if destination not in graph:
            return 0
        try:
            paths = nx.shortest_path(graph, source=destination)
        except nx.NodeNotFound:
            return 0

        min_dist = 1000
        for centre in enemy_centres:
            if centre in paths:
                dist = len(paths[centre]) - 1
                if dist < min_dist:
                    min_dist = dist

        if min_dist == 1000:
            return 0

        return max(0, 5 - min_dist)

    #This checks if any other units that we control has a legal support order backing this specific move
    def is_move_supportable(self, move_order, all_possible_orders, own_orderable_locations):
        for other_loc in own_orderable_locations:
            for candidate in all_possible_orders.get(other_loc, []):
                if ' S ' in candidate and move_order in candidate:
                    return True
        return False
    
    #This function checks if an enemy unit is currently occupying the space of our intended move destination
    def is_contested(self, destination):
        for power_name in self.game.powers.keys():
            if power_name == self.power_name:
                continue
            for unit in self.game.get_units(power_name):
                unit_loc = unit.split(' ')[1]
                if unit_loc == destination:
                    return True
        return False

    '''This combines our scoring factors into one final score, so dfistance, support and contested are combined into one score for the candidate order
    def score_order(self, is_hold, distance_score_val, is_supportable, is_contested_flag):
        if is_hold:
            return 1.0
        score = distance_score_val
        if is_supportable:
            score += 2
        if is_contested_flag:
            score -= 3
        return score

    #This is a combination of Ben's strategy and Bansi's oppeonent model'''
    
    def score_order(self, is_hold, distance_score_val, is_supportable, is_contested_flag, opponent_threat=0.0):

        if is_hold:
            return 1.0

        score = distance_score_val

        if is_supportable:
            score += 2

        if is_contested_flag:
            score -= 3

        # Opponent modelling contribution
        score -= 2 * opponent_threat

        return score

    #This scores every candidate order at one location and returns the best one only for the movement phase
    def score_movement_location(self, loc, all_possible_orders, own_orderable_locations, enemy_centres):
        '''
        Scores every candidate order at one location during a Movement phase,
        and returns the best one as a string.
        '''
        possible_orders = all_possible_orders.get(loc, [])
        if not possible_orders:
            return None
 
        moves, holds, supports = self.classify_orders(possible_orders)
 
        scored_candidates = []
 
        for move in moves:
            unit_type = move[0]
            graph = self.map_graph_army if unit_type == 'A' else self.map_graph_navy
            destination = self.get_move_destination(move)

            '''Ben's code
            #dist_score = self.distance_score(graph, destination, enemy_centres)
            #supportable = self.is_move_supportable(move, all_possible_orders, own_orderable_locations)
            #contested = self.is_contested(destination)
 
            #total_score = self.score_order(False, dist_score, supportable, contested)

            #Bansi's code combined with Ben's'''

            dist_score = self.distance_score(graph, destination, enemy_centres)

            supportable = self.is_move_supportable(move, all_possible_orders, own_orderable_locations)

            contested = self.is_contested(destination)

            # Get opponent threat associated with this destination
            opponent_threat = self.get_destination_threat(destination)

            
            total_score = self.score_order(
                False,
                dist_score,
                supportable,
                contested,
                opponent_threat
            )


            scored_candidates.append((total_score, move))
 
        for hold in holds:
            total_score = self.score_order(True, 0, False, False)
            scored_candidates.append((total_score, hold))
 
        if not scored_candidates:
            return possible_orders[0]
 
        max_score = max(s for s, o in scored_candidates)
        top_candidates = [o for s, o in scored_candidates if s == max_score]
        return random.choice(top_candidates)
    
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

            total_score = safety_score - (3 if contested else 0)
            scored_candidates.append((total_score, retreat))

        max_score = max(s for s, o in scored_candidates)
        top_candidates = [o for s, o in scored_candidates if s == max_score]
        return random.choice(top_candidates)

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

        total_score = (danger_score * 10) + (5 - tiebreak_score)

        return disband_order, total_score

    @timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    def update_game(self, all_power_orders):

        # Observe opponents before the game state changes
        if self.game.phase_type == 'M':
            self.update_opponent_aggression(all_power_orders)

        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    #This is called every turn and returns our full list of orders
    @timeout_decorator.timeout(1)
    def get_actions(self):

        '''Implement your agent here.'''
        
        own_info = self.get_own_units_and_locations()
        orderable_locations = own_info['orderable_locations']
 
        if not orderable_locations:
            return []
 
        all_possible_orders = self.game.get_all_possible_orders()
 
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
 
        power_orders = []
        for loc in orderable_locations:
            best_order = self.score_movement_location(
                loc, all_possible_orders, orderable_locations, enemy_centres
            )
            if best_order:
                power_orders.append(best_order)
 
        return power_orders

        '''
        Return a list of orders. Each order is a string, with specific format. For the format, read the game rule and game engine documentation.
        
        Expected format:
        A LON H                  # Army at LON holds
        F IRI - MAO              # Fleet at IRI moves to MAO (and attack)
        A WAL S F LON            # Army at WAL supports Fleet at LON (and hold)
        F NTH S A EDI - YOR      # Fleet at NTH supports Army at EDI to move to YOR
        F NWG C A NWY - EDI      # Fleet at NWG convoys Army at NWY to EDI
        A NWY - EDI VIA          # Army at NWY moves to EDI via convoy
        A WAL R LON              # Army at WAL retreats to LON
        A LON D                  # Disband Army at LON
        A LON B                  # Build Army at LON
        F EDI B                  # Build Fleet at EDI

        Note: If an invalid order is sent to the engine, it will be accepted but with a result of 'void' (no effect).
        Note: For a 'support' action, two orders are needed, one for the supporter and one for the supportee. (Same for 'convoy')
        Note: For each unit, if no order is given, it will 'hold' by default.

        Useful Functions:
        
        # This is a dict of all the possible orders for each unit at each location (for all powers).
        possible_orders = self.game.get_all_possible_orders()

        # This is a list of all orderable locations for the power you control.
        orderable_locations = self.game.get_orderable_locations(self.power_name)
    
        # Combining these two, you can have the full action space for the power you control.

        # You can re-use the build_map_graphs function in the GreedyAgent to build the connection graph of the map if needed.
        
        '''
