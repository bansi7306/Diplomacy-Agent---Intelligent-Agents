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

        '''Implement your agent here.'''

    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_map_graphs()

        '''Implement your agent here.'''

    def build_map_graphs(self):
        if not self.game:
            raise Exception('Game Not Initialised. Cannot Build Map Graphs.')

        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        locations = list(self.game.map.loc_type.keys()) # locations with '/' are not real provinces

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

    def get_enemy_centres(self):
        enemy_centres = []
        for i in self.game.map.scs:
            if i not in self.game.get_centers(self.power_name):  # all centres not controlled by self
                enemy_centres.append(i)
        return enemy_centres

    def get_own_units_and_locations(self):
        return {
            'units': self.game.get_units(self.power_name),
            'orderable_locations': self.game.get_orderable_locations(self.power_name)
        }

    def classify_orders(self, possible_orders):
        moves = []
        holds = []
        supports = []
        for order in possible_orders:
            if ' S ' in order:
                supports.append(order)
            elif ' - ' in order:
                moves.append(order)
            elif order.endswith(' H'):
                holds.append(order)
        return moves, holds, supports

    def get_move_destination(self, order):
        words = order.split(' ')
        dash_index = words.index('-')
        return words[dash_index + 1]

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

    def is_move_supportable(self, move_order, all_possible_orders, own_orderable_locations):
        for other_loc in own_orderable_locations:
            for candidate in all_possible_orders.get(other_loc, []):
                if ' S ' in candidate and move_order in candidate:
                    return True
        return False
    
    def is_contested(self, destination):
        for power_name in self.game.powers.keys():
            if power_name == self.power_name:
                continue
            for unit in self.game.get_units(power_name):
                unit_loc = unit.split(' ')[1]
                if unit_loc == destination:
                    return True
        return False

    def score_order(self, is_hold, distance_score_val, is_supportable, is_contested_flag):
        if is_hold:
            return 1.0
        score = distance_score_val
        if is_supportable:
            score += 2
        if is_contested_flag:
            score -= 3
        return score

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
 
            dist_score = self.distance_score(graph, destination, enemy_centres)
            supportable = self.is_move_supportable(move, all_possible_orders, own_orderable_locations)
            contested = self.is_contested(destination)
 
            total_score = self.score_order(False, dist_score, supportable, contested)
            scored_candidates.append((total_score, move))
 
        for hold in holds:
            total_score = self.score_order(True, 0, False, False)
            scored_candidates.append((total_score, hold))
 
        if not scored_candidates:
            return possible_orders[0]
 
        max_score = max(s for s, o in scored_candidates)
        top_candidates = [o for s, o in scored_candidates if s == max_score]
        return random.choice(top_candidates)

    @timeout_decorator.timeout(1) # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    def update_game(self, all_power_orders):
        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    @timeout_decorator.timeout(1)
    def get_actions(self):

        '''Implement your agent here.'''
        
        own_info = self.get_own_units_and_locations()
        orderable_locations = own_info['orderable_locations']
 
        if not orderable_locations:
            return []
 
        all_possible_orders = self.game.get_all_possible_orders()
 
        if self.game.phase_type != 'M':
            power_orders = []
            for loc in orderable_locations:
                possible = all_possible_orders.get(loc, [])
                if possible:
                    power_orders.append(random.choice(possible))
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
