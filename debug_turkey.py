import sys
from diplomacy import Game
from agent_09 import StudentAgent
from game import copy_game

# Usage:
#   python3 debug_turkey.py                 -> Turkey, full detail
#   python3 debug_turkey.py FRANCE          -> France, full detail
#   python3 debug_turkey.py FRANCE summary  -> France, one line per year
POWER = sys.argv[1] if len(sys.argv) > 1 else 'TURKEY'
SUMMARY = len(sys.argv) > 2 and sys.argv[2] == 'summary'

game = Game()
agent = StudentAgent()
agent.new_game(copy_game(game), POWER)

for turn in range(60):
    phase = game.get_current_phase()
    orders = agent.get_actions()
    is_movement = game.phase_type == 'M'

    if is_movement and not SUMMARY:
        print(f'--- {phase} ---')
        print('  Target:  ', agent.current_target)
        print('  Units:   ', game.get_units(POWER))
        print('  Centres: ', game.get_centers(POWER))
        print('  Orders:  ', orders)

    # everyone else holds (Static opponents)
    all_orders = {p: [] for p in game.powers}
    all_orders[POWER] = orders
    for p, o in all_orders.items():
        game.set_orders(p, o)
    game.process()
    agent.update_game(all_orders)

    if is_movement and not SUMMARY:
        # a move failed if we have no unit on its destination afterwards
        our_locs = {u.split()[1][:3] for u in game.get_units(POWER)}
        failed = []
        for o in orders:
            parts = o.split()
            if len(parts) >= 4 and parts[2] == '-':
                if parts[3][:3] not in our_locs:
                    failed.append(o)
        print('  Failed:  ', failed)

    if SUMMARY and phase.startswith('F') and phase.endswith('M'):
        print(f'{phase}: {len(game.get_centers(POWER))} centres, target {agent.current_target}')

    if game.is_game_done:
        break