from diplomacy import Game
from agent_09 import StudentAgent
from agent_baselines import StaticAgent
from game import copy_game

game = Game()
agent = StudentAgent()
agent.new_game(copy_game(game), 'TURKEY')

for turn in range(20):
    phase = game.get_current_phase()
    orders = agent.get_actions()

    if game.phase_type == 'M':
        print(f'--- {phase} ---')
        print('  Target:  ', agent.current_target)
        print('  Units:   ', game.get_units('TURKEY'))
        print('  Centres: ', game.get_centers('TURKEY'))
        print('  Orders:  ', orders)

    all_orders = {p: [] for p in game.powers}
    all_orders['TURKEY'] = orders
    for p, o in all_orders.items():
        game.set_orders(p, o)
    game.process()
    agent.update_game(all_orders)

    if game.is_game_done:
        break