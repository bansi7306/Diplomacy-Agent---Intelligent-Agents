from diplomacy import Game
from agent_09 import StudentAgent
import time


# ============================================================
# AGGRESSION TEST
# ============================================================

game = Game()

agent = StudentAgent()
agent.new_game(game, power_name="FRANCE")

attack_orders = {
    "GERMANY": ["A BUR - PAR"],
    "FRANCE": [],
}

peaceful_orders = {
    "GERMANY": ["A MUN - RUH"],
    "FRANCE": [],
}

print("Initial aggression:")
print(agent.opponent_model["GERMANY"]["aggression"])


# Germany attacks France three times.
for turn in range(1, 4):

    agent.update_opponent_aggression(attack_orders)

    aggression = agent.opponent_model["GERMANY"]["aggression"]

    print(f"After attack {turn}: {aggression:.4f}")


# Germany stops attacking France for three turns.
for turn in range(1, 4):

    agent.update_opponent_aggression(peaceful_orders)

    aggression = agent.opponent_model["GERMANY"]["aggression"]

    print(f"After peaceful turn {turn}: {aggression:.4f}")


# ============================================================
# PRESSURE TEST
# ============================================================

print("\n--- PRESSURE TEST ---")

# Create a separate game so this test doesn't affect
# the aggression test above.
pressure_game = Game()

# Place a German army in Burgundy.
pressure_game.set_units("GERMANY", ["A BUR"])

# Initialise a French agent with this board.
pressure_agent = StudentAgent()
pressure_agent.new_game(pressure_game, power_name="FRANCE")

pressure = pressure_agent.get_opponent_pressure("GERMANY")

aggression = pressure_agent.opponent_model["GERMANY"]["aggression"]

print("Germany pressure with army in Burgundy:", pressure)
print("Germany aggression:", aggression)

assert 0.0 <= pressure <= 1.0
assert aggression == 0.0

print("Pressure test passed!")


# ============================================================
# THREAT TEST
# ============================================================

print("\n--- THREAT TEST ---")

aggression = pressure_agent.opponent_model["GERMANY"]["aggression"]

pressure = pressure_agent.get_opponent_pressure("GERMANY")

strength = pressure_agent.get_opponent_strength("GERMANY")

threat = pressure_agent.get_opponent_threat("GERMANY")

print("Aggression:", aggression)
print("Pressure:", pressure)
print("Strength:", strength)
print("Threat:", threat)

expected_threat = (
    0.4 * aggression
    + 0.4 * pressure
    + 0.2 * strength
)

assert abs(threat - expected_threat) < 1e-9

print("Threat test passed!")


# ============================================================
# BUILDER A INTEGRATION TEST
# ============================================================

print("\n--- BUILDER A INTEGRATION TEST ---")

# Calculate Ben's original movement score without opponent threat.
base_score = agent.score_order(
    False,
    4,
    True,
    False,
    0.0
)

# Calculate the same movement score with opponent threat.
threat_score = agent.score_order(
    False,
    4,
    True,
    False,
    0.5
)

print("Ben's original score:", base_score)
print("Score with opponent modelling:", threat_score)

assert base_score == 6.0
assert threat_score == 5.0

print("Integration test passed!")


# ============================================================
# DESTINATION THREAT TEST
# ============================================================

print("\n--- DESTINATION THREAT TEST ---")

# Check whether the agent can calculate the threat
# associated with a movement destination.

destination_threat = pressure_agent.get_destination_threat("PAR")

print("Threat around Paris:", destination_threat)

assert 0.0 <= destination_threat <= 1.0

print("Destination threat test passed!")


print("\nAll opponent modelling tests passed!")



print("\n--- FULL AGENT TEST ---")

start = time.perf_counter()

actions = agent.get_actions()

elapsed = time.perf_counter() - start

print("Generated orders:")
for action in actions:
    print(action)

print(f"\nTime taken: {elapsed:.4f} seconds")

assert isinstance(actions, list)
assert elapsed < 1.0, "Agent exceeded the one-second limit!"

print("Full agent test passed!")