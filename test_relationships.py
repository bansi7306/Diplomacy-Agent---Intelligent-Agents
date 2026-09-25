from diplomacy import Game
from agent_09 import StudentAgent


def create_agent():
    """Create a fresh agent playing as France."""

    game = Game()
    agent = StudentAgent()
    agent.new_game(game, "FRANCE")

    return agent


def show_relationships(agent):
    """Print all opponents' numerical relationship scores."""

    relationships = agent.get_all_opponent_relationships()

    for country, score in relationships.items():
        print(f"{country}: {score:.2f}")

    return relationships


# ============================================================
# TEST 1: INITIAL RELATIONSHIPS
# ============================================================

print("\n--- TEST 1: INITIAL RELATIONSHIPS ---")

agent = create_agent()

relationships = show_relationships(agent)

# Every opponent should initially be neutral (5.0)
for country, score in relationships.items():
    assert score == 5.0, (
        f"{country} should start neutral, but got {score}"
    )

assert "FRANCE" not in relationships
assert len(relationships) == 6

print("Initial relationships test passed!")


# ============================================================
# TEST 2: REPEATED ATTACKS INCREASE HOSTILITY
# ============================================================

print("\n--- TEST 2: HOSTILE OPPONENT ---")

agent = create_agent()

attack_orders = {
    "GERMANY": ["A BUR - PAR"]
}

expected_scores = [6.5, 7.55, 8.29]

for turn in range(3):

    agent.update_opponent_relationships(attack_orders)

    relationships = agent.get_all_opponent_relationships()

    germany_score = relationships["GERMANY"]

    print(
        f"After attack {turn + 1}: "
        f"Germany = {germany_score:.2f}"
    )

    assert germany_score == expected_scores[turn]

assert relationships["GERMANY"] > 5.0

print("Hostile relationship test passed!")


# ============================================================
# TEST 3: SUPPORT INCREASES FRIENDLINESS
# ============================================================

print("\n--- TEST 3: FRIENDLY OPPONENT ---")

agent = create_agent()

support_orders = {
    "ENGLAND": ["F ENG S A PAR"]
}

expected_scores = [3.5, 2.45, 1.71]

for turn in range(3):

    agent.update_opponent_relationships(support_orders)

    relationships = agent.get_all_opponent_relationships()

    england_score = relationships["ENGLAND"]

    print(
        f"After support {turn + 1}: "
        f"England = {england_score:.2f}"
    )

    assert england_score == expected_scores[turn]

assert relationships["ENGLAND"] < 5.0

print("Friendly relationship test passed!")


# ============================================================
# TEST 4: PEACEFUL BEHAVIOUR RETURNS TOWARDS NEUTRAL
# ============================================================

print("\n--- TEST 4: RETURN TOWARDS NEUTRAL ---")

agent = create_agent()

# First establish hostility
for turn in range(3):
    agent.update_opponent_relationships(attack_orders)

relationships = agent.get_all_opponent_relationships()

print(
    f"After repeated attacks: "
    f"Germany = {relationships['GERMANY']:.2f}"
)

# Germany stops attacking France
peaceful_orders = {
    "GERMANY": ["A BUR H"]
}

previous_score = relationships["GERMANY"]

for turn in range(5):

    agent.update_opponent_relationships(peaceful_orders)

    relationships = agent.get_all_opponent_relationships()

    germany_score = relationships["GERMANY"]

    print(
        f"After peaceful turn {turn + 1}: "
        f"Germany = {germany_score:.2f}"
    )

    # Germany should become less hostile each turn
    assert germany_score < previous_score

    previous_score = germany_score

# Germany should be close to neutral after five peaceful turns
assert 4.0 <= relationships["GERMANY"] <= 6.0

print("Return towards neutral test passed!")


# ============================================================
# TEST 5: UNRELATED OPPONENT REMAINS NEUTRAL
# ============================================================

print("\n--- TEST 5: NEUTRAL OPPONENT ---")

agent = create_agent()

unrelated_orders = {
    "TURKEY": ["A CON H"]
}

for turn in range(3):
    agent.update_opponent_relationships(unrelated_orders)

relationships = agent.get_all_opponent_relationships()

print(f"Turkey: {relationships['TURKEY']:.2f}")

assert relationships["TURKEY"] == 5.0

print("Neutral relationship test passed!")


# ============================================================
# TEST 6: COMPLETE NUMERICAL RELATIONSHIP DICTIONARY
# ============================================================

print("\n--- TEST 6: COMPLETE RELATIONSHIP DICTIONARY ---")

agent = create_agent()

# Germany attacks France three times
for turn in range(3):
    agent.update_opponent_relationships({
        "GERMANY": ["A BUR - PAR"]
    })

# England supports France three times
for turn in range(3):
    agent.update_opponent_relationships({
        "ENGLAND": ["F ENG S A PAR"]
    })

relationships = show_relationships(agent)

# Germany should be hostile
assert relationships["GERMANY"] == 8.29

# England should be friendly
assert relationships["ENGLAND"] == 1.71

# Other opponents should remain neutral
assert relationships["AUSTRIA"] == 5.0
assert relationships["ITALY"] == 5.0
assert relationships["RUSSIA"] == 5.0
assert relationships["TURKEY"] == 5.0

# Our own country should not appear
assert "FRANCE" not in relationships

# All six opponents should appear
assert len(relationships) == 6

# Every score must be numerical and within the 0-10 range
for country, score in relationships.items():

    assert isinstance(score, (int, float)), (
        f"{country} has a non-numerical score: {score}"
    )

    assert 0 <= score <= 10, (
        f"{country} has a score outside the 0-10 range: {score}"
    )

print("Complete relationship dictionary test passed!")


print("\nAll numerical relationship tests passed!")