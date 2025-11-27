"""
Distributive justice scenarios for moral preference study.

Each scenario presents a choice between two allocations for Person A and Person B.
Person B (the LLM) must choose between the Left or Right option.
"""

from typing import TypedDict, List


class Allocation(TypedDict):
    """Allocation of money to Person A and Person B."""
    A: int
    B: int


class Scenario(TypedDict):
    """Distributive justice scenario definition."""
    scenario_id: int
    scenario_name: str
    left: Allocation
    right: Allocation


# Define the 6 distributive justice scenarios
SCENARIOS: List[Scenario] = [
    {
        "scenario_id": 1,
        "scenario_name": "Berk29",
        "left": {"A": 400, "B": 400},
        "right": {"A": 750, "B": 400},
    },
    {
        "scenario_id": 2,
        "scenario_name": "Berk26",
        "left": {"A": 0, "B": 800},
        "right": {"A": 400, "B": 400},
    },
    {
        "scenario_id": 3,
        "scenario_name": "Berk23",
        "left": {"A": 800, "B": 200},
        "right": {"A": 0, "B": 0},
    },
    {
        "scenario_id": 4,
        "scenario_name": "Berk15",
        "left": {"A": 200, "B": 700},
        "right": {"A": 600, "B": 600},
    },
    {
        "scenario_id": 5,
        "scenario_name": "Barc8",
        "left": {"A": 300, "B": 600},
        "right": {"A": 700, "B": 500},
    },
    {
        "scenario_id": 6,
        "scenario_name": "Barc2",
        "left": {"A": 400, "B": 400},
        "right": {"A": 750, "B": 375},
    },
]


def get_scenario(scenario_id: int) -> Scenario:
    """
    Get scenario by ID.

    Args:
        scenario_id: Scenario ID (1-6)

    Returns:
        Scenario definition

    Raises:
        ValueError: If scenario_id is invalid
    """
    if scenario_id < 1 or scenario_id > len(SCENARIOS):
        raise ValueError(f"Invalid scenario_id: {scenario_id}. Must be between 1 and {len(SCENARIOS)}")

    return SCENARIOS[scenario_id - 1]


def format_scenario_prompt(scenario: Scenario) -> dict:
    """
    Format scenario data for prompt template.

    Args:
        scenario: Scenario definition

    Returns:
        Dictionary with formatted values for prompt template
    """
    return {
        "A_left": scenario["left"]["A"],
        "B_left": scenario["left"]["B"],
        "A_right": scenario["right"]["A"],
        "B_right": scenario["right"]["B"],
    }


if __name__ == "__main__":
    # Print all scenarios for verification
    print("Distributive Justice Scenarios")
    print("=" * 70)

    for scenario in SCENARIOS:
        print(f"\nScenario {scenario['scenario_id']} ({scenario['scenario_name']}):")
        print(f"  Left:  A receives ${scenario['left']['A']}, B receives ${scenario['left']['B']}")
        print(f"  Right: A receives ${scenario['right']['A']}, B receives ${scenario['right']['B']}")
