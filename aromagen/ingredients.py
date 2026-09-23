"""
The cartridges fitted on the device, and which pump each one sits on.

One pump per vibe (see recipes.py):

    pump 1   confession
    pump 2   solitude      <- also the resting scent, diffused when nothing happens
    pump 3   resonance

**The oils are yours to choose.** Until they are decided, each pump is declared
with a placeholder: fill in `name`, `family` and `notes` below when you know
what goes in, and nothing else needs to change -- recipes address pumps, not
perfumes, so a cartridge swap never touches recipes.py.

`name` and `notes` are only shown by `python -m aromagen --recipes`; the
classifier never reads them. What a vibe *means* lives in its `theme`, in
recipes.py.

Pumps 4, 5 and 6 are free. To bring one into play, declare it here and add its
key to a recipe's `doses`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ingredient:
    valve: int      # 1..6, pump number on the device
    key: str        # short identifier used by recipes
    name: str       # product name as sold
    family: str     # olfactory family
    notes: str      # sensory description, shown in the catalogue


INGREDIENTS: tuple[Ingredient, ...] = (
    Ingredient(
        valve=1,
        key="pump1",
        name="Pump 1",
        family="to be chosen",
        notes="for CONFESSION -- something close and enclosed carries this best: "
              "a wood, a resin, a musk. Replace this line with the real oil.",
    ),
    Ingredient(
        valve=2,
        key="pump2",
        name="Pump 2",
        family="to be chosen",
        notes="for SOLITUDE -- the resting scent, present far more often than the "
              "other two, so keep it quiet and easy to live with. Replace this "
              "line with the real oil.",
    ),
    Ingredient(
        valve=3,
        key="pump3",
        name="Pump 3",
        family="to be chosen",
        notes="for RESONANCE -- something bright and radiant that carries across "
              "a room reads best here. Replace this line with the real oil.",
    ),
    # Pumps 4, 5 and 6: free. Declare one here to use it in a recipe.
)

BY_KEY: dict[str, Ingredient] = {i.key: i for i in INGREDIENTS}
BY_VALVE: dict[int, Ingredient] = {i.valve: i for i in INGREDIENTS}

ALL_VALVES = tuple(range(1, 7))
ASSIGNED_VALVES = tuple(sorted(BY_VALVE))
EMPTY_VALVES = tuple(v for v in ALL_VALVES if v not in BY_VALVE)

assert len(BY_KEY) == len(INGREDIENTS), "duplicate ingredient key"
assert len(BY_VALVE) == len(INGREDIENTS), "two ingredients on the same pump"


def valve_name(valve: int) -> str:
    """Display name for a pump, whether or not a cartridge is assigned."""
    ing = BY_VALVE.get(valve)
    return ing.name if ing else f"pump {valve} (empty)"
