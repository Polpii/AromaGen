"""
The vibes of the piece, and the scent each one diffuses.

Three vibes, **one pump each**, so every vibe is a single unmistakable material
rather than a blend a visitor has to unpick. It also makes dosing trivial -- one
valve opens for the whole burst -- and satisfies the hardware limit of one valve
at a time for free.

    confession   pump 1    anything sad or vulnerable
    solitude     pump 2    flat and neutral             <- the resting scent
    resonance    pump 3    anything happy or connecting

Solitude is the resting state and the honest answer much of the time, which is
why the engine lets it take the room back only slowly (see engine.py).

Which oil sits on which pump is declared in ingredients.py, and is yours to
choose; nothing here needs to change when a cartridge is swapped.

`doses` gives each ingredient a share from 0 to 3. The device cannot modulate
flow -- a valve is either open or shut -- so dosing is done by OPEN TIME within
a burst (see device.burst_schedule). With one pump per vibe the share is
irrelevant; it matters again as soon as a vibe blends two pumps.

`theme` is not decoration: it goes verbatim into the classifier prompt and is
what makes the model tell these vibes apart. After editing one, re-run
`python tests/check_vibes.py`.

Keywords are the offline fallback only, used when the language model is
unavailable. English first, French second: the room is expected to speak mostly
English, but French must work too.
"""

from dataclasses import dataclass, field

from .ingredients import BY_KEY as ING

MAX_DOSE = 3


@dataclass(frozen=True)
class Recipe:
    key: str
    label: str              # display name
    theme: str              # what it stands for; read by the classifier
    doses: dict[str, int]   # ingredient key -> 0..3
    keywords: tuple[str, ...] = field(default=())  # offline lexical fallback

    def valves(self) -> dict[int, int]:
        """
        Return {pump number: dose} for non-zero doses of mounted cartridges.
        Anything not currently fitted is skipped -- see `available`, which is
        what actually gates a recipe from being chosen.
        """
        return {ING[k].valve: d for k, d in self.doses.items() if d > 0 and k in ING}

    def describe(self) -> str:
        parts = [f"{ING[k].name if k in ING else k} {d}/{MAX_DOSE}"
                 for k, d in sorted(self.doses.items(), key=lambda kv: -kv[1]) if d > 0]
        return " + ".join(parts)

    def missing(self) -> tuple:
        """Ingredient keys this recipe needs that are not currently mounted."""
        return tuple(k for k, d in self.doses.items() if d > 0 and k not in ING)

    @property
    def available(self) -> bool:
        return not self.missing()


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        key="confession",
        label="Confession",
        theme="Anything sad, heavy or vulnerable. One voice opening up -- an "
              "admission, a secret, a fear, a regret, shame -- and sadness of "
              "every kind: grief, loss, loneliness, disappointment, bad news. "
              "Whether it comes out heavily or is slipped in lightly.",
        doses={"pump1": 3},
        keywords=("to be honest", "honestly", "I have never told", "I need to tell you",
                  "can I tell you", "the truth is", "I am scared", "I am afraid",
                  "I feel like", "I am ashamed", "I regret", "between us",
                  "I lied", "I never told anyone", "I am not okay",
                  "sad", "I miss", "passed away", "died", "lonely", "alone",
                  "depressed", "crying", "I cried", "heartbroken", "it hurts",
                  "je dois te dire", "pour etre honnete", "en vrai", "j ai jamais dit",
                  "j ai peur", "je me sens", "j ai honte", "je regrette",
                  "entre nous", "j ai menti", "je vais pas bien",
                  "triste", "me manque", "decede", "mort", "tout seul", "toute seule",
                  "deprime", "je pleure", "j ai pleure", "ca fait mal"),
    ),
    Recipe(
        key="solitude",
        label="Solitude",
        theme="Flat and neutral, nothing felt either way: silence, an empty room, "
              "or talk that carries no emotion -- logistics, directions, plain "
              "greetings, small talk nobody reacts to. Nothing sad and nothing "
              "happy. The resting state of the piece.",
        doses={"pump2": 3},
        keywords=("hello", "hi", "okay", "alright", "anyway", "what time",
                  "coffee", "weather", "the train", "see you", "sure", "the exit",
                  "downstairs", "schedule", "meeting", "tickets", "I will wait",
                  "never mind", "whatever", "I guess",
                  "bonjour", "salut", "d accord", "voila", "quelle heure",
                  "cafe", "metro", "meteo", "reunion", "planning", "la sortie",
                  "en bas", "on se met", "j attends", "tant pis", "bref"),
    ),
    Recipe(
        key="resonance",
        label="Resonance",
        theme="Anything happy, warm or connecting: joy, excitement, laughter, "
              "affection, compliments, good news, and people meeting each other "
              "-- agreement, recognition, finishing one another's thoughts, "
              "enthusiasm bouncing from one person to the next. However small "
              "the subject.",
        doses={"pump3": 3},
        keywords=("exactly", "totally", "same here", "me too", "that is it",
                  "I know right", "absolutely", "so true", "you get it",
                  "we said the same", "right?", "yes!", "haha",
                  "so happy", "I love", "amazing", "wonderful", "congratulations",
                  "so good to see you", "beautiful", "we did it", "best night",
                  "exactement", "carrement", "moi aussi", "c est ca",
                  "tout a fait", "grave", "trop vrai", "je suis d accord",
                  "on dit la meme chose", "voila exactement",
                  "trop content", "trop contente", "genial", "j adore", "trop bien",
                  "felicitations", "ca fait plaisir", "trop beau", "on s amuse"),
    ),
)

BY_KEY: dict[str, Recipe] = {r.key: r for r in RECIPES}
DEFAULT_RECIPE = "solitude"

# Only recipes whose cartridges are actually fitted can be chosen. A recipe
# calling for a pump that is empty would otherwise open the wrong valve.
AVAILABLE_RECIPES: tuple = tuple(r for r in RECIPES if r.available)
UNAVAILABLE_RECIPES: tuple = tuple(r for r in RECIPES if not r.available)

for _r in RECIPES:
    for _k, _d in _r.doses.items():
        assert 0 <= _d <= MAX_DOSE, f"{_r.key}: dose {_d} out of range for {_k}"
assert BY_KEY[DEFAULT_RECIPE].available, (
    f"the resting vibe {DEFAULT_RECIPE!r} needs a pump that is not fitted: "
    f"{BY_KEY[DEFAULT_RECIPE].missing()}")


def catalogue_for_prompt() -> str:
    """Recipe descriptions injected into the classifier prompt."""
    return "\n".join(f"- {r.key}: {r.theme}" for r in AVAILABLE_RECIPES)


def availability_report() -> str:
    """One-line summary of what can actually be diffused right now."""
    parts = [f"{len(AVAILABLE_RECIPES)}/{len(RECIPES)} vibes available"]
    if UNAVAILABLE_RECIPES:
        details = ", ".join(f"{r.key} (needs {'+'.join(r.missing())})"
                            for r in UNAVAILABLE_RECIPES)
        parts.append(f"unavailable: {details}")
    return "; ".join(parts)
