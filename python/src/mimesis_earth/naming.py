"""Seeded syllable-based place-name generator. Each world draws its own
sound inventory, giving worlds distinct linguistic flavor."""

import numpy as np

ONSETS = [
    "b", "br", "c", "cr", "d", "dr", "f", "g", "gr", "h", "k", "kh", "l",
    "m", "n", "p", "pr", "r", "s", "sh", "st", "t", "th", "tr", "v", "z",
]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ai", "ea", "ia", "ou"]
CODAS = ["", "", "", "n", "r", "l", "s", "th", "m", "nd", "rk"]
SUFFIXES = ["ia", "a", "or", "un", "eth", "ara", "is", "ov", "ane", "und"]


def make_namer(rng: np.random.Generator):
    """Returns a zero-arg function producing unique capitalized names."""
    onsets = [str(x) for x in rng.choice(ONSETS, size=10, replace=False)]
    vowels = [str(x) for x in rng.choice(VOWELS, size=5, replace=False)]
    codas = [str(x) for x in rng.choice(CODAS, size=6, replace=False)]
    suffixes = [str(x) for x in rng.choice(SUFFIXES, size=4, replace=False)]
    used: set[str] = set()

    def pick(seq: list[str]) -> str:
        return seq[int(rng.integers(len(seq)))]

    def namer() -> str:
        name = ""
        for _ in range(50):
            n_syllables = int(rng.integers(2, 4))
            parts = [pick(onsets) + pick(vowels) for _ in range(n_syllables - 1)]
            if rng.random() < 0.6:
                parts.append(pick(onsets) + pick(suffixes))
            else:
                parts.append(pick(onsets) + pick(vowels) + pick(codas))
            name = "".join(parts).capitalize()
            if name not in used:
                used.add(name)
                return name
        name = f"{name}-{len(used)}"  # exhausted retries: disambiguate
        used.add(name)
        return name

    return namer
