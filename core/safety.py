import re


# --------------------------------------------
# STRICT HACKATHON SAFETY PATTERNS
# --------------------------------------------

HARMFUL_PATTERNS = [

    # Direct violence
    r"\bkill(ing|ed)?\b",
    r"\bmurder(ing|er)?\b",
    r"\bassassinat(e|ion)\b",
    r"\bstab(bed|bing)?\b",
    r"\bshoot(ing)?\b",
    r"\bstrangle\b",
    r"\bpoison(ing)?\b",
    r"\btorture\b",
    r"\bslaughter\b",
    r"\bmutilate\b",

    # Explosives & weapons
    r"\bbomb\b",
    r"\bexplosive\b",
    r"\bi\.?e\.?d\.?\b",
    r"\bdetonator\b",
    r"\bwarhead\b",
    r"\bweapon\b",
    r"\bfirearm\b",
    r"\bgunpowder\b",
    r"\bammo\b",

    # Drug manufacturing
    r"\bfentanyl\b",
    r"\bmeth\b",
    r"\bamphetamine\b",
    r"\bclandestine lab\b",
    r"\brecipe for (drugs|meth|ice)\b",
    r"\bpill press\b",

    # Cybercrime
    r"\bhack(ing|er)?\b",
    r"\bphish(ing)?\b",
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bddos\b",
    r"\bsql injection\b",
    r"\bbrute force\b",

    # Violence paraphrases
    r"\beliminate\b",
    r"\bget rid of\b",
    r"\bneutralize\b",
    r"\bend (his|her|their) life\b",
    r"\bremove someone\b",
    r"\bpermanently remove\b"
]


def is_harmful_input(text: str) -> bool:
    """
    Returns True if input text appears harmful or malicious.
    """

    text = text.lower().strip()

    # --------------------------------------------
    # 1. Direct keyword match
    # --------------------------------------------
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text):
            return True

    # --------------------------------------------
    # 2. Intent-based detection
    # --------------------------------------------
    # If user asks HOW TO + harmful action
    if "how to" in text and any(word in text for word in [
        "kill", "eliminate", "remove", "neutralize", "destroy"
    ]):
        return True

    # --------------------------------------------
    # 3. Additional intent phrases
    # --------------------------------------------
    if "ways to" in text and any(word in text for word in [
        "kill", "eliminate", "remove", "neutralize"
    ]):
        return True

    return False