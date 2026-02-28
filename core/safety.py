import re

HARMFUL_PATTERNS = [
    # violence
    r"\bkill\b", r"\bmurder\b", r"\bassassinate\b",
    r"\bstab\b", r"\bshoot\b", r"\bpoison\b",
    r"\bexplode\b", r"\bexplosion\b", r"\bbomb\b",
    r"\bdetonate\b", r"\bblast\b", r"\bdestroy\b",

    # illegal activity
    r"\bhack\b", r"\bcrack\b", r"\bbypass\b",
    r"\bexploit\b", r"\bsteal\b", r"\billegal\b",

    # drugs
    r"\bcocaine\b", r"\bheroin\b", r"\bmeth\b",
    r"\bsynthesize\b", r"\bmake drugs\b"
]

def is_harmful_input(text: str) -> bool:
    text = text.lower()
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text):
            return True
    return False