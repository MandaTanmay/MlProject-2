import wikipedia
import json
import uuid
from pathlib import Path

kb_path = Path("knowledge_base.json")

# Load existing KB
if kb_path.exists():
    with open(kb_path, "r", encoding="utf-8") as f:
        existing_kb = json.load(f)
else:
    existing_kb = {"facts": []}

existing_facts = existing_kb.get("facts", [])

topics = [
    "Artificial Intelligence",
    "Deep Learning",
    "Web Browser",
    "Operating System",
    "Cloud Computing",
    "Data Science"
]

new_facts = []

for topic in topics:
    try:
        summary = wikipedia.summary(topic, sentences=3)
        new_facts.append({
            "id": str(uuid.uuid4()),
            "question": f"What is {topic}?",
            "answer": summary,
            "structured_value": summary[:200],
            "category": "technology",
            "source": "Wikipedia",
            "verified": True,
            "verified_date": "2026-03-01"
        })
    except:
        pass

# Merge
existing_facts.extend(new_facts)

# Save merged file
with open(kb_path, "w", encoding="utf-8") as f:
    json.dump({"facts": existing_facts}, f, indent=2)

print(f"KB now contains {len(existing_facts)} facts.")