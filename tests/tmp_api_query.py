import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import requests

API = "http://localhost:8001/query"

for q in ["srkr"]:
    try:
        r = requests.post(API, json={"query": q}, timeout=10)
        try:
            print('Query:', q)
            print('Status:', r.status_code)
            print('Response JSON:')
            print(r.json())
        except Exception:
            print('Non-JSON response:', r.text)
    except Exception as e:
        print('Request failed:', e)
