import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.domain_classifier import DomainClassifier

c=DomainClassifier()
print('model_loaded=', c.is_loaded)
for q in ['2+2','calculate 2+2','what is 2+2','Explain meta-learning','movie times tonight', 'explain about quantam computing', 'explain about quantum computing', 'srkr', 'SRKR Engineering College', 'srkr engineering college']:
    d, conf = c.predict(q)
    print(q, '->', d, conf)
