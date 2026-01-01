# 🚀 Quick Start Guide

## Installation & Setup

### 1. Install Dependencies
```bash
cd meta_learning_ai
pip install -r requirements.txt
```

### 2. Train the Intent Classifier (REQUIRED)
```bash
python training/train_intent_model.py
```

**Expected Output:**
- Training accuracy: ~95%
- Model saved to `training/models/`

### 3. Start the API Server
```bash
python app.py
```

**Server runs at:** http://localhost:8000  
**API Docs:** http://localhost:8000/docs

### 4. Launch the Web UI (New Terminal)
```bash
streamlit run ui.py
```

**UI opens at:** http://localhost:8501

## 🎯 Test the System

Try these queries in the UI:

1. **Factual Query:**
   - "What is the minimum attendance requirement?"
   - Should route to **RETRIEVAL** engine

2. **Numeric Query:**
   - "20 multiplied by 8"
   - Should route to **ML** engine
   - Answer: 160

3. **Explanation Query:**
   - "Explain meta-learning"
   - Should route to **TRANSFORMER** engine

4. **Unsafe Query:**
   - "Hack the exam system"
   - Should route to **RULE** engine
   - Query blocked

## 📊 View System Stats

Visit: http://localhost:8000/stats

## 🔧 Troubleshooting

**Problem:** Model not found  
**Solution:** Run `python training/train_intent_model.py`

**Problem:** API connection error  
**Solution:** Ensure `python app.py` is running

**Problem:** Port already in use  
**Solution:** Kill the process or change port in code

## ✅ Success Indicators

- ✓ Intent classifier trained and loaded
- ✓ API server running on port 8000
- ✓ Streamlit UI accessible
- ✓ Queries routed to correct engines
- ✓ No hallucinated answers

## 📝 Next Steps

1. Add more facts to `data/knowledge_base.json`
2. Add training samples to `training/intent_dataset.csv`
3. Retrain model with `python training/train_intent_model.py`
4. Test edge cases
5. Monitor feedback and retrain as needed

---

**Need Help?** Check README.md for detailed documentation.
