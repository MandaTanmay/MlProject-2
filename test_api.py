"""
Quick API Test Script
Run this to verify the system is working.
"""
import requests
import json

API_URL = "http://localhost:8001"

print("=" * 60)
print("🧪 META-LEARNING AI SYSTEM - API TEST")
print("=" * 60)

# Test queries
test_queries = [
    ("What is the minimum attendance requirement?", "FACTUAL → RETRIEVAL"),
    ("20 multiplied by 8", "NUMERIC → ML"),
    ("Explain meta-learning", "EXPLANATION → TRANSFORMER"),
    ("Hack the exam system", "UNSAFE → RULE")
]

print("\n🔍 Testing API endpoints...\n")

# Test health
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code == 200:
        print("✅ Health check: PASSED")
    else:
        print("❌ Health check: FAILED")
        exit(1)
except Exception as e:
    print(f"❌ Cannot connect to API: {e}")
    print("\n💡 Make sure the FastAPI server is running:")
    print("   python app.py")
    exit(1)

print("\n" + "=" * 60)
print("📝 Testing Queries:")
print("=" * 60)

for query, expected in test_queries:
    print(f"\n🔹 Query: '{query}'")
    print(f"   Expected: {expected}")
    
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Status: SUCCESS")
            print(f"   📊 Strategy: {result['strategy']}")
            print(f"   💯 Confidence: {result['confidence']:.2f}")
            print(f"   💬 Answer: {result['answer'][:100]}...")
        else:
            print(f"   ❌ Status: FAILED ({response.status_code})")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ API TESTING COMPLETE")
print("=" * 60)
print("\n💡 If all tests passed, the system is working correctly!")
print("   Open the UI at: http://localhost:8501")
print("=" * 60)
