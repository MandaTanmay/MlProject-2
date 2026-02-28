"""
Test Script: Verify Feedback Storage
Tests if feedback data is being stored in the SQLite database
"""

import sqlite3
from datetime import datetime
from pathlib import Path

def test_feedback_storage():
    """Test if feedback can be stored in database"""
    
    print("=" * 80)
    print("🧪 FEEDBACK STORAGE TEST")
    print("=" * 80)
    
    db_path = Path("feedback/feedback.db")
    
    # Test 1: Check if database exists
    print("\n1️⃣  Checking database file...")
    if db_path.exists():
        print(f"   ✓ Database exists at {db_path.absolute()}")
        print(f"   Size: {db_path.stat().st_size} bytes")
    else:
        print(f"   ✗ Database NOT found at {db_path}")
        return False
    
    # Test 2: Connect to database
    print("\n2️⃣  Testing connection...")
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        print("   ✓ Connection successful")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False
    
    # Test 3: Check tables exist
    print("\n3️⃣  Checking tables...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"   Tables found: {tables}")
    
    if 'feedback' not in tables:
        print("   ✗ 'feedback' table NOT found!")
        return False
    print("   ✓ 'feedback' table exists")
    
    # Test 4: Test INSERT operation
    print("\n4️⃣  Testing INSERT (storing test feedback)...")
    try:
        cursor.execute("""
            INSERT INTO feedback (
                timestamp, query, predicted_intent, predicted_confidence,
                strategy_used, answer, user_feedback, user_comment, was_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            "Test query for database",
            "FACTUAL",
            0.95,
            "RETRIEVAL",
            "Test answer",
            1,
            "Test comment",
            1
        ))
        conn.commit()
        print("   ✓ INSERT operation successful")
    except Exception as e:
        print(f"   ✗ INSERT failed: {e}")
        conn.close()
        return False
    
    # Test 5: Verify data was inserted
    print("\n5️⃣  Verifying data insertion...")
    cursor.execute("SELECT COUNT(*) FROM feedback")
    count = cursor.fetchone()[0]
    print(f"   Total records in feedback table: {count}")
    
    if count > 0:
        print("   ✓ Data is being stored successfully!")
        
        # Show latest record
        cursor.execute("SELECT id, timestamp, query, predicted_intent FROM feedback ORDER BY id DESC LIMIT 1")
        latest = cursor.fetchone()
        print(f"\n   Latest record:")
        print(f"   ID: {latest[0]}")
        print(f"   Timestamp: {latest[1]}")
        print(f"   Query: {latest[2]}")
        print(f"   Intent: {latest[3]}")
    else:
        print("   ✗ No data found in feedback table!")
        conn.close()
        return False
    
    # Test 6: Check if FeedbackStore class works
    print("\n6️⃣  Testing FeedbackStore class...")
    try:
        from feedback.feedback_store import FeedbackStore
        store = FeedbackStore()
        
        success = store.store_feedback(
            query="Another test query",
            predicted_intent="NUMERIC",
            predicted_confidence=0.92,
            strategy="ML",
            answer="42",
            user_feedback=1,
            user_comment="Good answer"
        )
        
        if success:
            print("   ✓ FeedbackStore.store_feedback() works!")
        else:
            print("   ✗ FeedbackStore.store_feedback() failed")
            
    except Exception as e:
        print(f"   ✗ FeedbackStore test failed: {e}")
    
    # Test 7: Display all feedback
    print("\n7️⃣  All feedback in database:")
    print("-" * 80)
    cursor.execute("SELECT id, timestamp, query, predicted_intent, user_feedback FROM feedback")
    rows = cursor.fetchall()
    
    if rows:
        for row in rows:
            feedback_emoji = "👍" if row[4] == 1 else "👎"
            print(f"  ID {row[0]}: [{row[1]}] {row[2][:40]}... Intent: {row[3]} {feedback_emoji}")
    else:
        print("  No records found")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("✅ FEEDBACK STORAGE TEST COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    test_feedback_storage()
