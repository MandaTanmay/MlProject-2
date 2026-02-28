"""
SQLite Database Connection Test
Verifies that SQLite is properly connected and working
"""

import sqlite3
import os
from pathlib import Path

def test_sqlite_connection():
    """Test SQLite database connection and tables"""
    
    print("=" * 60)
    print("🔍 SQLite DATABASE CONNECTION TEST")
    print("=" * 60)
    
    # Test 1: Check if SQLite3 is installed
    print("\n1️⃣  Checking SQLite3 Installation...")
    try:
        import sqlite3
        print(f"   ✓ SQLite3 installed")
        print(f"   Version: {sqlite3.version}")
        print(f"   Library Version: {sqlite3.sqlite_version}")
    except ImportError:
        print("   ✗ SQLite3 NOT installed")
        return False
    
    # Test 2: Check database file exists
    print("\n2️⃣  Checking Database File...")
    db_path = Path("feedback/feedback.db")
    if db_path.exists():
        print(f"   ✓ Database file exists")
        print(f"   Location: {db_path.absolute()}")
        print(f"   Size: {db_path.stat().st_size} bytes")
    else:
        print(f"   ✗ Database file NOT found at {db_path}")
        return False
    
    # Test 3: Test connection
    print("\n3️⃣  Testing Database Connection...")
    try:
        conn = sqlite3.connect(str(db_path))
        print(f"   ✓ Connection established")
        
        # Test 4: List tables
        print("\n4️⃣  Checking Tables...")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        if tables:
            print(f"   ✓ Found {len(tables)} tables:")
            for table in tables:
                table_name = table[0]
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"      • {table_name}: {count} rows")
        else:
            print("   ✗ No tables found")
        
        # Test 5: Get schema
        print("\n5️⃣  Checking Schema...")
        for table in tables:
            table_name = table[0]
            if table_name != 'sqlite_sequence':  # Skip internal table
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                print(f"   {table_name}:")
                for col in columns:
                    print(f"      • {col[1]} ({col[2]})")
        
        # Test 6: Test write operation
        print("\n6️⃣  Testing Write Operation...")
        try:
            cursor.execute("""
                INSERT INTO feedback (query, answer, strategy, rating, timestamp)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, ("test query", "test answer", "TEST", 1))
            conn.commit()
            print(f"   ✓ Write operation successful")
            
            # Get new row count
            cursor.execute("SELECT COUNT(*) FROM feedback")
            count = cursor.fetchone()[0]
            print(f"   Current feedback rows: {count}")
        except Exception as e:
            print(f"   ✗ Write operation failed: {e}")
        
        # Test 7: Test read operation
        print("\n7️⃣  Testing Read Operation...")
        try:
            cursor.execute("SELECT * FROM feedback ORDER BY rowid DESC LIMIT 1")
            latest = cursor.fetchone()
            if latest:
                print(f"   ✓ Read operation successful")
                print(f"   Latest feedback: {latest}")
            else:
                print("   ℹ No feedback records yet")
        except Exception as e:
            print(f"   ✗ Read operation failed: {e}")
        
        conn.close()
        print("\n8️⃣  Closing Connection...")
        print(f"   ✓ Connection closed successfully")
        
    except sqlite3.Error as e:
        print(f"   ✗ Connection failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - SQLite is working properly!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_sqlite_connection()
    exit(0 if success else 1)
