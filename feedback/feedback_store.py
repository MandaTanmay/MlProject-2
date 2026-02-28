"""
Feedback Store - User Feedback Collection
Stores user feedback for intent classification improvement.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class FeedbackStore:
    """
    Stores and manages user feedback for system improvement.
    Feedback is used ONLY for retraining intent classifier.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize feedback store.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = Path(__file__).parent / "feedback.db"
        
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                predicted_intent TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                strategy_used TEXT NOT NULL,
                answer TEXT NOT NULL,
                user_feedback INTEGER NOT NULL,
                user_comment TEXT,
                was_correct INTEGER NOT NULL DEFAULT 0
            )
        """)
        
        # Create retraining log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                samples_used INTEGER NOT NULL,
                accuracy_before REAL,
                accuracy_after REAL,
                improvement REAL,
                notes TEXT
            )
        """)
        
        # Create routing log table (for Phase 7)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                active_intents TEXT NOT NULL,
                primary_intent TEXT NOT NULL,
                engine_chain TEXT NOT NULL,
                status TEXT NOT NULL,
                is_unsafe INTEGER NOT NULL DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✓ Feedback database initialized at {self.db_path}")
    
    def store_feedback(self, query: str, predicted_intent: str, 
                      predicted_confidence: float, strategy: str,
                      answer: str, user_feedback: int,
                      user_comment: str = "") -> bool:
        """
        Store user feedback.
        
        Args:
            query: Original query
            predicted_intent: Intent predicted by classifier
            predicted_confidence: Confidence score
            strategy: Strategy used (RETRIEVAL, ML, TRANSFORMER, RULE)
            answer: Answer provided
            user_feedback: 1 for positive (👍), -1 for negative (👎)
            user_comment: Optional user comment
            
        Returns:
            True if stored successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine if prediction was correct based on feedback
            was_correct = 1 if user_feedback > 0 else 0
            
            cursor.execute("""
                INSERT INTO feedback (
                    timestamp, query, predicted_intent, predicted_confidence,
                    strategy_used, answer, user_feedback, user_comment, was_correct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                query,
                predicted_intent,
                predicted_confidence,
                strategy,
                answer,
                user_feedback,
                user_comment,
                was_correct
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to store feedback: {e}")
            return False

    def store_routing_log(self, query: str, active_intents: List[str],
                          primary_intent: str, engine_chain: List[str],
                          status: str, is_unsafe: bool) -> bool:
        """
        Store routing decisions and unsafe attempts.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO routing_logs (
                    timestamp, query, active_intents, primary_intent,
                    engine_chain, status, is_unsafe
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                query,
                json.dumps(active_intents),
                primary_intent,
                json.dumps(engine_chain),
                status,
                1 if is_unsafe else 0
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"✗ Failed to store routing log: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored feedback.
        
        Returns:
            Dictionary with feedback statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total feedback count
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]
            
            # Positive vs negative
            cursor.execute("SELECT user_feedback, COUNT(*) FROM feedback GROUP BY user_feedback")
            feedback_distribution = dict(cursor.fetchall())
            
            # Accuracy by intent
            cursor.execute("""
                SELECT predicted_intent, 
                       SUM(was_correct) as correct,
                       COUNT(*) as total
                FROM feedback
                GROUP BY predicted_intent
            """)
            
            intent_accuracy = {}
            for intent, correct, total in cursor.fetchall():
                intent_accuracy[intent] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total > 0 else 0
                }
            
            # Recent feedback (last 10)
            cursor.execute("""
                SELECT timestamp, query, predicted_intent, user_feedback
                FROM feedback
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            recent_feedback = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_feedback": total_feedback,
                "positive_feedback": feedback_distribution.get(1, 0),
                "negative_feedback": feedback_distribution.get(-1, 0),
                "satisfaction_rate": feedback_distribution.get(1, 0) / total_feedback if total_feedback > 0 else 0,
                "intent_accuracy": intent_accuracy,
                "recent_feedback": recent_feedback
            }
            
        except Exception as e:
            print(f"✗ Failed to get feedback stats: {e}")
            return {}
    
    def get_training_data(self, min_confidence: float = 0.8,
                         only_correct: bool = True) -> List[Dict[str, Any]]:
        """
        Get feedback data suitable for retraining.
        
        Args:
            min_confidence: Minimum confidence threshold
            only_correct: Only include positive feedback
            
        Returns:
            List of training samples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if only_correct:
                cursor.execute("""
                    SELECT query, predicted_intent
                    FROM feedback
                    WHERE was_correct = 1 AND predicted_confidence >= ?
                """, (min_confidence,))
            else:
                cursor.execute("""
                    SELECT query, predicted_intent
                    FROM feedback
                    WHERE predicted_confidence >= ?
                """, (min_confidence,))
            
            samples = []
            for query, intent in cursor.fetchall():
                samples.append({
                    "query": query,
                    "intent": intent
                })
            
            conn.close()
            
            print(f"✓ Retrieved {len(samples)} training samples from feedback")
            return samples
            
        except Exception as e:
            print(f"✗ Failed to get training data: {e}")
            return []
    
    def log_retraining(self, samples_used: int, accuracy_before: float,
                      accuracy_after: float, notes: str = "") -> bool:
        """
        Log a retraining session.
        
        Args:
            samples_used: Number of samples used
            accuracy_before: Accuracy before retraining
            accuracy_after: Accuracy after retraining
            notes: Optional notes
            
        Returns:
            True if logged successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            improvement = accuracy_after - accuracy_before
            
            cursor.execute("""
                INSERT INTO retraining_log (
                    timestamp, samples_used, accuracy_before,
                    accuracy_after, improvement, notes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                samples_used,
                accuracy_before,
                accuracy_after,
                improvement,
                notes
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to log retraining: {e}")
            return False
    
    def get_retraining_history(self) -> List[Dict[str, Any]]:
        """
        Get history of retraining sessions.
        
        Returns:
            List of retraining sessions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, samples_used, accuracy_before,
                       accuracy_after, improvement, notes
                FROM retraining_log
                ORDER BY timestamp DESC
            """)
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "timestamp": row[0],
                    "samples_used": row[1],
                    "accuracy_before": row[2],
                    "accuracy_after": row[3],
                    "improvement": row[4],
                    "notes": row[5]
                })
            
            conn.close()
            return history
            
        except Exception as e:
            print(f"✗ Failed to get retraining history: {e}")
            return []
    
    def clear_feedback(self, older_than_days: int = None) -> int:
        """
        Clear old feedback data.
        
        Args:
            older_than_days: Clear feedback older than N days
            
        Returns:
            Number of records deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if older_than_days:
                from datetime import timedelta
                cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                cursor.execute("DELETE FROM feedback WHERE timestamp < ?", (cutoff_date,))
            else:
                cursor.execute("DELETE FROM feedback")
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"✓ Cleared {deleted} feedback records")
            return deleted
            
        except Exception as e:
            print(f"✗ Failed to clear feedback: {e}")
            return 0
