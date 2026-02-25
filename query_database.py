"""
SQLite Database Query Helper
Easy way to view all data in the feedback database
"""

import sqlite3
from pathlib import Path
from datetime import datetime

class DatabaseViewer:
    def __init__(self, db_path='feedback/feedback.db'):
        self.db_path = db_path
        
    def view_all_feedback(self):
        """View all feedback records"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM feedback ORDER BY id DESC')
            rows = cursor.fetchall()
            
            print('\n' + '=' * 100)
            print(f'ALL FEEDBACK RECORDS ({len(rows)} total)')
            print('=' * 100)
            
            if not rows:
                print("No feedback records yet.")
                return
            
            for row in rows:
                print(f'\n[ID: {row[0]}] {row[1]}')
                print(f'  Query: {row[2]}')
                print(f'  Intent: {row[3]} (Confidence: {row[4]:.2f})')
                print(f'  Strategy: {row[5]}')
                print(f'  Answer: {row[6][:100]}...' if len(str(row[6])) > 100 else f'  Answer: {row[6]}')
                print(f'  Feedback: {row[7]} | Was Correct: {row[9]} | Comment: {row[8]}')
    
    def view_statistics(self):
        """View database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total records
            cursor.execute('SELECT COUNT(*) FROM feedback')
            total = cursor.fetchone()[0]
            
            if total == 0:
                print("\nNo feedback records yet.")
                return
            
            # Positive/Negative
            cursor.execute('SELECT SUM(CASE WHEN user_feedback = 1 THEN 1 ELSE 0 END) FROM feedback')
            positive = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(CASE WHEN user_feedback = -1 THEN 1 ELSE 0 END) FROM feedback')
            negative = cursor.fetchone()[0] or 0
            
            # By intent
            cursor.execute('''
                SELECT predicted_intent, COUNT(*), 
                       SUM(was_correct),
                       ROUND(100.0 * SUM(was_correct) / COUNT(*), 2)
                FROM feedback
                GROUP BY predicted_intent
                ORDER BY COUNT(*) DESC
            ''')
            intent_stats = cursor.fetchall()
            
            # By strategy
            cursor.execute('''
                SELECT strategy_used, COUNT(*), 
                       SUM(was_correct),
                       ROUND(100.0 * SUM(was_correct) / COUNT(*), 2)
                FROM feedback
                GROUP BY strategy_used
                ORDER BY COUNT(*) DESC
            ''')
            strategy_stats = cursor.fetchall()
            
            print('\n' + '=' * 80)
            print('DATABASE STATISTICS')
            print('=' * 80)
            print(f'\nTotal Records: {total}')
            print(f'Positive Feedback: {positive} ({(positive/total*100):.1f}%)')
            print(f'Negative Feedback: {negative} ({(negative/total*100):.1f}%)')
            
            if total > 0:
                satisfaction = (positive / total) * 100
                print(f'Satisfaction Rate: {satisfaction:.2f}%')
            
            print('\n📊 Accuracy by Intent:')
            print('-' * 70)
            for intent, count, correct, accuracy in intent_stats:
                print(f'  {intent:15} | Count: {count:3} | Correct: {correct:3} | Accuracy: {accuracy}%')
            
            print('\n📊 Accuracy by Strategy:')
            print('-' * 70)
            for strategy, count, correct, accuracy in strategy_stats:
                print(f'  {strategy:15} | Count: {count:3} | Correct: {correct:3} | Accuracy: {accuracy}%')
    
    def view_negative_feedback(self):
        """View only negative feedback"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, query, answer, user_comment
                FROM feedback
                WHERE user_feedback = -1
                ORDER BY id DESC
            ''')
            rows = cursor.fetchall()
            
            print('\n' + '=' * 100)
            print(f'NEGATIVE FEEDBACK ({len(rows)} records)')
            print('=' * 100)
            
            if not rows:
                print("No negative feedback records.")
                return
            
            for row in rows:
                print(f'\n[ID: {row[0]}] {row[1]}')
                print(f'Query: {row[2]}')
                print(f'Answer: {row[3]}')
                print(f'Comment: {row[4]}')
    
    def view_retraining_history(self):
        """View model retraining history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM retraining_log ORDER BY id DESC')
            rows = cursor.fetchall()
            
            print('\n' + '=' * 100)
            print(f'RETRAINING HISTORY ({len(rows)} retrainings)')
            print('=' * 100)
            
            if not rows:
                print("No retraining history yet.")
                return
            
            for row in rows:
                print(f'\n[ID: {row[0]}] {row[1]}')
                print(f'  Samples Used: {row[2]}')
                print(f'  Accuracy: {row[3]:.4f} → {row[4]:.4f}')
                print(f'  Improvement: +{row[5]:.4f}')
                print(f'  Notes: {row[6]}')
    
    def view_misclassified(self):
        """View only misclassified queries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, query, predicted_intent, strategy_used, user_comment
                FROM feedback
                WHERE was_correct = 0
                ORDER BY id DESC
            ''')
            rows = cursor.fetchall()
            
            print('\n' + '=' * 100)
            print(f'MISCLASSIFIED QUERIES ({len(rows)} records)')
            print('=' * 100)
            
            if not rows:
                print("No misclassified queries.")
                return
            
            for row in rows:
                print(f'\n[ID: {row[0]}] {row[1]}')
                print(f'Query: {row[2]}')
                print(f'Predicted Intent: {row[3]}')
                print(f'Strategy Used: {row[4]}')
                print(f'Comment: {row[5]}')

def main():
    viewer = DatabaseViewer()
    
    while True:
        print("\n" + "=" * 60)
        print("🗄️  DATABASE QUERY TOOL")
        print("=" * 60)
        print("\n1. View all feedback")
        print("2. View statistics")
        print("3. View negative feedback only")
        print("4. View misclassified queries")
        print("5. View retraining history")
        print("6. Exit\n")
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            viewer.view_all_feedback()
        elif choice == '2':
            viewer.view_statistics()
        elif choice == '3':
            viewer.view_negative_feedback()
        elif choice == '4':
            viewer.view_misclassified()
        elif choice == '5':
            viewer.view_retraining_history()
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid option. Please select 1-6.")

if __name__ == "__main__":
    main()
