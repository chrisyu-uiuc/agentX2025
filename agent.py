import sqlite3
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from functools import wraps
import os
import random

YOUR_GEMINI_KEY = ""

class GeminiSQLiteChat:
    def __init__(self, db_path: str, gemini_api_key: str):
        """
        Initialize the chat bot with database and Gemini API connections
        
        Args:
            db_path: Path to SQLite database file
            gemini_api_key: Google Gemini API key
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Create chat history table if it doesn't exist
        self.setup_chat_history()
        
    def setup_chat_history(self):
        """Create a table to store chat history"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_message TEXT,
                bot_response TEXT,
                database_actions TEXT
            )
        ''')
        self.conn.commit()
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get complete database schema information"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema = {}
        for table in tables:
            table_name = table[0]
            
            # Get table info (columns, types, etc.)
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            
            # Get sample data (first 3 rows)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            sample_data = cursor.fetchall()
            
            schema[table_name] = {
                'columns': [dict(col) for col in columns],
                'foreign_keys': [dict(fk) for fk in foreign_keys],
                'row_count': row_count,
                'sample_data': [dict(row) for row in sample_data]
            }
        
        return schema
    
    def execute_sql_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                self.conn.commit()
                return [{'affected_rows': cursor.rowcount, 'success': True}]
            else:
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except sqlite3.Error as e:
            return [{'error': str(e), 'success': False}]
    
    def extract_sql_from_response(self, text: str) -> List[str]:
        """Extract SQL queries from Gemini's response"""
        # Look for SQL queries in code blocks or between SQL markers
        sql_patterns = [
            r'```sql\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'SQL:\s*(.*?)(?:\n|$)',
            r'QUERY:\s*(.*?)(?:\n|$)'
        ]
        
        queries = []
        for pattern in sql_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                # Clean up the query
                query = match.strip()
                if query and any(keyword in query.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE']):
                    queries.append(query)
        
        return queries
    
    def format_results_as_natural_text(self, results: List[Dict], query_type: str) -> str:
        """Format query results as natural, conversational text"""
        if not results:
            return "I didn't find any results for that query."
        
        # Handle error responses
        if any('error' in result for result in results):
            errors = [result['error'] for result in results if 'error' in result]
            return f"I encountered an issue while running that query: {errors[0]}. Could you please rephrase your request?"
        
        # Handle modification queries (INSERT, UPDATE, DELETE)
        if query_type.upper() in ['INSERT', 'UPDATE', 'DELETE']:
            affected_rows = sum(result.get('affected_rows', 0) for result in results if 'affected_rows' in result)
            if query_type.upper() == 'INSERT':
                if affected_rows == 1:
                    return "Perfect! I've successfully added the new record to your database."
                else:
                    return f"Great! I've added {affected_rows} new records to your database."
            elif query_type.upper() == 'UPDATE':
                if affected_rows == 1:
                    return "Done! I've updated that record with the new information."
                else:
                    return f"All set! I've updated {affected_rows} records with the changes you requested."
            elif query_type.upper() == 'DELETE':
                if affected_rows == 1:
                    return "The record has been successfully removed from your database."
                else:
                    return f"I've removed {affected_rows} records from your database as requested."
        
        # Handle SELECT queries - format as conversational text
        if len(results) == 1:
            record = results[0]
            response = "Here's what I found:<br><br>"
            for key, value in record.items():
                if value is not None:
                    field_name = key.replace('_', ' ').title()
                    response += f"<strong>{field_name}:</strong> {value}<br>"
            return response
        
        elif len(results) <= 5:
            response = f"I found {len(results)} records that match your request:<br><br>"
            for i, record in enumerate(results, 1):
                response += f"<strong>Record {i}:</strong><br>"
                for key, value in record.items():
                    if value is not None:
                        field_name = key.replace('_', ' ').title()
                        response += f"&nbsp;&nbsp;• <strong>{field_name}:</strong> {value}<br>"
                response += "<br>"
            return response
        
        else:
            response = f"I found {len(results)} records total. Here are the first 5:<br><br>"
            for i, record in enumerate(results[:5], 1):
                response += f"<strong>Record {i}:</strong><br>"
                for key, value in record.items():
                    if value is not None:
                        field_name = key.replace('_', ' ').title()
                        response += f"&nbsp;&nbsp;• <strong>{field_name}:</strong> {value}<br>"
                response += "<br>"
            response += f"<em>There are {len(results) - 5} more records. Would you like me to show you more specific results or filter them differently?</em>"
            return response
    
    def create_context_prompt(self, user_message: str) -> str:
        """Create a comprehensive prompt with database context"""
        schema = self.get_database_schema()
        
        context = f"""
You are a helpful, friendly AI database assistant. Your job is to help users interact with their SQLite database using natural conversation.

CRITICAL RESPONSE RULES:
1. ONLY provide SQL queries in ```sql code blocks when the user asks for database operations
2. Do NOT include any conversational text alongside SQL queries
3. If you need to query the database, ONLY return the SQL query in a code block
4. Do NOT provide explanations or commentary when returning SQL queries
5. Let the system handle formatting the results naturally

CURRENT DATABASE SCHEMA:
"""
        
        for table_name, info in schema.items():
            if table_name == 'chat_history':  # Skip internal chat history table
                continue
                
            context += f"\nTable: {table_name} (contains {info['row_count']} records)\n"
            context += "Columns:\n"
            for col in info['columns']:
                nullable = "optional" if not col['notnull'] else "required"
                primary = " (Primary Key)" if col['pk'] else ""
                context += f"  - {col['name']}: {col['type']} ({nullable}){primary}\n"
            
            # Add foreign key information
            if info['foreign_keys']:
                context += "Foreign Keys:\n"
                for fk in info['foreign_keys']:
                    context += f"  - {fk['from']} references {fk['table']}({fk['to']})\n"
            
            if info['sample_data']:
                context += "Sample records:\n"
                for i, row in enumerate(info['sample_data'][:2]):
                    sample_dict = dict(row)
                    # Clean up the display
                    clean_sample = {k: v for k, v in sample_dict.items() if v is not None}
                    context += f"  Example {i+1}: {clean_sample}\n"
            context += "\n"
        
        context += f"""
RECENT CONVERSATION:
{self.get_recent_history()}

USER REQUEST: {user_message}

Instructions:
- If this requires a database query, respond ONLY with the SQL in ```sql blocks
- If this is a general question, respond conversationally
- Do NOT mix SQL and conversational text
- Use proper JOIN syntax when querying related tables
- Consider foreign key relationships when generating queries
"""
        
        return context
    
    def get_recent_history(self, limit: int = 3) -> str:
        """Get recent conversation history"""
        if not self.conversation_history:
            return "This is the start of our conversation."
        
        history = ""
        for entry in self.conversation_history[-limit:]:
            # Truncate long responses for context
            user_msg = entry['user'][:100] + "..." if len(entry['user']) > 100 else entry['user']
            bot_msg = entry['bot'][:150] + "..." if len(entry['bot']) > 150 else entry['bot']
            history += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
        
        return history
    
    def save_chat_to_db(self, user_message: str, bot_response: str, db_actions: List[str]):
        """Save conversation to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (timestamp, user_message, bot_response, database_actions)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user_message,
            bot_response,
            json.dumps(db_actions)
        ))
        self.conn.commit()
    def chat(self, user_message: str) -> str:
            """Enhanced chat function - LLM-first approach"""
            try:
                # Create streamlined prompt
                prompt = self.create_context_prompt(user_message)
                
                # Get LLM response
                response = self.model.generate_content(prompt)
                llm_response = response.text.strip()
                
                # Check if response contains SQL - if so, execute it
                sql_queries = self.extract_sql_from_response(llm_response)
                
                if sql_queries:
                    # Execute the SQL and return formatted results
                    db_actions = []
                    final_response = ""
                    successful_executions = 0
                    
                    for query in sql_queries:  # Try all queries
                        result = self.execute_sql_query(query)
                        db_actions.append(query)
                        
                        if result and not any('error' in r for r in result):
                            successful_executions += 1
                            # For INSERT/UPDATE/DELETE operations, show success message
                            if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                                continue  # Don't format these, just count as successful
                            else:
                                query_type = query.strip().upper().split()[0]
                                final_response = self.format_results_as_natural_text(result, query_type)
                    
                    # If we had successful operations but no SELECT result to show
                    if successful_executions > 0 and not final_response:
                        if any(q.strip().upper().startswith('INSERT') for q in sql_queries):
                            final_response = "Perfect! I've successfully created the new order and updated everything as requested."
                        elif any(q.strip().upper().startswith('UPDATE') for q in sql_queries):
                            final_response = "Done! I've updated the records as requested."
                        else:
                            final_response = "Operation completed successfully!"
                    
                    if not final_response:
                        final_response = "I couldn't execute that query. Please try rephrasing your request."
                else:
                    # Check for database-related keywords and try to extract queries
                    if any(word in user_message.lower() for word in ['show', 'find', 'get', 'list', 'what', 'who', 'purchase', 'order', 'user', 'product']):
                        # Force LLM to generate SQL
                        sql_prompt = f"Generate SQL query for: {user_message}\nDatabase schema: {self.get_database_schema()}\nReturn only SQL:"
                        sql_response = self.model.generate_content(sql_prompt)
                        sql_queries = self.extract_sql_from_response(sql_response.text)
                        
                        if sql_queries:
                            result = self.execute_sql_query(sql_queries[0])
                            if result and not any('error' in r for r in result):
                                query_type = sql_queries[0].strip().upper().split()[0]
                                final_response = self.format_results_as_natural_text(result, query_type)
                            else:
                                final_response = llm_response
                        else:
                            final_response = llm_response
                    else:
                        final_response = llm_response
                    db_actions = []
                
                # Clean up formatting and remove SQL code blocks from response
                final_response = re.sub(r'```sql.*?```', '', final_response, flags=re.DOTALL)
                final_response = re.sub(r'```.*?```', '', final_response, flags=re.DOTALL)
                final_response = re.sub(r'\n\s*\n', '<br><br>', final_response)
                final_response = re.sub(r'\n', '<br>', final_response)
                final_response = final_response.strip()
                
                # Update conversation history (keep last 5 only)
                self.conversation_history.append({
                    'user': user_message,
                    'bot': final_response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only recent history to save memory
                if len(self.conversation_history) > 5:
                    self.conversation_history = self.conversation_history[-5:]
                
                # Save to database
                self.save_chat_to_db(user_message, final_response, db_actions)
                
                return final_response
                
            except Exception as e:
                return f"Hi there! I'm having a small technical issue. Try asking me about your database - like 'show me all users' or 'what products do we have?'"
                
    def show_database_status(self) -> Dict[str, Any]:
        """Show current database status"""
        schema = self.get_database_schema()
        
        table_count = 0
        total_records = 0
        tables_info = []
        
        for table_name, info in schema.items():
            if table_name == 'chat_history':  # Skip internal table
                continue
                
            table_count += 1
            total_records += info['row_count']
            
            # Show column names
            col_names = [col['name'] for col in info['columns']]
            
            # Show foreign key relationships
            relationships = []
            for fk in info['foreign_keys']:
                relationships.append(f"{fk['from']} -> {fk['table']}.{fk['to']}")
            
            table_info = {
                'name': table_name,
                'record_count': info['row_count'],
                'column_count': len(info['columns']),
                'columns': col_names,
                'foreign_keys': relationships
            }
            
            if info['sample_data'] and info['row_count'] > 0:
                latest = dict(info['sample_data'][0])
                # Show a simplified version of the latest record
                key_field = next((k for k in ['name', 'title', 'id'] if k in latest), list(latest.keys())[0])
                table_info['latest_entry'] = latest.get(key_field, 'N/A')
            
            tables_info.append(table_info)
        
        return {
            'summary': {
                'table_count': table_count,
                'total_records': total_records
            },
            'tables': tables_info
        }
    
    def get_chat_history(self, limit: int = 10) -> List[Dict]:
        """Get chat history from database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, user_message, bot_response, database_actions
            FROM chat_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def generate_sample_data_with_gemini(self, table_name: str, num_records: int = 100) -> List[str]:
        """Use Gemini to generate realistic sample data for a table"""
        try:
            # Get table schema
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            
            # Skip if no columns or if it's the chat_history table
            if not columns or table_name == 'chat_history':
                return []
            
            # Build schema description for Gemini
            schema_desc = f"Table: {table_name}\nColumns:\n"
            insert_columns = []
            fk_constraints = {}
            
            for col in columns:
                col_dict = dict(col)
                # Skip auto-increment primary keys and timestamp fields with defaults
                if (col_dict['pk'] and 'INTEGER' in col_dict['type'].upper()) or \
                   (col_dict['dflt_value'] and 'CURRENT_TIMESTAMP' in str(col_dict['dflt_value'])):
                    continue
                    
                insert_columns.append(col_dict['name'])
                nullable = "optional" if not col_dict['notnull'] else "required"
                schema_desc += f"  - {col_dict['name']}: {col_dict['type']} ({nullable})\n"
            
            # Add foreign key information
            if foreign_keys:
                schema_desc += "Foreign Key Constraints:\n"
                for fk in foreign_keys:
                    fk_dict = dict(fk)
                    schema_desc += f"  - {fk_dict['from']} references {fk_dict['table']}({fk_dict['to']})\n"
                    fk_constraints[fk_dict['from']] = {
                        'table': fk_dict['table'],
                        'column': fk_dict['to']
                    }
            
            if not insert_columns:
                return []
            
            # Get available foreign key values
            fk_values = {}
            for col_name, fk_info in fk_constraints.items():
                try:
                    cursor.execute(f"SELECT {fk_info['column']} FROM {fk_info['table']} LIMIT 50")
                    values = [row[0] for row in cursor.fetchall()]
                    if values:
                        fk_values[col_name] = values
                except:
                    pass
            
            # Create prompt for Gemini
            prompt = f"""
Generate {num_records} realistic sample records for this database table. Return ONLY valid SQL INSERT statements, one per line.

{schema_desc}

Foreign Key Values Available:
{json.dumps(fk_values, indent=2)}

Requirements:
1. Generate realistic, diverse data that makes sense for each field
2. Use proper SQL INSERT syntax: INSERT INTO {table_name} ({', '.join(insert_columns)}) VALUES (...)
3. Use single quotes for text values, proper numbers for numeric fields
4. For foreign key fields, ONLY use values from the available foreign key values listed above
5. Make data realistic and varied (different names, realistic prices, varied categories, etc.)
6. Return exactly {num_records} INSERT statements
7. Do not include any explanations, comments, or code blocks - just the SQL statements

Example format:
INSERT INTO {table_name} ({', '.join(insert_columns[:2])}) VALUES ('example', 123);

Generate {num_records} diverse, realistic records now:
"""
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            sql_statements = response.text.strip().split('\n')
            
            # Clean and validate SQL statements
            clean_statements = []
            for statement in sql_statements:
                statement = statement.strip()
                if statement and statement.upper().startswith('INSERT'):
                    # Remove any trailing semicolon for consistency
                    if statement.endswith(';'):
                        statement = statement[:-1]
                    clean_statements.append(statement)
            
            return clean_statements[:num_records]  # Ensure we don't exceed requested number
            
        except Exception as e:
            print(f"Error generating sample data for {table_name}: {e}")
            return []
    
    def populate_table_with_sample_data(self, table_name: str, num_records: int = 100) -> Dict[str, Any]:
        """Populate a specific table with sample data using Gemini"""
        try:
            # Generate SQL statements using Gemini
            insert_statements = self.generate_sample_data_with_gemini(table_name, num_records)
            
            if not insert_statements:
                return {
                    'success': False,
                    'message': f'Could not generate sample data for table: {table_name}',
                    'records_added': 0
                }
            
            # Execute the INSERT statements
            successful_inserts = 0
            failed_inserts = 0
            errors = []
            
            for statement in insert_statements:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(statement)
                    successful_inserts += 1
                except sqlite3.Error as e:
                    failed_inserts += 1
                    errors.append(f"Error with statement '{statement[:50]}...': {str(e)}")
                    if len(errors) > 5:  # Limit error reporting
                        errors.append("... (more errors)")
                        break
            
            self.conn.commit()
            
            return {
                'success': True,
                'table_name': table_name,
                'records_requested': num_records,
                'records_added': successful_inserts,
                'records_failed': failed_inserts,
                'errors': errors[:5]  # Limit to first 5 errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error populating table {table_name}: {str(e)}',
                'records_added': 0
            }
    
    def populate_all_tables_with_sample_data(self, num_records: int = 100) -> Dict[str, Any]:
        """Populate all tables with sample data using Gemini - respects foreign key dependencies"""
        try:
            # Get all tables
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = cursor.fetchall()
            
            # Define table dependency order (parent tables first)
            table_order = [
                'users', 'categories', 'suppliers', 'warehouses', 'brands',
                'products', 'customers', 'addresses', 'payment_methods',
                'orders', 'order_items', 'inventory', 'inventory_movements',
                'product_reviews', 'customer_addresses', 'wishlists', 'wishlist_items',
                'promotions', 'promotion_products', 'shopping_carts', 'cart_items'
            ]
            
            # Get existing tables and order them
            existing_tables = [table[0] for table in tables if table[0] != 'chat_history']
            ordered_tables = []
            
            # Add tables in dependency order
            for table in table_order:
                if table in existing_tables:
                    ordered_tables.append(table)
            
            # Add any remaining tables not in the predefined order
            for table in existing_tables:
                if table not in ordered_tables:
                    ordered_tables.append(table)
            
            results = {}
            total_added = 0
            
            for table_name in ordered_tables:
                print(f"🎲 Generating sample data for table: {table_name}")
                result = self.populate_table_with_sample_data(table_name, num_records)
                results[table_name] = result
                total_added += result.get('records_added', 0)
            
            return {
                'success': True,
                'message': f'Sample data generation completed',
                'total_records_added': total_added,
                'tables_processed': len(results),
                'processing_order': ordered_tables,
                'details': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error during bulk population: {str(e)}',
                'total_records_added': 0
            }

    def close(self):
        """Close database connection"""
        self.conn.close()


# Flask REST API
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chat bot instance
chat_bot = None

def handle_errors(f):
    """Decorator to handle errors consistently"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e),
                'message': 'An internal error occurred'
            }), 500
    return decorated_function

@app.route('/api/health', methods=['GET'])
@handle_errors
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'message': 'Gemini SQLite Chat API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
@handle_errors
def chat_endpoint():
    """Main chat endpoint"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing message in request body'
        }), 400
    
    user_message = data['message'].strip()
    
    if not user_message:
        return jsonify({
            'success': False,
            'error': 'Message cannot be empty'
        }), 400
    
    # Process the message
    response = chat_bot.chat(user_message)
    
    return jsonify({
        'success': True,
        'response': response,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/database/status', methods=['GET'])
@handle_errors
def database_status():
    """Get database status"""
    status = chat_bot.show_database_status()
    
    return jsonify({
        'success': True,
        'data': status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/database/schema', methods=['GET'])
@handle_errors
def get_schema():
    """Get database schema"""
    schema = chat_bot.get_database_schema()
    
    # Filter out chat_history table for API responses
    filtered_schema = {k: v for k, v in schema.items() if k != 'chat_history'}
    
    return jsonify({
        'success': True,
        'data': filtered_schema,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat/history', methods=['GET'])
@handle_errors
def get_chat_history():
    """Get chat history"""
    limit = request.args.get('limit', 10, type=int)
    
    if limit < 1:
        limit = 10
    elif limit > 100:
        limit = 100
    
    history = chat_bot.get_chat_history(limit)
    
    return jsonify({
        'success': True,
        'data': history,
        'count': len(history),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/database/query', methods=['POST'])
@handle_errors
def direct_query():
    """Direct SQL query endpoint (use with caution)"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({
            'success': False,
            'error': 'Missing query in request body'
        }), 400
    
    query = data['query'].strip()
    params = data.get('params', ())
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Query cannot be empty'
        }), 400
    
    # Basic security check - only allow SELECT statements for now
    if not query.upper().strip().startswith('SELECT'):
        return jsonify({
            'success': False,
            'error': 'Only SELECT queries are allowed through this endpoint'
        }), 400
    
    results = chat_bot.execute_sql_query(query, params)
    
    return jsonify({
        'success': True,
        'data': results,
        'query': query,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/database/populate', methods=['POST'])
@handle_errors
def populate_database():
    """Populate database with sample data using Gemini AI"""
    data = request.get_json() or {}
    
    # Get parameters
    table_name = data.get('table_name')  # Optional: specific table
    num_records = data.get('num_records', 100)  # Default: 100 records
    
    # Validate num_records
    if num_records < 1 or num_records > 1000:
        return jsonify({
            'success': False,
            'error': 'num_records must be between 1 and 1000'
        }), 400
    
    if table_name:
        # Populate specific table
        result = chat_bot.populate_table_with_sample_data(table_name, num_records)
    else:
        # Populate all tables
        result = chat_bot.populate_all_tables_with_sample_data(num_records)
    
    if result['success']:
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'success': False,
            'error': result.get('message', 'Failed to populate database'),
            'data': result,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/database/clear', methods=['POST'])
@handle_errors
def clear_database():
    """Clear all data from database tables (except chat_history)"""
    data = request.get_json() or {}
    table_name = data.get('table_name')  # Optional: specific table
    
    try:
        cursor = chat_bot.conn.cursor()
        
        if table_name:
            # Clear specific table
            if table_name == 'chat_history':
                return jsonify({
                    'success': False,
                    'error': 'Cannot clear chat_history table'
                }), 400
            
            cursor.execute(f"DELETE FROM {table_name}")
            affected_rows = cursor.rowcount
            chat_bot.conn.commit()
            
            return jsonify({
                'success': True,
                'message': f'Cleared {affected_rows} records from table: {table_name}',
                'records_deleted': affected_rows,
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Clear all tables except chat_history
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = cursor.fetchall()
            
            total_deleted = 0
            cleared_tables = []
            
            for table in tables:
                table_name = table[0]
                if table_name != 'chat_history':
                    cursor.execute(f"DELETE FROM {table_name}")
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    cleared_tables.append({
                        'table': table_name,
                        'records_deleted': deleted
                    })
            
            chat_bot.conn.commit()
            
            return jsonify({
                'success': True,
                'message': f'Cleared all tables. Total records deleted: {total_deleted}',
                'total_records_deleted': total_deleted,
                'tables_cleared': cleared_tables,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to clear database'
        }), 500

@app.route('/api/database/backup', methods=['POST'])
@handle_errors
def backup_database():
    """Create a backup of the database"""
    data = request.get_json() or {}
    backup_name = data.get('backup_name', f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    try:
        import shutil
        backup_path = f"{backup_name}.db"
        shutil.copy2(chat_bot.db_path, backup_path)
        
        return jsonify({
            'success': True,
            'message': f'Database backed up successfully',
            'backup_file': backup_path,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to backup database'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

def initialize_database(chat_bot_instance):
    """Initialize the database with sample data"""
    logger.info("Setting up sample database...")
    
    # Users table
    chat_bot_instance.execute_sql_query('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Products table
    chat_bot_instance.execute_sql_query('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT,
            stock INTEGER DEFAULT 0,
            description TEXT
        )
    ''')
    
    # Orders table
    chat_bot_instance.execute_sql_query('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            total_amount REAL,
            status TEXT DEFAULT 'pending',
            order_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Order items table
    chat_bot_instance.execute_sql_query('''
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER DEFAULT 1,
            unit_price REAL,
            FOREIGN KEY (order_id) REFERENCES orders (id),
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
    ''')
    
    # Categories table
    chat_bot_instance.execute_sql_query('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample data if tables are empty
    cursor = chat_bot_instance.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        # Sample users
        chat_bot_instance.execute_sql_query('''
            INSERT INTO users (name, email, role) VALUES 
            ('Alice Johnson', 'alice@example.com', 'admin'),
            ('Bob Smith', 'bob@example.com', 'user'),
            ('Carol Davis', 'carol@example.com', 'user'),
            ('David Wilson', 'david@example.com', 'user'),
            ('Emma Brown', 'emma@example.com', 'moderator')
        ''')
        
        # Sample categories
        chat_bot_instance.execute_sql_query('''
            INSERT INTO categories (name, description) VALUES 
            ('Electronics', 'Electronic devices and gadgets'),
            ('Furniture', 'Home and office furniture'),
            ('Books', 'Books and educational materials'),
            ('Appliances', 'Kitchen and home appliances'),
            ('Sports', 'Sports equipment and accessories')
        ''')
        
        # Sample products
        chat_bot_instance.execute_sql_query('''
            INSERT INTO products (name, price, category, stock, description) VALUES 
            ('Gaming Laptop', 1299.99, 'Electronics', 5, 'High-performance gaming laptop'),
            ('Office Chair', 249.99, 'Furniture', 12, 'Ergonomic office chair'),
            ('Python Programming Book', 39.99, 'Books', 25, 'Learn Python programming'),
            ('Wireless Mouse', 29.99, 'Electronics', 50, 'Bluetooth wireless mouse'),
            ('Coffee Maker', 89.99, 'Appliances', 8, 'Automatic drip coffee maker'),
            ('Standing Desk', 399.99, 'Furniture', 6, 'Height-adjustable standing desk'),
            ('Smartphone', 799.99, 'Electronics', 15, 'Latest smartphone model'),
            ('Basketball', 24.99, 'Sports', 30, 'Professional basketball'),
            ('Blender', 129.99, 'Appliances', 10, 'High-speed blender'),
            ('JavaScript Guide', 34.99, 'Books', 20, 'Complete JavaScript guide')
        ''')
        
        # Sample orders
        chat_bot_instance.execute_sql_query('''
            INSERT INTO orders (user_id, total_amount, status) VALUES 
            (1, 1329.98, 'completed'),
            (2, 279.98, 'pending'),
            (3, 39.99, 'shipped'),
            (4, 824.98, 'completed'),
            (5, 154.98, 'processing')
        ''')
        
        # Sample order items
        chat_bot_instance.execute_sql_query('''
            INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES 
            (1, 1, 1, 1299.99),
            (1, 4, 1, 29.99),
            (2, 2, 1, 249.99),
            (2, 4, 1, 29.99),
            (3, 3, 1, 39.99),
            (4, 7, 1, 799.99),
            (4, 8, 1, 24.99),
            (5, 5, 1, 89.99),
            (5, 9, 1, 129.99)
        ''')
    
    logger.info("Database setup complete!")

def create_app():
    """Application factory"""
    global chat_bot
    
    # Configuration
    DB_PATH = os.getenv('DB_PATH', 'chatbot.db')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', YOUR_GEMINI_KEY)
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your-gemini-api-key-here':
        logger.error("Please set your Gemini API key in the GEMINI_API_KEY environment variable")
        raise ValueError("Gemini API key not configured")
    
    # Initialize the chat bot
    logger.info("Initializing Gemini SQLite Chat Bot...")
    chat_bot = GeminiSQLiteChat(DB_PATH, GEMINI_API_KEY)
    
    # Set up database
    initialize_database(chat_bot)
    
    logger.info("✅ API Server ready!")
    
    return app

def print_startup_info():
    """Print startup information"""
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    print("\n" + "="*60)
    print("🚀 Gemini SQLite Chat REST API")
    print("="*60)
    print(f"📡 Server: http://{HOST}:{PORT}")
    print("\n📋 Available Endpoints:")
    print("  • GET  /api/health              - Health check")
    print("  • POST /api/chat                - Chat with the bot")
    print("  • GET  /api/database/status     - Database overview")
    print("  • GET  /api/database/schema     - Database schema")
    print("  • GET  /api/chat/history        - Chat history")
    print("  • POST /api/database/query      - Direct SQL queries")
    print("  • POST /api/database/populate   - Populate with sample data")
    print("  • POST /api/database/clear      - Clear database tables")
    print("  • POST /api/database/backup     - Backup database")
    print("\n💡 Example requests:")
    print("  # Chat with the bot")
    print("  curl -X POST http://localhost:5000/api/chat \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"message\": \"Show me all users\"}'")
    print("")
    print("  # Get database status")
    print("  curl -X GET http://localhost:5000/api/database/status")
    print("")
    print("  # Populate database with sample data")
    print("  curl -X POST http://localhost:5000/api/database/populate \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"num_records\": 50}'")
    print("")
    print("  # Clear all data")
    print("  curl -X POST http://localhost:5000/api/database/clear")
    print("\n🔧 Environment Variables:")
    print("  • GEMINI_API_KEY - Your Gemini API key (Required)")
    print("  • DB_PATH - Database file path (default: chatbot.db)")
    print("  • HOST - Server host (default: 0.0.0.0)")
    print("  • PORT - Server port (default: 5000)")
    print("  • DEBUG - Debug mode (default: False)")
    print("="*60 + "\n")

if __name__ == '__main__':
    try:
        app = create_app()
        
        # Get configuration from environment
        HOST = os.getenv('HOST', '0.0.0.0')
        PORT = int(os.getenv('PORT', 5000))
        DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        
        print_startup_info()
        
        app.run(host=HOST, port=PORT, debug=DEBUG)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print("\n🔧 Setup checklist:")
        print("1. Install required packages:")
        print("   pip install flask flask-cors google-generativeai")
        print("2. Set your Gemini API key:")
        print("   export GEMINI_API_KEY='your-actual-api-key'")
        print("3. Get your API key from: https://makersuite.google.com/app/apikey")
        
    finally:
        if chat_bot:
            chat_bot.close()
            logger.info("🔒 Database connection closed.")