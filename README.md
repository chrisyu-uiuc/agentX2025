# DBInsight - AI-Powered Database Assistant



DBInsight is an intelligent chatbot that allows users to interact with their SQLite database using natural language. Powered by Google's Gemini AI, it translates conversational queries into SQL commands and presents results in an easy-to-understand format.

## ✨ Features

- ​**Natural Language Interface**: Ask questions about your database in plain English
- ​**AI-Powered SQL Generation**: Gemini AI translates questions into optimized SQL queries
- ​**Interactive Web UI**: Clean, responsive interface with quick action buttons
- ​**Database Management**: View schema, generate sample data, and backup your database
- ​**Conversation History**: Track all your queries and responses
- ​**Multi-Platform**: Web-based interface works on desktop and mobile

## 🛠️ Tech Stack

- ​**Backend**: Python (Flask)
- ​**Frontend**: HTML5, CSS3, JavaScript
- ​**AI**: Google Gemini API
- ​**Database**: SQLite
- ​**Deployment**: Docker-ready

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Node.js (for development)

### Installation
1. Installation & Setup
First, install the required dependencies using pip:

pip install flask flask-cors google-generativeai sqlite3 python-dotenv

2. Running the Application
Save the Python code as agent.py and run it:
python agent.py
This will start the Flask server at http://localhost:5000.

3. Accessing the Web UI
Open your browser and navigate to:
http://localhost:5000

# Authors

- ​**​[Chris YU]​**​ -[GitHub](https://github.com/chrisyu-uiuc)  


MIT License

Copyright (c) [2025] [Chris YU]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
