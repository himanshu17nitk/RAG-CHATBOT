Customer Support Chatbot

This is a customer support chatbot that can answer questions based on uploaded documents. It uses AI to understand and respond to user queries.

Setup Instructions

1. Create a Python virtual environment:
   python -m venv venv

2. Activate the virtual environment:
   On Windows: venv\Scripts\activate
   On Mac/Linux: source venv/bin/activate

3. Install required packages:
   pip install -r requirements.txt

4. Start the server:
   python main.py

The server will start on http://localhost:8000

Usage

To ask a question, use this curl command:
curl -X POST "http://localhost:8000/predict" -H "X-Client-Secret: usf123" -H "Content-Type: application/json" -d '{
       "query": "how to proceed for emergency appointments?",
       "session_id": "SESSION_ID_HERE",
       "user_id": "himanshu"
}'

To upload a document for training, use this curl command:
curl -X POST "http://localhost:8000/train" -H "X-Client-Secret: usf123" -F "user_id=himanshu" -F "file=@data/HealthBot.pdf"

Notes:
- Replace SESSION_ID_HERE with a unique session identifier
- Replace himanshu with your user ID
- The file path in the train command should point to your PDF file
- The client secret usf123 is required for authentication 