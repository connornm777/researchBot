from flask import Flask, request, jsonify
from data_manager import DataManager

app = Flask(__name__)

# Initialize DataManager without modifying files
manager = DataManager()

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    top_k = int(request.args.get('top_k', 5))
    results = manager.search(query, top_k=top_k)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
