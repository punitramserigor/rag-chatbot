from flask import Flask, request, jsonify
import langchain.document_loaders
from model import RAG
from dotenv import load_dotenv
import langchain

langchain.debug=True

load_dotenv()

app = Flask(__name__)


@app.route("/load_db", methods=["POST"])
def load_db():
    data: dict = request.get_json()
    session_id = data.get("session_id")
    file_path = data.get("file_path")
    supported_formats = ["pptx", "docx", "pdf", "txt", "md"]
    file_extension = file_path.split(".")[-1]
    if file_path and session_id:
        if file_extension in supported_formats:
            # try:
                rag_model = RAG(session_id=session_id)
                rag_model.load_db(file_path)
                return jsonify({"success": "Database loaded successfully"}), 200
            # except Exception as e:
            #     return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    else:
        return jsonify({"error": "Missing required parameters"}), 400


@app.route("/doc_invoke", methods=["POST"])
def doc_invoke():
    data: dict = request.get_json()
    session_id = data.get("session_id")
    query = data.get("query")

    if session_id and query:
        # try:
            rag_model = RAG(session_id=session_id)
            answer = rag_model.invoke(query=query)
            return jsonify({"answer": answer}), 200
        # except Exception as e:
        #     return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Missing required parameters"}), 400


@app.route("/delete_db", methods=["POST"])
def delete_db():
    data: dict = request.get_json()
    session_id = data.get("session_id")

    if session_id:
        try:
            rag_model = RAG(session_id=session_id)
            rag_model.delete_db()
            return jsonify({"success": "Database deleted successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Missing required parameters"}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
