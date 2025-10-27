# # from flask import Flask, request, jsonify
# # import pandas as pd
# # from churn import run_pipeline  # import your pipeline function

# # app = Flask(__name__)

# # @app.route("/predict", methods=["POST"])
# # # @app.route("/predict", methods=["POST"])
# # def predict():
# #     """
# #     Accepts:
# #     1) CSV file in 'file'
# #     2) JSON object in request body (single row or list of rows)
# #     """
# #     try:
# #         import pandas as pd

# #         # 1) CSV file
# #         if 'file' in request.files:
# #             file = request.files['file']
# #             df = pd.read_csv(file)

# #         # 2) JSON object
# #         else:
# #             data = request.get_json()  # must call the method
# #             if data is None:
# #                 return jsonify({"error": "No data provided"}), 400
# #             # if single row dict, convert to list of dicts
# #             if isinstance(data, dict):
# #                 data = [data]
# #             df = pd.DataFrame(data)

# #         # Run your pipeline
# #         results = run_pipeline(df, test_size=0.2)
# #         return jsonify({
# #             "best_features": results['best_features'],
# #             "baseline_metrics": results['baseline_metrics'],
# #             "final_metrics": results['final_metrics']
# #         })
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # # def predict():
# # #     """
# # #     Expects JSON body with key 'data' containing CSV text or file path.
# # #     """
# # #     try:
# # #         # Get uploaded CSV
# # #         if 'file' in request.files:
# # #             file = request.files['file']
# # #             df = pd.read_csv(file)
# # #         elif 'data' in request.json:
# # #             from io import StringIO
# # #             csv_data = request.json['data']
# # #             df = pd.read_csv(StringIO(csv_data))
# # #         else:
# # #             return jsonify({"error": "No data provided"}), 400

# # #         # Run your pipeline
# # #         results = run_pipeline(df, test_size=0.2)
# # #         return jsonify({
# # #             "best_features": results['best_features'],
# # #             "baseline_metrics": results['baseline_metrics'],
# # #             "final_metrics": results['final_metrics']
# # #         })
# # #     except Exception as e:
# # #         return jsonify({"error": str(e)}), 500

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=5000)
# # ____________________________________________________________________________________________________________________________
# from flask import Flask, request, jsonify
# import pandas as pd
# from churn import run_pipeline  # import your existing pipeline

# app = Flask(__name__)

# # Columns that should be dropped before feeding the pipeline
# IRRELEVANT_COLS = ['CustomerID', 'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason']

# @app.route("/")
# def home():
#     return "Churn prediction API is running. Use POST /predict with JSON or CSV file."

# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Accepts:
#     1) CSV file in 'file' (multipart/form-data)
#     2) JSON object (single row or list of rows)
#     Returns JSON with best_features, baseline_metrics, final_metrics
#     """
#     try:
#         # -------------------------
#         # 1) Handle CSV file upload
#         # -------------------------
#         if 'file' in request.files:
#             file = request.files['file']
#             df = pd.read_csv(file)

#         # -------------------------
#         # 2) Handle JSON input
#         # -------------------------
#         else:
#             data = request.get_json()
#             if data is None:
#                 return jsonify({"error": "No data provided"}), 400

#             # convert single row dict to list
#             if isinstance(data, dict):
#                 data = [data]

#             if not isinstance(data, list):
#                 return jsonify({"error": "JSON must be dict or list of dicts"}), 400

#             df = pd.DataFrame(data)

#         # -------------------------
#         # 3) Drop irrelevant columns
#         # -------------------------
#         df = df.drop(columns=[c for c in IRRELEVANT_COLS if c in df.columns], errors='ignore')

#         # -------------------------
#         # 4) Run pipeline
#         # -------------------------
#         # For single row, set test_size=0.5 to avoid errors in train_test_split
#         test_size = 0.2 if len(df) > 1 else 0.5
#         # results = run_pipeline(df, test_size=test_size)
#         results = run_pipeline(df, test_size=test_size, show_plots=False)


#         # -------------------------
#         # 5) Return JSON response
#         # -------------------------
#         return jsonify({
#             "best_features": results['best_features'],
#             "baseline_metrics": results['baseline_metrics'],
#             "final_metrics": results['final_metrics']
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
# __________________________________________________________________________________________________________________________________________
import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from churn import run_pipeline
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

def make_json_safe(obj):
    """Recursively convert numpy/pandas/shap/LabelEncoder objects to JSON-safe Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series, list, tuple)):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, LabelEncoder):
        return {"classes": obj.classes_.tolist()}
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    else:
        return obj


@app.route("/predict", methods=["POST"])
def predict():
    print("DEBUG — request.files:", request.files)
    print("DEBUG — request.form:", request.form)
    print("DEBUG — request.content_type:", request.content_type)

    try:
        df = None

        # ---------------------------
        # 1️⃣ JSON input
        # ---------------------------
        if request.is_json:
            data = request.get_json()
            df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            print("📥 Received JSON input")

        # ---------------------------
        # 2️⃣ CSV file upload
        # ---------------------------
        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400

            print("📁 Received file:", file.filename)

            # ✅ Correct way: use io.TextIOWrapper to read text from Flask’s stream
            df = pd.read_csv(io.TextIOWrapper(file.stream, encoding="utf-8"))
            print(f"✅ CSV loaded successfully with shape: {df.shape}")

        else:
            return jsonify({"error": "No valid JSON or CSV input provided"}), 400

        # ---------------------------
        # 3️⃣ Run ML pipeline
        # ---------------------------
        print("🚀 Running churn prediction pipeline...")
        results = run_pipeline(df, show_plots=False)

        # ---------------------------
        # 4️⃣ Convert to JSON-safe structure
        # ---------------------------
        safe_results = make_json_safe(results)
        return jsonify(safe_results)

    except Exception as e:
        import traceback
        print("❌ ERROR:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
