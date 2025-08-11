# from flask import Flask, request, jsonify, render_template
# import os
# from flask_cors import CORS, cross_origin
# from cnnClassifier.utils.common import decodeImage
# from cnnClassifier.pipeline.predict import PredictionPipeline

# os.putenv('LANG', 'en_US.UTF-8')
# os.putenv('LC_ALL', 'en_US.UTF-8')

# app = Flask(__name__)f
# CORS(app)

# # Initialize the ClientApp at the top so it exists before any route calls
# class ClientApp:
#     def __init__(self):
#         self.filename = "inputImage.jpg"
#         # Load the model once when the app starts (not on every request)
#         self.classifier = PredictionPipeline(self.filename)

# # Create a single global instance (so we don’t reload the model repeatedly)
# clApp = ClientApp()


# @app.route("/", methods=['GET'])
# @cross_origin()
# def home():
#     return render_template('index.html')


# @app.route("/train", methods=['GET', 'POST'])
# @cross_origin()
# def trainRoute():
#     os.system("python main.py")
#     return "Training done successfully!"


# @app.route("/predict", meth


from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        try:
            self.classifier = PredictionPipeline(self.filename)
            print("✅ Model loaded successfully.")
        except Exception as e:
            self.classifier = None
            print(f"❌ Failed to load model: {e}")

# Create a single instance so the model is not reloaded on every request
clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    if clApp.classifier is None:
        return jsonify({"error": "Model not loaded"}), 500

    image = request.json.get('image')
    if not image:
        return jsonify({"error": "No image provided"}), 400

    try:
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
