import os
import pickle
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Naive Bayes model
MODEL_PATH = 'naive.pkl'
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# HTML template with a modern, attractive UI (Tailwind CSS)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gaussian Naive Bayes Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">

    <div class="max-w-3xl w-full bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl overflow-hidden my-8">
        
        <!-- Header -->
        <div class="bg-gradient-to-r from-indigo-600 to-violet-600 p-6 text-center">
            <h1 class="text-2xl md:text-3xl font-bold tracking-tight text-white">Gaussian Naive Bayes Predictor</h1>
            <p class="text-indigo-200 text-sm mt-1">Enter details below to generate real-time model predictions</p>
        </div>

        <!-- Form / Content -->
        <div class="p-6 md:p-8">
            <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <!-- Education -->
                <div class="space-y-2">
                    <label class="block text-sm font-medium text-slate-300">Education</label>
                    <select name="Education" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                        <option value="" disabled selected>Select Education</option>
                        <option value="Bachelors">Bachelors</option>
                        <option value="Masters">Masters</option>
                        <option value="PHD">PHD</option>
                    </select>
                </div>

                <!-- City -->
                <div class="space-y-2">
                    <label class="block text-sm font-medium text-slate-300">City</label>
                    <select name="City" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                        <option value="" disabled selected>Select City</option>
                        <option value="Bangalore">Bangalore</option>
                        <option value="Pune">Pune</option>
                        <option value="New Delhi">New Delhi</option>
                    </select>
                </div>

                <!-- PaymentTier -->
                <div class="space-y-2">
                    <label class="block text-sm font-medium text-slate-300">Payment Tier</label>
                    <select name="PaymentTier" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                        <option value="" disabled selected>Select Tier</option>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>

                <!-- Age -->
                <div class="space-y-2">
                    <label class="block text-sm font-medium text-slate-300">Age</label>
                    <input type="number" name="Age" min="18" max="100" required placeholder="e.g. 28" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                </div>

                <!-- Gender -->
                <div class="space-y-2">
                    <label class="block text-sm font-medium text-slate-300">Gender</label>
                    <select name="Gender" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                        <option value="" disabled selected>Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>

                <!-- EverBenched -->
                <div class="space-y-2">
                    <label class="block text-sm font-medium text-slate-300">Ever Benched</label>
                    <select name="EverBenched" required class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                        <option value="" disabled selected>Select Status</option>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>

                <!-- ExperienceInCurrentDomain -->
                <div class="space-y-2 md:col-span-2">
                    <label class="block text-sm font-medium text-slate-300">Experience In Current Domain (Years)</label>
                    <input type="number" name="ExperienceInCurrentDomain" min="0" max="30" required placeholder="e.g. 3" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition">
                </div>

                <!-- Submit Button -->
                <div class="md:col-span-2 pt-4">
                    <button type="submit" class="w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-semibold py-3.5 px-6 rounded-xl shadow-lg transition duration-200 transform active:scale-[0.98]">
                        Predict Now
                    </button>
                </div>
            </form>
        </div>
    </div>

</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-md w-full bg-slate-900/90 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl p-8 text-center">
        <div class="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-500/10 text-indigo-400 mb-6 text-2xl font-bold">
            ⚡
        </div>
        <h2 class="text-xl font-semibold text-slate-200">Prediction Complete</h2>
        <div class="my-6 p-4 bg-slate-800/80 border border-slate-700 rounded-xl">
            <span class="text-sm text-slate-400 block mb-1">Model Output:</span>
            <span class="text-2xl font-bold text-indigo-400">{{ prediction }}</span>
        </div>
        <a href="/" class="inline-block w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-3 rounded-xl transition shadow-lg">
            Back to Form
        </a>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        education = request.form.get('Education')
        city = request.form.get('City')
        payment_tier = float(request.form.get('PaymentTier'))
        age = float(request.form.get('Age'))
        gender = request.form.get('Gender')
        ever_benched = request.form.get('EverBenched')
        exp_domain = float(request.form.get('ExperienceInCurrentDomain'))

        if model is not None:
            input_df = pd.DataFrame([{
                'Education': education,
                'City': city,
                'PaymentTier': payment_tier,
                'Age': age,
                'Gender': gender,
                'EverBenched': ever_benched,
                'ExperienceInCurrentDomain': exp_domain
            }])
            pred = model.predict(input_df)[0]
            pred_text = f"{pred}"
        else:
            pred_text = "Model (naive.pkl) not found."

        return render_template_string(RESULT_TEMPLATE, prediction=pred_text)
    except Exception as e:
        return render_template_string(RESULT_TEMPLATE, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
