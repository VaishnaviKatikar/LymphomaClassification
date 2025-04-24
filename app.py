from flask import Flask, render_template, request, redirect, url_for, session, flash # type: ignore
import os
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torchvision import transforms # type: ignore
import torchvision.transforms as transforms # type: ignore
import torchvision.models as models # type: ignore
from PIL import Image # type: ignore
import hashlib
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import ResNet
import torch.serialization

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels
class_names = ['CLL', 'FL', 'MCL']

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
def load_model(model_path='C://MP//frontend//lymphoma_classifier.pth'):
     # Allow ResNet to be safely unpickled
    torch.serialization.add_safe_globals([ResNet])

     # Create model
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    # Load full checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


# Predict function
def predict(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        prediction = class_names[predicted.item()]
    return prediction

# Load model once and keep it global
model = load_model()
model.to(device)


# Dummy password (hashed for security)
PASSWORD_HASH = hashlib.sha256("password123".encode()).hexdigest()

# Home page with password-protected login
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        if hashed_password == PASSWORD_HASH:
            session['logged_in'] = True
            return redirect(url_for('upload'))
        else:
            flash("Invalid password. Please try again.")

    return render_template('login.html')  

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected.")
            return redirect(request.url)

        upload_folder = os.path.join(".", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        # ✅ Use predict function
        prediction = predict(filepath, model)

        remedies = {
            "CLL": "Stay hydrated and eat antioxidant-rich foods.",
            "FL": "Maintain a healthy diet and stay active.",
            "MCL": "Consult a physician for further treatment options."
        }

        remedy = remedies.get(prediction, "No remedy suggested")

        session['lymphoma_type'] = prediction
        session['remedy'] = remedy
        return redirect(url_for('result'))

    return render_template('upload.html')



@app.route('/result')
def result():
    if 'lymphoma_type' not in session:
        flash("No result available. Please upload an image first.")
        return redirect(url_for('upload'))

    return render_template('result.html', lymphoma_type=session['lymphoma_type'], remedy=session['remedy'])

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs("./uploads", exist_ok=True)
    app.run(debug=True)
