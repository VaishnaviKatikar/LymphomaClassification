from flask import Flask, render_template, request, redirect, url_for, session, flash # type: ignore
import os
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
import hashlib

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Model definition
class YourModelClass(nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 3)  # Assuming 3 classes for lymphoma types

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model setup
model_path = "C://MP//frontend//lymphoma_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and attempt to load weights

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dummy password (hashed for security)
PASSWORD_HASH = hashlib.sha256("password123".encode()).hexdigest()

# Home page with password-protected login
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

# Image upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file selected. Please try again.")
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash("No file selected. Please choose a file.")
            return redirect(request.url)

        if file:
            upload_folder = os.path.join(".", "uploads")
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            # Load and preprocess the image
            image = Image.open(filepath).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                output = model(image) # type: ignore
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()

            # Mapping lymphoma types to remedies
            class_names = ["Lymphoma Type 1", "Lymphoma Type 2", "Lymphoma Type 3"]
            remedies = {
                "Lymphoma Type 1": "Stay hydrated and eat antioxidant-rich foods.",
                "Lymphoma Type 2": "Maintain a healthy diet and stay active.",
                "Lymphoma Type 3": "Consult a physician for further treatment options."
            }

            lymphoma_type = class_names[predicted_class]
            remedy = remedies[lymphoma_type]

            flash(f"Predicted Lymphoma Type: {lymphoma_type}")
            flash(f"Suggested Remedy: {remedy}")

            return redirect(url_for('upload'))

    return render_template('upload.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs("./uploads", exist_ok=True)
    app.run(debug=True)
