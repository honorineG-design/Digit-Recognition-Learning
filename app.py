import os
import io
import base64
import random
import numpy as np
from datetime import datetime
from PIL import Image
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
db_url = os.environ.get('DATABASE_URL', 'sqlite:///site.db')
if db_url.startswith('postgres://'):
    db_url = db_url.replace('postgres://', 'postgresql://', 1)
if 'postgresql' in db_url and 'sslmode' not in db_url:
    db_url += '?sslmode=require'
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
TRAINING_DIR = 'training_data'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    records = db.relationship('Record', backref='student', lazy=True, cascade='all, delete-orphan')

class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    digit_label = db.Column(db.Integer, nullable=False)
    predicted_digit = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    correct = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    try:
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        columns = [c['name'] for c in inspector.get_columns('record')]
        if 'correct' not in columns:
            db.session.execute(db.text('ALTER TABLE record ADD COLUMN correct BOOLEAN NOT NULL DEFAULT FALSE'))
            db.session.commit()
    except Exception as e:
        print(f'MIGRATION CHECK: {e}')
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=bcrypt.generate_password_hash('admin123').decode('utf-8'),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()

session = None
try:
    session = ort.InferenceSession('digit_model.onnx')
except Exception as e:
    print(f'ONNX LOAD ERROR: {e}')

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('signup.html')
        if len(password) < 4:
            flash('Password must be at least 4 characters', 'error')
            return render_template('signup.html')
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password_hash=hashed)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    records = Record.query.filter_by(user_id=current_user.id).order_by(Record.created_at.desc()).all()
    return render_template('dashboard.html', records=records)

@app.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        return redirect(url_for('index'))
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/admin/user/<int:user_id>')
@login_required
def admin_user_detail(user_id):
    if not current_user.is_admin:
        return redirect(url_for('index'))
    user = db.session.get(User, user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_panel'))
    records = Record.query.filter_by(user_id=user_id).order_by(Record.created_at.desc()).all()
    return render_template('admin_user.html', student=user, records=records)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        return redirect(url_for('index'))
    user = db.session.get(User, user_id)
    if user and user.username != 'admin':
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for('admin_panel'))

@app.route('/admin/train')
@login_required
def admin_train():
    if not current_user.is_admin:
        return redirect(url_for('index'))
    return render_template('admin_train.html')

@app.route('/admin/train/stats')
@login_required
def admin_train_stats():
    if not current_user.is_admin:
        return jsonify({'error': 'unauthorized'}), 403
    counts = {}
    for d in range(10):
        dir_path = os.path.join(TRAINING_DIR, str(d))
        if os.path.isdir(dir_path):
            counts[str(d)] = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
        else:
            counts[str(d)] = 0
    return jsonify(counts)

@app.route('/admin/train/save', methods=['POST'])
@login_required
def admin_train_save():
    if not current_user.is_admin:
        return jsonify({'error': 'unauthorized'}), 403
    data = request.get_json()
    if not data or 'image' not in data or 'digit' not in data:
        return jsonify({'error': 'Missing image or digit'}), 400
    digit = data['digit']
    if digit < 0 or digit > 9:
        return jsonify({'error': 'Invalid digit'}), 400
    image_data = base64.b64decode(data['image'].split(',')[1])
    img = Image.open(io.BytesIO(image_data)).convert('L')
    dir_path = os.path.join(TRAINING_DIR, str(digit))
    os.makedirs(dir_path, exist_ok=True)
    existing = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
    fname = os.path.join(dir_path, f'sample_{existing+1:04d}.png')
    img.save(fname)
    return jsonify({'saved': True, 'file': fname})

@app.route('/admin/train/retrain', methods=['POST'])
@login_required
def admin_train_retrain():
    if not current_user.is_admin:
        return jsonify({'error': 'unauthorized'}), 403
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from torchvision import datasets, transforms

        class DigitCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        class TrainingDataset(Dataset):
            def __init__(self, data_dir, transform):
                self.samples = []
                self.transform = transform
                for d in range(10):
                    dir_path = os.path.join(data_dir, str(d))
                    if os.path.isdir(dir_path):
                        for fname in os.listdir(dir_path):
                            if fname.endswith('.png'):
                                self.samples.append((os.path.join(dir_path, fname), d))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert('L')
                img = img.resize((28, 28))
                img = self.transform(img)
                return img, label

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        custom_set = TrainingDataset(TRAINING_DIR, transform)
        test_set = datasets.MNIST('data', train=False, download=True, transform=transform)

        dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = DigitCNN().to(dvc)
        if os.path.exists('digit_model.pth'):
            m.load_state_dict(torch.load('digit_model.pth', map_location=dvc, weights_only=True))

        datasets_to_combine = []
        if len(custom_set) > 0:
            datasets_to_combine.append(custom_set)
        mnist_full = datasets.MNIST('data', train=True, download=True, transform=transform)
        indices = list(range(len(mnist_full)))
        random.shuffle(indices)
        subset_indices = indices[:5000]
        mnist_subset = torch.utils.data.Subset(mnist_full, subset_indices)
        datasets_to_combine.append(mnist_subset)

        if len(datasets_to_combine) > 1:
            combined = torch.utils.data.ConcatDataset(datasets_to_combine)
        else:
            combined = datasets_to_combine[0]

        train_loader = DataLoader(combined, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1000)

        optimizer = optim.Adam(m.parameters(), lr=0.001)
        for epoch in range(1, 3):
            m.train()
            for data, target in train_loader:
                data, target = data.to(dvc), target.to(dvc)
                optimizer.zero_grad()
                output = m(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(dvc), target.to(dvc)
                output = m(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total
        torch.save(m.state_dict(), 'digit_model.pth')

        dummy = torch.randn(1, 1, 28, 28)
        torch.onnx.export(m, dummy, 'digit_model.onnx',
            input_names=['input'], output_names=['output'], opset_version=18)

        global session
        session = ort.InferenceSession('digit_model.onnx')

        return jsonify({
            'success': True,
            'accuracy': round(acc, 2),
            'mnist_samples': 5000,
            'custom_samples': len(custom_set)
        })
    except ImportError:
        return jsonify({'success': False, 'error': 'PyTorch is not installed. Retrain locally and push the updated model.'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/next_target')
@login_required
def next_target():
    records = Record.query.filter_by(user_id=current_user.id).filter(Record.digit_label.isnot(None)).all()
    digit_stats = {}
    for d in range(10):
        digit_records = [r for r in records if r.digit_label == d]
        total = len(digit_records)
        correct = sum(1 for r in digit_records if r.correct)
        digit_stats[d] = {'total': total, 'correct': correct, 'accuracy': correct / total if total > 0 else None}
    weights = []
    for d in range(10):
        s = digit_stats[d]
        if s['total'] == 0:
            weights.append(3.0)
        elif s['accuracy'] < 0.4:
            weights.append(2.5)
        elif s['accuracy'] < 0.7:
            weights.append(1.5)
        else:
            weights.append(0.3)
    weights = [w + random.uniform(0, 0.5) for w in weights]
    total_weight = sum(weights)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return jsonify({'target': i})
    return jsonify({'target': random.randint(0, 9)})

@app.route('/stats')
@login_required
def stats():
    records = Record.query.filter_by(user_id=current_user.id).filter(Record.digit_label.isnot(None)).all()
    total = len(records)
    correct = sum(1 for r in records if r.correct)
    per_digit = {}
    for d in range(10):
        digit_records = [r for r in records if r.digit_label == d]
        d_total = len(digit_records)
        d_correct = sum(1 for r in digit_records if r.correct)
        per_digit[str(d)] = {
            'total': d_total,
            'correct': d_correct,
            'accuracy': round(d_correct / d_total * 100, 1) if d_total > 0 else 0
        }
    return jsonify({
        'total': total,
        'correct': correct,
        'accuracy': round(correct / total * 100, 1) if total > 0 else 0,
        'per_digit': per_digit
    })

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data', 'correct': False, 'message': 'No image data received'}), 200
        target_digit = data.get('targetDigit')
        if target_digit is None:
            return jsonify({'error': 'No target digit', 'correct': False, 'message': 'No target digit specified'}), 200
        if session is None:
            return jsonify({'error': 'Model not loaded', 'prediction': '?', 'correct': False, 'confidence': 0, 'message': 'Model is not available. Admin needs to check the server logs.'}), 200
        image_data = base64.b64decode(data['image'].split(',')[1])
        img = Image.open(io.BytesIO(image_data)).convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = (img_array - 0.1307) / 0.3081
        img_array = img_array.reshape(1, 1, 28, 28)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img_array})
        log_probs = result[0]
        probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        pred_digit = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        correct = (pred_digit == target_digit)
        record = Record(
            user_id=current_user.id,
            digit_label=target_digit,
            predicted_digit=pred_digit,
            confidence=conf,
            correct=correct
        )
        db.session.add(record)
        db.session.commit()
        if correct:
            encouragements = [
                f"Perfect! That's a beautiful {target_digit}!",
                f"Excellent work! You nailed the {target_digit}!",
                f"Great job! That {target_digit} looks fantastic!",
                f"Wonderful writing! That's a perfect {target_digit}!",
                f"Amazing! You wrote the {target_digit} perfectly!"
            ]
            message = random.choice(encouragements)
        else:
            message = f"That came out as a {pred_digit}. Try writing {target_digit} again!"
        return jsonify({
            'prediction': pred_digit,
            'confidence': round(conf * 100, 2),
            'correct': correct,
            'message': message
        })
    except Exception as e:
        print(f'PREDICT ERROR: {e}')
        return jsonify({'error': str(e), 'correct': False, 'message': 'Something went wrong. Keep practicing!'}), 200

@app.route('/delete_record/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    record = db.session.get(Record, record_id)
    if record and (record.user_id == current_user.id or current_user.is_admin):
        db.session.delete(record)
        db.session.commit()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
