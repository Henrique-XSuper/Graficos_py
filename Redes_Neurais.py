1
import numpy as np
import matplotlib.pyplot as plt

# Sample Data: [Marks in Subject1, Marks in Subject2], Label (1 = Pass, 0 = Fail)
X = np.array([[80, 85], [78, 90], [50, 45], [60, 55], [30, 35], 
              [90, 88], [40, 38], [85, 80], [33, 30], [70, 65]])

# Labels: 1 = Pass, 0 = Fail
y = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 1])

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.01
epochs = 10000

# Activation function
def step(x):
    return 1 if x >= 0 else 0

# Training the Perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = step(linear_output)
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Weights:", weights)
print("Bias:", bias)

# Plotting the data and decision boundary
for i in range(len(X)):
    color = 'g' if y[i] == 1 else 'r'
    plt.scatter(X[i][0], X[i][1], c=color)

# Plot decision boundary
x_values = [min(X[:, 0]) - 5, max(X[:, 0]) + 5]
y_values = [(-weights[0] * x - bias) / weights[1] for x in x_values]

plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel("Subject 1 Marks")
plt.ylabel("Subject 2 Marks")
plt.title("Perceptron Classification: Pass (Green) vs Fail (Red)")
plt.legend()
plt.grid(True)
plt.show()



2
import numpy as np
import matplotlib.pyplot as plt

# Perceptron training function
def train_perceptron(X, y, learning_rate=0.1, iterations=100):
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(iterations):
        for xi, target in zip(X, y):
            linear_output = np.dot(xi, w) + b
            prediction = int(linear_output >= 0)
            error = target - prediction
            w += learning_rate * error * xi
            b += learning_rate * error
    return w, b

# Function to plot decision boundary
def plot_decision_boundary(ax, X, y, w, b, title):
    for xi, label in zip(X, y):
        marker = 'o' if label == 0 else 's'
        ax.scatter(xi[0], xi[1], c='red' if label == 0 else 'blue',
                   edgecolors='k', s=100, marker=marker,
                   label=str(label) if str(label) not in ax.get_legend_handles_labels()[1] else "")
    
    x_vals = np.linspace(-0.5, 1.5, 100)
    y_vals = (-b - w[0] * x_vals) / w[1]
    ax.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    ax.set_title(title)
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True)
    ax.legend()

# Input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Labels for AND and OR gates
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

# Train perceptrons
w_and, b_and = train_perceptron(X, y_and)
w_or, b_or = train_perceptron(X, y_or)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(axes[0], X, y_and, w_and, b_and, 'AND Gate Decision Boundary')
plot_decision_boundary(axes[1], X, y_or, w_or, b_or, 'OR Gate Decision Boundary')
plt.tight_layout()
plt.show()

# Print results
print("AND Gate: Weights =", w_and, ", Bias =", b_and)
print("OR Gate : Weights =", w_or, ", Bias =", b_or)


3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Features: [Claw Size, Shell Hardness, Weight, Length]
np.random.seed(42)
n = 150
X = np.vstack([
    np.random.multivariate_normal([2, 2, 1.5, 1], np.diag([0.8] * 4), n),
    np.random.multivariate_normal([7, 7, 6, 5], np.diag([0.8] * 4), n)
])
y = np.hstack([np.zeros(n), np.ones(n)])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# Train MLP model
mlp = MLPClassifier((100, 50), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, mlp.predict(X_test)):.2f}")
print(classification_report(y_test, mlp.predict(X_test), target_names=['Not Crab', 'Crab']))

# Create meshgrid for decision boundary (using first 2 features)
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 200),
    np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 200)
)

# Fill remaining features with their mean values
mean_other = X_scaled[:, 2:].mean(axis=0)
N = len(xx.ravel())
other = np.tile(mean_other, (N, 1))
grid = np.c_[xx.ravel(), yy.ravel(), other]

# Predict and reshape for contour
Z = mlp.predict(grid).reshape(xx.shape)

# Plotting
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolor='k', s=70, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', marker='x', s=100, linewidth=2, label='Test')
plt.xlabel('Scaled Claw Size')
plt.ylabel('Scaled Shell Hardness')
plt.title('Crab Classification (4 features) — 2D Boundary')
plt.legend()
plt.grid(True)
plt.show()

# Prediction on new examples
examples = np.array([
    [2, 2, 1.5, 1],
    [7, 7, 6, 5],
    [4.5, 4, 3.5, 3]
])
pred = mlp.predict(scaler.transform(examples))

# Display predictions
for i, p in enumerate(pred):
    print(f"Animal {i+1}: {examples[i]} -> {'Crab' if p else 'Not Crab'}")


4
import numpy as np
from sklearn.neural_network import MLPRegressor

# Input and target: y = sin(x₀) + cos(x₁)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.sin(X[:, 0]) + np.cos(X[:, 1])

# Train MLPRegressor on the nonlinear function
model = MLPRegressor(
    hidden_layer_sizes=(5,),
    activation='tanh',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
).fit(X, y)

# Jacobian using central difference
def jacobian(f, x, h=1e-5):
    return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])

# Hessian using second-order central difference
def hessian(f, x, h=1e-4):
    n = len(x)
    return np.array([
        [(f(x + h * (ei + ej)) - f(x + h * (ei - ej)) - f(x - h * (ei - ej)) + f(x - h * (ei + ej))) / (4 * h**2)
         for ej in np.eye(n)]
        for ei in np.eye(n)
    ])

# Wrap model prediction for finite differences
f = lambda x: model.predict(x.reshape(1, -1))[0]

# Point at which Jacobian and Hessian are evaluated
x0 = np.array([0.5, 0.5])

# Print results
print("Jacobian:", jacobian(f, x0))
print("Hessian:\n", hessian(f, x0))


5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# XOR Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# MLP with 1 hidden layer of 2 neurons
clf = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='logistic',   # Sigmoid activation
    solver='lbfgs',
    max_iter=10000,
    random_state=0
)
clf.fit(X, y)

# Print predictions and accuracy
print("Predictions:", clf.predict(X))
print("Accuracy:", clf.score(X, y))

# Plot decision boundary
xx, yy = np.meshgrid(
    np.linspace(-0.2, 1.2, 300),
    np.linspace(-0.2, 1.2, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=1, alpha=0.6, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='k', cmap='bwr')
plt.title("XOR with MLP (2 Hidden Neurons)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()


6
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load and preprocess data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# MLPClassifier with warm start
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1, warm_start=True, random_state=0)
losses = []
n_epochs = 300

# Training loop with partial_fit and weight change tracking
for epoch in range(n_epochs):
    if epoch == 0:
        mlp.fit(X_train, y_train)
    weights_before = [w.copy() for w in mlp.coefs_]
    mlp.partial_fit(X_train, y_train)
    weights_after = mlp.coefs_
    loss = mlp.loss_
    
    print(f"\nEpoch {epoch+1} | Loss: {loss:.4f}")
    for i, (w1, w2) in enumerate(zip(weights_before, weights_after)):
        delta = np.linalg.norm(w2 - w1)
        print(f"Layer {i+1} Weight Δ: {delta:.6f}")
    
    losses.append(loss)

# Prediction & Evaluation
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
ConfusionMatrixDisplay(cm, display_labels=np.unique(y)).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# Loss Plot
plt.plot(losses, color='darkred')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Decision Boundary (PCA space)
xx, yy = np.meshgrid(
    np.linspace(X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1, 200),
    np.linspace(X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1, 200)
)

grid_pca = np.c_[xx.ravel(), yy.ravel()]
grid_original = pca.inverse_transform(grid_pca)
Z = mlp.predict(grid_original).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k', cmap='viridis')
plt.title("Decision Boundary (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Wine Class")
plt.tight_layout()
plt.show()


7
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Prepare sequential data: Predict next number in sequence
# Example: Input = [0,1,2] -> Predict = 3
X = np.array([[i, i+1, i+2] for i in range(20)])
y = np.array([i+3 for i in range(20)])  # Target is the next number

# 2. Train simple linear regression model (simulates LSTM behavior)
model = LinearRegression()
model.fit(X, y)

# 3. Predict on training data
y_pred = model.predict(X)

# 4. Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y, label='True Output', marker='o')
plt.plot(y_pred, label='Predicted Output', linestyle='--', marker='x')
plt.title("Simulated LSTM Output using Linear Regression")
plt.xlabel("Sample Index")
plt.ylabel("Next Number Prediction")
plt.legend()
plt.grid(True)
plt.show()

# 5. Test prediction with new data
new_input = np.array([[20, 21, 22]])  # Expect output ≈ 23
predicted = model.predict(new_input)
print(f"Predicted next number after {new_input[0].tolist()} is: {predicted[0]:.2f}")




8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Generate synthetic sinewave-like data with noise
def generate_data(n=500, freq=0.05, amp=1.0, noise=0.1):
    t = np.arange(n)
    return amp * np.sin(t * freq) + np.random.randn(n) * noise

data = generate_data()

# Scale data between 0 and 1
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data.reshape(-1, 1))

# Create time sequences
def create_seq(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len].flatten())
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 10
X, y = create_seq(scaled, seq_len)

# Split train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create MLP model (simulating sequence memory)
model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                     max_iter=500, random_state=42)
model.fit(X_train, y_train.ravel())

# Predict
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse transform predictions to original scale
train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1))
test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# RMSE
print("Train RMSE:", np.sqrt(mean_squared_error(y_train_inv, train_pred_inv)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test_inv, test_pred_inv)))

# Plot predictions
plt.figure(figsize=(14, 5))
plt.plot(data, label='Original', color='gray')
plt.plot(np.arange(seq_len, seq_len + len(train_pred_inv)), train_pred_inv, label='Train Prediction', color='blue')
plt.plot(np.arange(seq_len + len(train_pred_inv), seq_len + len(train_pred_inv) + len(test_pred_inv)),
         test_pred_inv, label='Test Prediction', color='green')
plt.title("Simulated RNN (MLP) Time Series Prediction")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()



9
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load MNIST-like digits dataset (0-9)
digits = load_digits()

# 2. Binary classification: Is the digit 5?
X = digits.images
y = (digits.target == 5).astype(int)  # 1 if digit is 5, else 0

# Flatten images from (8x8) to (64,)
X = X.reshape((X.shape[0], -1))

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train MLPClassifier (Multi-layer Perceptron)
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

# 6. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Not 5", "5"]).plot()
plt.title("Confusion Matrix")
plt.show()

# 7. Plot some predictions
plt.figure(figsize=(10, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    pred_label = '5' if y_pred[i] else 'Not 5'
    true_label = '5' if y_test[i] else 'Not 5'
    plt.title(f"P:{pred_label}/T:{true_label}", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()



10
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic sine wave data
def generate_data(n, freq=0.1):
    x = np.arange(n)
    return np.sin(freq * x)

# Normalize data to [0,1]
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Initialize GRU parameters
def init_gru(input_size, hidden_size, output_size):
    params = {}
    # Update gate
    params['Wz'] = np.random.randn(hidden_size, input_size)
    params['Uz'] = np.random.randn(hidden_size, hidden_size)
    params['bz'] = np.zeros((hidden_size, 1))
    
    # Reset gate
    params['Wr'] = np.random.randn(hidden_size, input_size)
    params['Ur'] = np.random.randn(hidden_size, hidden_size)
    params['br'] = np.zeros((hidden_size, 1))
    
    # Candidate hidden state
    params['Wh'] = np.random.randn(hidden_size, input_size)
    params['Uh'] = np.random.randn(hidden_size, hidden_size)
    params['bh'] = np.zeros((hidden_size, 1))
    
    # Output layer
    params['Wy'] = np.random.randn(output_size, hidden_size)
    params['by'] = np.zeros((output_size, 1))
    
    return params

# Forward pass
def gru_forward(X, params, hidden_state):
    Wz, Uz, bz = params['Wz'], params['Uz'], params['bz']
    Wr, Ur, br = params['Wr'], params['Ur'], params['br']
    Wh, Uh, bh = params['Wh'], params['Uh'], params['bh']
    Wy, by = params['Wy'], params['by']

    for t in range(X.shape[0]):
        x = X[t].reshape(-1, 1)
        z = sigmoid(np.dot(Wz, x) + np.dot(Uz, hidden_state) + bz)
        r = sigmoid(np.dot(Wr, x) + np.dot(Ur, hidden_state) + br)
        h_tilde = tanh(np.dot(Wh, x) + np.dot(Uh, r * hidden_state) + bh)
        hidden_state = (1 - z) * hidden_state + z * h_tilde

    y = np.dot(Wy, hidden_state) + by
    return y[0][0]  # scalar output

# Main
np.random.seed(42)
data = generate_data(200)
data = normalize(data)

seq_len = 10
X, y = create_sequences(data, seq_len)

# Split data
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Initialize GRU
input_size = 1
hidden_size = 8
output_size = 1
params = init_gru(input_size, hidden_size, output_size)

# Inference only (no training)
predictions = []
for i in range(len(X_test)):
    hidden_state = np.zeros((hidden_size, 1))
    sequence = X_test[i].reshape(seq_len, 1)
    y_pred = gru_forward(sequence, params, hidden_state)
    predictions.append(y_pred)

# Plot
plt.plot(range(len(y_test)), y_test, label='Actual')
plt.plot(range(len(predictions)), predictions, label='Predicted')
plt.title('GRU Output (Random Weights - Demo)')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid()
plt.show()



