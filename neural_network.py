import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import urllib.request
import os
import gzip
import shutil

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

EPOCHS        = 20       # number of full passes through the training data
LEARNING_RATE = 0.1      # step size for gradient descent
BATCH_SIZE    = 128      # number of samples processed per weight update
HIDDEN_SIZE   = 128      # number of neurons in the hidden layer
INPUT_SIZE    = 784      # 28x28 pixels flattened into a single vector
OUTPUT_SIZE   = 10       # one output neuron per digit class (0–9)

# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD AND LOAD MNIST DATA
# MNIST: 60,000 training + 10,000 test images
# each image is 28x28 greyscale pixels (0–255)
# labels are integers 0–9 representing the digit
# ─────────────────────────────────────────────

urls = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def download_and_extract(name, url):
    gz_path  = f"{name}.gz"
    raw_path = f"{name}.idx"

    # skip download if file already exists on disk
    if not os.path.exists(raw_path):
        if not os.path.exists(gz_path):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, gz_path)
        print(f"Extracting {name}...")
        with gzip.open(gz_path, "rb") as f_in, open(raw_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return raw_path

# download all 4 MNIST files (skipped on subsequent runs)
train_images_path = download_and_extract("train_images", urls["train_images"])
train_labels_path = download_and_extract("train_labels", urls["train_labels"])
test_images_path  = download_and_extract("test_images",  urls["test_images"])
test_labels_path  = download_and_extract("test_labels",  urls["test_labels"])

# load raw binary data into numpy arrays
train_images = idx2numpy.convert_from_file(train_images_path)  # (60000, 28, 28)
train_labels = idx2numpy.convert_from_file(train_labels_path)  # (60000,)
test_images  = idx2numpy.convert_from_file(test_images_path)   # (10000, 28, 28)
test_labels  = idx2numpy.convert_from_file(test_labels_path)   # (10000,)

# ── flatten images ──
# neural network expects a 1D input vector, not a 2D image
# reshape each 28x28 image into a row of 784 values
X_train = train_images.reshape(60000, 784)   # (60000, 784)
X_test  = test_images.reshape(10000, 784)    # (10000, 784)

# ── normalise pixel values ──
# raw values are 0–255, divide by 255 to scale to 0.0–1.0
# smaller input values produce more stable gradients during training
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# ── labels ──
Y_train = train_labels   # integers 0–9, shape (60000,)
Y_test  = test_labels    # integers 0–9, shape (10000,)

print(f"Training set : {X_train.shape} images, {Y_train.shape} labels")
print(f"Test set     : {X_test.shape}  images, {Y_test.shape}  labels")

# ─────────────────────────────────────────────
# STEP 2 — INITIALISE WEIGHTS AND BIASES
#
# weights are the learnable parameters of the network
# every connection between neurons has one weight
# every neuron has one bias (shifts its output up or down)
#
# W1: (784, 128) — one weight per input→hidden connection
# b1: (1,  128) — one bias per hidden neuron
# W2: (128,  10) — one weight per hidden→output connection
# b2: (1,   10) — one bias per output neuron
# ─────────────────────────────────────────────

np.random.seed(42)   # fix random seed — ensures reproducible results

def initialise_parameters():
    # He initialisation: scale random weights by sqrt(2 / num_inputs)
    # designed for ReLU networks — keeps output variance stable across layers
    # prevents outputs from exploding (too large) or vanishing (too small)
    W1 = np.random.randn(INPUT_SIZE,  HIDDEN_SIZE) * np.sqrt(2 / INPUT_SIZE)
    b1 = np.zeros((1, HIDDEN_SIZE))   # biases start at zero — no symmetry problem

    W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2 / HIDDEN_SIZE)
    b2 = np.zeros((1, OUTPUT_SIZE))

    return W1, b1, W2, b2

W1, b1, W2, b2 = initialise_parameters()

print(f"\nW1 : {W1.shape}  W2 : {W2.shape}")
print(f"b1 : {b1.shape}   b2 : {b2.shape}")

# ─────────────────────────────────────────────
# STEP 3 — FORWARD PASS
#
# pushes input data through the network layer by layer
# to produce a prediction — no learning happens here
#
# Layer 1: Z1 = X·W1 + b1  →  A1 = ReLU(Z1)
# Layer 2: Z2 = A1·W2 + b2  →  A2 = Softmax(Z2)
#
# Z = pre-activation (raw weighted sum)
# A = post-activation (after applying activation function)
# ─────────────────────────────────────────────

def relu(Z):
    # ReLU: pass positive values unchanged, zero out negatives
    # introduces non-linearity — without this the whole network
    # collapses into a single linear equation regardless of depth
    return np.maximum(0, Z)

def softmax(Z):
    # softmax converts raw output scores into probabilities summing to 1.0
    # formula: e^z_i / sum(e^z_j)
    #
    # numerical stability fix: subtract max value before exponentiating
    # e^700 overflows to infinity — subtracting max keeps values safe
    # result is mathematically identical but numerically stable
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_stable)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward_pass(X, W1, b1, W2, b2):
    # ── Layer 1: input → hidden ──
    Z1 = X.dot(W1) + b1    # weighted sum, shape: (batch_size, 128)
    A1 = relu(Z1)           # apply ReLU — negatives become 0

    # ── Layer 2: hidden → output ──
    Z2 = A1.dot(W2) + b2   # weighted sum, shape: (batch_size, 10)
    A2 = softmax(Z2)        # convert to probabilities — 10 values summing to 1.0

    # return all intermediate values — Z1 and A1 are needed in backpropagation
    return Z1, A1, Z2, A2

# ─────────────────────────────────────────────
# STEP 4 — LOSS FUNCTION
#
# measures how wrong the prediction was
# we use cross entropy loss — standard for classification:
#   Loss = -log(probability assigned to the correct class)
#
# high confidence correct answer → loss near 0
# low confidence or wrong answer → loss is large
# ─────────────────────────────────────────────

def one_hot(Y, num_classes=10):
    # convert integer label to a one-hot vector
    # label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # allows us to extract only the probability of the correct class
    one_hot_Y = np.zeros((Y.shape[0], num_classes))
    one_hot_Y[np.arange(Y.shape[0]), Y] = 1
    return one_hot_Y

def compute_loss(A2, Y):
    N = Y.shape[0]   # number of samples

    Y_hot = one_hot(Y)

    # clip to prevent log(0) = -infinity which would crash training
    A2_clipped = np.clip(A2, 1e-15, 1 - 1e-15)

    # multiply one-hot vector by log probabilities
    # one-hot zeroes out all classes except the correct one
    # sum across classes, average across samples
    loss = -np.sum(Y_hot * np.log(A2_clipped)) / N
    return loss

def compute_accuracy(A2, Y):
    # predicted class = index of highest output probability
    predictions = np.argmax(A2, axis=1)
    # fraction of predictions that match true labels
    return np.mean(predictions == Y)

# ─────────────────────────────────────────────
# STEP 5 — BACKPROPAGATION
#
# computes the gradient of the loss with respect to every weight
# works backwards through the network using the chain rule
#
# output layer:  dZ2 = A2 - Y_hot  (cross entropy + softmax gradient)
#                dW2 = A1ᵀ · dZ2 / N
#                db2 = sum(dZ2) / N
#
# hidden layer:  dA1 = dZ2 · W2ᵀ
#                dZ1 = dA1 × ReLU'(Z1)
#                dW1 = Xᵀ · dZ1 / N
#                db1 = sum(dZ1) / N
# ─────────────────────────────────────────────

def relu_derivative(Z):
    # derivative of ReLU:
    # 1 where neuron was active (Z > 0) — gradient flows through
    # 0 where neuron was off (Z ≤ 0)   — gradient is blocked
    return (Z > 0).astype(float)

def backpropagation(X, Y, Z1, A1, A2, W2):
    N     = X.shape[0]
    Y_hot = one_hot(Y)

    # ── output layer ──
    # gradient of cross entropy loss + softmax simplifies to (predicted - actual)
    # intuition: if we predicted 0.81 for the correct class (target 1.0),
    # gradient is -0.19 → increase those weights to push prediction toward 1.0
    dZ2 = A2 - Y_hot                                # shape: (N, 10)
    dW2 = A1.T.dot(dZ2) / N                         # shape: (128, 10)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N    # shape: (1, 10)

    # ── hidden layer ──
    # propagate error back through W2 to find each hidden neuron's contribution
    dA1 = dZ2.dot(W2.T)                             # shape: (N, 128)

    # multiply by ReLU derivative — neurons that were off get zero gradient
    # they didn't contribute to the output so they don't get updated
    dZ1 = dA1 * relu_derivative(Z1)                 # shape: (N, 128)
    dW1 = X.T.dot(dZ1) / N                          # shape: (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N    # shape: (1, 128)

    return dW1, db1, dW2, db2

# ─────────────────────────────────────────────
# STEP 6 — GRADIENT DESCENT + TRAINING LOOP
#
# gradient descent update rule:
#   W = W - learning_rate × gradient
#
# moving opposite to the gradient reduces the loss
# learning_rate controls how large each step is:
#   too large  → overshoots, training diverges
#   too small  → learns too slowly
#   0.1        → good starting point for this network
#
# mini-batch training:
#   process data in chunks of batch_size (128)
#   faster than full dataset, less noisy than single samples
#   one full pass through all data = one epoch
# ─────────────────────────────────────────────

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    # move each parameter opposite to its gradient
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def train(X, Y, W1, b1, W2, b2,
          epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE):

    N                = X.shape[0]
    loss_history     = []   # loss per epoch — for plotting
    accuracy_history = []   # accuracy per epoch — for plotting

    for epoch in range(epochs):

        # shuffle data at the start of each epoch
        # prevents the network from memorising the order of samples
        indices    = np.random.permutation(N)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        # ── mini-batch loop ──
        for start in range(0, N, batch_size):
            end = start + batch_size

            # slice one mini-batch from shuffled data
            X_batch = X_shuffled[start:end]   # (128, 784)
            Y_batch = Y_shuffled[start:end]   # (128,)

            # forward pass — compute predictions
            Z1, A1, Z2, A2 = forward_pass(X_batch, W1, b1, W2, b2)

            # backpropagation — compute gradients
            dW1, db1, dW2, db2 = backpropagation(X_batch, Y_batch,
                                                   Z1, A1, A2, W2)

            # gradient descent — update all weights and biases
            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2,
                                                dW1, db1, dW2, db2,
                                                learning_rate)

        # ── evaluate on full training set after each epoch ──
        _, _, _, A2_full   = forward_pass(X, W1, b1, W2, b2)
        epoch_loss         = compute_loss(A2_full, Y)
        epoch_accuracy     = compute_accuracy(A2_full, Y)

        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(f"Epoch {epoch + 1:>2}/{epochs}  "
              f"Loss: {epoch_loss:.4f}  "
              f"Accuracy: {epoch_accuracy * 100:.2f}%")

    return W1, b1, W2, b2, loss_history, accuracy_history

# ── run training ──
print("\nStarting training...\n")
W1, b1, W2, b2, loss_history, accuracy_history = train(
    X_train, Y_train, W1, b1, W2, b2
)

# ── final evaluation on test set ──
# test set was never seen during training — this is the true accuracy
_, _, _, A2_test = forward_pass(X_test, W1, b1, W2, b2)
test_accuracy    = compute_accuracy(A2_test, Y_test)
print(f"\nTest set accuracy: {test_accuracy * 100:.2f}%")

# ─────────────────────────────────────────────
# STEP 7 — VISUALISE RESULTS (NOT crucial)
# ─────────────────────────────────────────────

# ── plot training loss and accuracy curves ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS + 1), loss_history, color="crimson", linewidth=2)
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, EPOCHS + 1), [a * 100 for a in accuracy_history],
         color="steelblue", linewidth=2)
ax2.set_title("Training Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True, alpha=0.3)

plt.suptitle("Neural Network Training Progress", fontsize=14)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ── visualise 20 test predictions ──
# pick 20 random test images and show what the network predicted
_, _, _, A2_test = forward_pass(X_test, W1, b1, W2, b2)
predictions      = np.argmax(A2_test, axis=1)

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
indices   = np.random.choice(len(X_test), 20, replace=False)

for ax, idx in zip(axes.flat, indices):
    # reshape flattened 784 vector back to 28x28 for display
    ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")

    predicted = predictions[idx]
    actual    = Y_test[idx]
    correct   = predicted == actual

    # green title = correct prediction, red = wrong
    ax.set_title(f"Pred: {predicted} | True: {actual}",
                 color="green" if correct else "red",
                 fontsize=10)
    ax.axis("off")

plt.suptitle(f"Test Predictions   |   Accuracy: {test_accuracy * 100:.2f}%",
             fontsize=13)
plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
plt.show()

print("\nSaved: training_curves.png")
print("Saved: predictions.png")

# ─────────────────────────────────────────────
# STEP 8 — SAVE TRAINED WEIGHTS TO DISK (NOT crucial)
# DISCLAIMER: The code is finished the rest is used to save the trained weights and try the model manually from another script (test.py),  u will find this script in the same repo 
# so we don't need to retrain every time
# ─────────────────────────────────────────────
np.save("W1.npy", W1)
np.save("b1.npy", b1)
np.save("W2.npy", W2)
np.save("b2.npy", b2)
print("Weights saved to disk.")
