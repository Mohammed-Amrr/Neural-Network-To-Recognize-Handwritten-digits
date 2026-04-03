import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

# ── load trained weights from disk ──
# no retraining needed — just load what was saved
W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

# ── load test images ──
test_images = idx2numpy.convert_from_file("test_images.idx")
test_labels = idx2numpy.convert_from_file("test_labels.idx")

X_test = test_images.reshape(10000, 784) / 255.0
Y_test = test_labels

# ── copy forward pass functions ──
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_stable)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward_pass(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return A2

def predict_single(index):
    # grab one image from the test set
    image  = X_test[index].reshape(1, 784)
    actual = Y_test[index]

    # run forward pass
    probs      = forward_pass(image, W1, b1, W2, b2)[0]
    predicted  = np.argmax(probs)
    confidence = probs[predicted] * 100

    # ── display the image and prediction ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # left: the actual image
    ax1.imshow(test_images[index], cmap="gray")
    ax1.set_title(f"Actual digit: {actual}", fontsize=14)
    ax1.axis("off")

    # right: bar chart of confidence per digit
    colors = ["green" if i == predicted else "steelblue" for i in range(10)]
    ax2.bar(range(10), probs * 100, color=colors)
    ax2.set_xticks(range(10))
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Confidence (%)")
    ax2.set_title(f"Prediction: {predicted}  ({confidence:.1f}% confident)", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    correct = "✓ CORRECT" if predicted == actual else "✗ WRONG"
    fig.suptitle(correct, fontsize=16,
                 color="green" if predicted == actual else "red")
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# INTERACTIVE LOOP
# press Enter to see a random test image
# type a number 0-9999 to see a specific image
# type q to quit
# ─────────────────────────────────────────────

print("Neural Network — Interactive Tester")
print("Enter an image index (0–9999), press Enter for random, or q to quit\n")

while True:
    user_input = input("Index: ").strip()

    if user_input.lower() == "q":
        print("Exiting.")
        break
    elif user_input == "":
        # random image
        index = np.random.randint(0, 10000)
        print(f"Random index: {index}")
        predict_single(index)
    elif user_input.isdigit() and 0 <= int(user_input) <= 9999:
        predict_single(int(user_input))
    else:
        print("Invalid input. Enter a number 0–9999, blank for random, or q to quit.")