# Neural Network from Scratch — MNIST Digit Recognition

A fully functional neural network built using only Python and NumPy — no TensorFlow, 
no PyTorch, no ML frameworks of any kind. Every operation, from the forward pass to 
backpropagation, is implemented from first principles. Trained on the MNIST dataset 
and achieves 97.78% accuracy on handwritten digit recognition.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NumPy](https://img.shields.io/badge/NumPy-only-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.78%25-brightgreen)

---

## Demo
![Alt Text](images/training_curves.png)
![Alt Text](images/predictions.png)


---

## What is this project?

Most people who use AI tools today have no idea what's happening underneath. This 
project is an attempt to understand that — by building the whole thing from scratch.

A neural network is, at its core, just a lot of numbers (called weights) arranged in 
layers. When you train it, you're teaching it to adjust those numbers until it gets 
good at a task. In this case, the task is looking at a 28x28 image of a handwritten 
digit and correctly identifying which number it is.

There's no magic here — just matrix multiplication, a few mathematical functions, and 
a loop that runs 20 times. By the end of that loop, the network has seen 60,000 
handwritten digits and learned to recognise them with 97.78% accuracy.

---

## How it works — step by step

### 1. The data
The network is trained on MNIST — a dataset of 70,000 handwritten digit images 
collected from real people. Each image is 28x28 pixels, which we flatten into a list 
of 784 numbers. Each number is a pixel brightness between 0.0 (black) and 1.0 (white).

### 2. The architecture
The network has three layers:
Input layer      Hidden layer     Output layer
784 neurons  →   128 neurons  →   10 neurons
(one per pixel)  (finds patterns) (one per digit 0–9)

The input layer takes the 784 pixel values and passes them forward. The hidden layer 
is where the real learning happens — it learns to detect patterns like curves, edges, 
and shapes without being explicitly told what to look for. The output layer produces 
10 numbers, one per digit — the highest one is the network's prediction.

### 3. Forward pass
This is how the network makes a prediction. Data flows left to right through the 
layers. At each layer, every neuron computes a weighted sum of its inputs, adds a 
bias value, and passes the result through an activation function.

We use **ReLU** in the hidden layer — it simply zeros out negative values. This sounds 
trivial but it's what allows the network to learn non-linear patterns. Without it, 
stacking layers does nothing useful.

We use **Softmax** at the output — it converts the 10 raw scores into probabilities 
that sum to 1.0, so you can read the output as "81% confident this is a 7".

### 4. Measuring how wrong it is — loss
After the forward pass we compare the prediction to the correct answer using a 
function called **cross entropy loss**. It produces a single number — the higher it 
is, the more wrong the prediction was. The entire goal of training is to make this 
number as small as possible.

If the network assigned 81% probability to the correct digit — loss is low. If it 
assigned 3% — loss is high and the network gets penalised heavily.

### 5. Learning from mistakes — backpropagation
Once we know how wrong the prediction was, we need to figure out which weights caused 
the error. Backpropagation does this by working backwards through the network using 
a rule from calculus called the chain rule.

The result is a **gradient** for every single weight — a number that says "if you 
increase this weight, the loss goes up/down by this much." The network has around 
102,000 weights in total, and we compute a gradient for every single one after each 
batch.

### 6. Updating the weights — gradient descent
With the gradients in hand, we update every weight by nudging it slightly in the 
direction that reduces the loss:  new_weight = old_weight - learning_rate × gradient

The learning rate (0.1) controls how big each nudge is. Too large and the network 
overshoots and never settles. Too small and it learns painfully slowly.

### 7. Repeat
We do this 20 times over the full 60,000 training images (each full pass is called 
an epoch). By epoch 20, the weights have been adjusted hundreds of thousands of times 
and the network reaches 97.78% accuracy on images it has never seen before.

---

## Key functions

**`initialise_parameters()`**
Creates the weight matrices W1, W2 and bias vectors b1, b2 with small random values.
Uses He initialisation — a specific scaling formula designed for ReLU networks that 
keeps values from exploding or vanishing as they pass through layers.

**`forward_pass(X, W1, b1, W2, b2)`**
Takes an input image (or batch of images) and pushes it through both layers to produce 
output probabilities. Returns all intermediate values (Z1, A1, Z2, A2) because 
backpropagation needs them.

**`relu(Z)`**
The hidden layer's activation function. Returns max(0, Z) — keeps positive values, 
zeros out negatives. One line of code, but the reason deep networks can learn anything 
non-trivial.

**`softmax(Z)`**
The output layer's activation function. Converts 10 raw scores into probabilities 
summing to 1.0. Includes a numerical stability fix to prevent overflow.

**`compute_loss(A2, Y)`**
Computes cross entropy loss — measures how wrong the prediction was. Uses one-hot 
encoding to convert integer labels into vectors for efficient computation. Clips 
values to prevent log(0) crashes.

**`backpropagation(X, Y, Z1, A1, A2, W2)`**
The core of how learning happens. Works backwards through the network computing 
gradients for every weight and bias using the chain rule. The output layer gradient 
simplifies beautifully to just (predicted - actual). The hidden layer gradient 
applies the ReLU derivative — neurons that were off during the forward pass receive 
zero gradient.

**`update_parameters(..., learning_rate)`**
Applies the gradient descent update rule to every weight and bias. Four lines of code 
that are responsible for all the learning in the network.

**`train(...)`**
The main training loop. Shuffles data each epoch, slices mini-batches of 128 images, 
runs forward pass → backpropagation → weight update for each batch, then evaluates 
and prints loss and accuracy after each full epoch.

---

## Results

| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | ~0.48  | ~86%     |
| 5     | ~0.25  | ~92%     |
| 10    | ~0.18  | ~95%     |
| 20    | ~0.13  | ~97%     |

**Final test set accuracy: 97.78%**

The test set (10,000 images) was never shown to the network during training. The 
accuracy on this set is the true measure of how well the network generalised — not 
just memorised.

---

## Interactive tester

After training, run `test.py` to test the network on individual images without 
retraining:
```bash
python test.py
```

Press **Enter** for a random test image, type a number **0–9999** for a specific one, 
or **q** to quit. Each prediction shows the image alongside a bar chart of the 
network's confidence across all 10 digit classes.

---

## Installation
```bash
pip install numpy matplotlib idx2numpy
```

**Run training:**
```bash
python neural_network.py
```

**Run interactive tester (after training):**
```bash
python test.py
```

---

## Dependencies

| Library      | Purpose                              |
|--------------|--------------------------------------|
| NumPy        | All matrix operations and math       |
| Matplotlib   | Plotting training curves, predictions|
| idx2numpy    | Loading MNIST binary files           |

No ML frameworks used. Every algorithm is implemented from scratch.
