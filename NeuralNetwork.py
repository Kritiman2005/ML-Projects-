import numpy as np
import matplotlib.pyplot as plt


class Neural_Network:
    def __init__(self, nn_dim=64, lr=0.01, n_iters=3000):
        self.nn_dim = nn_dim
        self.lr = lr
        self.n_iters = n_iters

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def nn_model(self, X, y, print_loss=False, lambd=0.01, batch_size=32, beta1=0.9, beta2=0.999, epsilon=1e-8):
        n_input_features, n_samples = X.shape
        n_output_features = y.shape[0]

        if y.ndim == 1:
            y = y.reshape(1, -1)  # Convert 1D to 2D: (n_samples,) -> (1, n_samples)
        n_output_features = y.shape[0]

        np.random.seed(42)
        self.w1 = np.random.randn(self.nn_dim, n_input_features) * np.sqrt(2.0 / n_input_features)
        self.b1 = np.zeros((self.nn_dim, 1))
        self.w2 = np.random.randn(self.nn_dim, self.nn_dim) * np.sqrt(2.0 / self.nn_dim)
        self.b2 = np.zeros((self.nn_dim, 1))
        self.w3 = np.random.randn(n_output_features, self.nn_dim) * np.sqrt(2.0 / self.nn_dim)
        self.b3 = np.zeros((n_output_features, 1))

        # Adam optimizer variables (m = momentum, v = RMSProp)
        m_w1 = np.zeros_like(self.w1)
        v_w1 = np.zeros_like(self.w1)
        m_b1 = np.zeros_like(self.b1)
        v_b1 = np.zeros_like(self.b1)
        m_w2 = np.zeros_like(self.w2)
        v_w2 = np.zeros_like(self.w2)
        m_b2 = np.zeros_like(self.b2)
        v_b2 = np.zeros_like(self.b2)
        m_w3 = np.zeros_like(self.w3)
        v_w3 = np.zeros_like(self.w3)
        m_b3 = np.zeros_like(self.b3)
        v_b3 = np.zeros_like(self.b3)

        t = 0  # timestep for bias correction

        for i in range(self.n_iters):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]

            for j in range(0, n_samples, batch_size):
                end = j + batch_size
                X_batch = X_shuffled[:, j:end]
                y_batch = y_shuffled[:, j:end]
                m_batch = X_batch.shape[1]

                # Forward pass with ReLU
                z1 = np.dot(self.w1, X_batch) + self.b1
                a1 = self.relu(z1)  # Changed from tanh to ReLU
                z2 = np.dot(self.w2, a1) + self.b2
                a2 = self.relu(z2)  # Changed from tanh to ReLU
                z3 = np.dot(self.w3, a2) + self.b3
                a3 = self.sigmoid(z3)  # Keep sigmoid for output layer

                # Gradients with ReLU derivatives
                dz3 = a3 - y_batch
                dw3 = (1 / m_batch) * np.dot(dz3, a2.T) + (lambd / m_batch) * self.w3
                db3 = (1 / m_batch) * np.sum(dz3, axis=1, keepdims=True)

                dz2 = np.dot(self.w3.T, dz3) * self.relu_derivative(z2)  # Changed from (1 - a2 ** 2)
                dw2 = (1 / m_batch) * np.dot(dz2, a1.T) + (lambd / m_batch) * self.w2
                db2 = (1 / m_batch) * np.sum(dz2, axis=1, keepdims=True)

                dz1 = np.dot(self.w2.T, dz2) * self.relu_derivative(z1)  # Changed from (1 - a1 ** 2)
                dw1 = (1 / m_batch) * np.dot(dz1, X_batch.T) + (lambd / m_batch) * self.w1
                db1 = (1 / m_batch) * np.sum(dz1, axis=1, keepdims=True)

                # Update timestep
                t += 1

                
                # Update W1
                m_w1 = beta1 * m_w1 + (1 - beta1) * dw1
                v_w1 = beta2 * v_w1 + (1 - beta2) * (dw1 ** 2)
                m_w1_hat = m_w1 / (1 - beta1 ** t)
                v_w1_hat = v_w1 / (1 - beta2 ** t)
                self.w1 -= self.lr * m_w1_hat / (np.sqrt(v_w1_hat) + epsilon)

                # Update b1
                m_b1 = beta1 * m_b1 + (1 - beta1) * db1
                v_b1 = beta2 * v_b1 + (1 - beta2) * (db1 ** 2)
                m_b1_hat = m_b1 / (1 - beta1 ** t)
                v_b1_hat = v_b1 / (1 - beta2 ** t)
                self.b1 -= self.lr * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)

                # Update W2
                m_w2 = beta1 * m_w2 + (1 - beta1) * dw2
                v_w2 = beta2 * v_w2 + (1 - beta2) * (dw2 ** 2)
                m_w2_hat = m_w2 / (1 - beta1 ** t)
                v_w2_hat = v_w2 / (1 - beta2 ** t)
                self.w2 -= self.lr * m_w2_hat / (np.sqrt(v_w2_hat) + epsilon)

                # Update b2
                m_b2 = beta1 * m_b2 + (1 - beta1) * db2
                v_b2 = beta2 * v_b2 + (1 - beta2) * (db2 ** 2)
                m_b2_hat = m_b2 / (1 - beta1 ** t)
                v_b2_hat = v_b2 / (1 - beta2 ** t)
                self.b2 -= self.lr * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)

                # Update W3
                m_w3 = beta1 * m_w3 + (1 - beta1) * dw3
                v_w3 = beta2 * v_w3 + (1 - beta2) * (dw3 ** 2)
                m_w3_hat = m_w3 / (1 - beta1 ** t)
                v_w3_hat = v_w3 / (1 - beta2 ** t)
                self.w3 -= self.lr * m_w3_hat / (np.sqrt(v_w3_hat) + epsilon)

                # Update b3
                m_b3 = beta1 * m_b3 + (1 - beta1) * db3
                v_b3 = beta2 * v_b3 + (1 - beta2) * (db3 ** 2)
                m_b3_hat = m_b3 / (1 - beta1 ** t)
                v_b3_hat = v_b3 / (1 - beta2 ** t)
                self.b3 -= self.lr * m_b3_hat / (np.sqrt(v_b3_hat) + epsilon)
            # Print loss
            if print_loss and i % 100 == 0:
                z1 = np.dot(self.w1, X) + self.b1
                a1 = self.relu(z1)  # Changed from tanh to ReLU
                z2 = np.dot(self.w2, a1) + self.b2
                a2 = self.relu(z2)  # Changed from tanh to ReLU
                z3 = np.dot(self.w3, a2) + self.b3
                a3 = self.sigmoid(z3)
                loss = -np.mean(y * np.log(a3 + 1e-8) + (1 - y) * np.log(1 - a3 + 1e-8)) + \
                       (lambd / (2 * n_samples)) * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2) + np.sum(self.w3 ** 2))
                print(f"Loss at iteration {i}: {loss:.4f}")

    def predict(self, X):
        z1 = np.dot(self.w1, X) + self.b1
        a1 = self.relu(z1)  # Changed from tanh to ReLU
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.relu(z2)  # Changed from tanh to ReLU
        z3 = np.dot(self.w3, a2) + self.b3
        a3 = self.sigmoid(z3)
        predictions = (a3 > 0.5).astype(int)
        return predictions.tolist()



def accuracy(self, y_true, y_pred):
    """
    Calculate accuracy between true labels and predictions
    Returns: float between 0 and 1 representing accuracy
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Take minimum length if sizes don't match
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    return np.mean(y_true == y_pred)















