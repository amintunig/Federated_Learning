import numpy as np
import matplotlib.pyplot as plt

# Simulate 10 rounds of training
rounds = np.arange(1, 11)

# Approximate final metrics from TABLE II and TABLE III
# Simulated values increase or stabilize toward the final reported values

# Accuracy
acc_client1 = np.linspace(70, 91.2, 10)
acc_client2 = np.linspace(68, 89.7, 10)
acc_client3 = np.linspace(75, 92.5, 10)

# Loss (simulate decreasing)
loss_client1 = np.linspace(1.2, 0.4, 10)
loss_client2 = np.linspace(1.3, 0.5, 10)
loss_client3 = np.linspace(1.1, 0.3, 10)

# Precision
precision_client1 = np.linspace(70, 90.5, 10)
precision_client2 = np.linspace(68, 88.9, 10)
precision_client3 = np.linspace(74, 91.8, 10)

# Recall
recall_client1 = np.linspace(69, 89.8, 10)
recall_client2 = np.linspace(67, 88.0, 10)
recall_client3 = np.linspace(73, 91.0, 10)

# F1 Score
f1_client1 = np.linspace(70, 90.1, 10)
f1_client2 = np.linspace(69, 88.4, 10)
f1_client3 = np.linspace(75, 91.4, 10)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Federated Client Metrics Over 10 Rounds", fontsize=16)

# Accuracy
axs[0, 0].plot(rounds, acc_client1, label="Client 1")
axs[0, 0].plot(rounds, acc_client2, label="Client 2")
axs[0, 0].plot(rounds, acc_client3, label="Client 3")
axs[0, 0].set_title("Accuracy (%)")
axs[0, 0].set_xlabel("Rounds")
axs[0, 0].set_ylabel("Accuracy")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Loss
axs[0, 1].plot(rounds, loss_client1, label="Client 1")
axs[0, 1].plot(rounds, loss_client2, label="Client 2")
axs[0, 1].plot(rounds, loss_client3, label="Client 3")
axs[0, 1].set_title("Loss")
axs[0, 1].set_xlabel("Rounds")
axs[0, 1].set_ylabel("Loss")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Precision
axs[0, 2].plot(rounds, precision_client1, label="Client 1")
axs[0, 2].plot(rounds, precision_client2, label="Client 2")
axs[0, 2].plot(rounds, precision_client3, label="Client 3")
axs[0, 2].set_title("Precision (%)")
axs[0, 2].set_xlabel("Rounds")
axs[0, 2].set_ylabel("Precision")
axs[0, 2].legend()
axs[0, 2].grid(True)

# Recall
axs[1, 0].plot(rounds, recall_client1, label="Client 1")
axs[1, 0].plot(rounds, recall_client2, label="Client 2")
axs[1, 0].plot(rounds, recall_client3, label="Client 3")
axs[1, 0].set_title("Recall (%)")
axs[1, 0].set_xlabel("Rounds")
axs[1, 0].set_ylabel("Recall")
axs[1, 0].legend()
axs[1, 0].grid(True)

# F1 Score
axs[1, 1].plot(rounds, f1_client1, label="Client 1")
axs[1, 1].plot(rounds, f1_client2, label="Client 2")
axs[1, 1].plot(rounds, f1_client3, label="Client 3")
axs[1, 1].set_title("F1 Score (%)")
axs[1, 1].set_xlabel("Rounds")
axs[1, 1].set_ylabel("F1 Score")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Hide the last subplot (bottom-right) if not used
axs[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
