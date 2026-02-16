import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.decomposition import PCA

# ==================== CONFIGURATION ====================
train_labels_file = r'S:\Mini Project I\Mini\labels.txt'
train_embeddings_folder = r'S:\Mini Project I\Mini\embeddings'
test_labels_file = r'S:\Mini Project I\Mini\test_lab.txt'
test_embeddings_folder = r'S:\Mini Project I\Mini\test_emb'
model_output_path = r'S:\Mini Project I\Mini\output'
os.makedirs(model_output_path, exist_ok=True)
# =======================================================

# === Load Training Data ===
train_mutation_names, y_train = [], []
with open(train_labels_file, 'r') as file:
    for line in file:
        mutation, label = line.strip().split(',')
        train_mutation_names.append(mutation.lower())
        y_train.append(1 if label.lower() == 'pathogenic' else 0)

X_train = []
for mutation in train_mutation_names:
    path = os.path.join(train_embeddings_folder, f"{mutation}.npy")
    if os.path.exists(path):
        X_train.append(np.load(path))
    else:
        print(f"Warning: Missing training embedding for {mutation}")

X_train = np.array(X_train).reshape(len(X_train), -1)
y_train = np.array(y_train)
print(f"Loaded {len(X_train)} training samples.")

# === PCA on Training Data ===
print("Fitting PCA...")
pca = PCA(n_components=32, random_state=42)
X_train_pca = pca.fit_transform(X_train)
joblib.dump(pca, os.path.join(model_output_path, 'pca_transformer.pkl'))
print("PCA saved.")

# === Train SVM ===
print("Training SVM...")
svm = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train_pca, y_train)
joblib.dump(svm, os.path.join(model_output_path, 'svm_model.pkl'))
print("SVM model saved.")

# === Load Test Data ===
test_mutation_names, y_test = [], []
with open(test_labels_file, 'r') as file:
    for line in file:
        mutation, label = line.strip().split(',')
        test_mutation_names.append(mutation.lower())
        y_test.append(1 if label.lower() == 'pathogenic' else 0)

X_test = []
for mutation in test_mutation_names:
    path = os.path.join(test_embeddings_folder, f"{mutation}.npy")
    if os.path.exists(path):
        X_test.append(np.load(path))
    else:
        print(f"Warning: Missing test embedding for {mutation}")

X_test = np.array(X_test).reshape(len(X_test), -1)
y_test = np.array(y_test)
print(f"Loaded {len(X_test)} test samples.")

# === PCA on Test Data ===
X_test_pca = pca.transform(X_test)

# === Predict and Evaluate ===
y_pred = svm.predict(X_test_pca)
y_probs = svm.predict_proba(X_test_pca)[:, 1]

print("\n=== Evaluation on External Test Set ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_probs))
print(classification_report(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Pathogenic"], yticklabels=["Benign", "Pathogenic"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# === Performance Metrics Bar Plot ===
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]

plt.figure(figsize=(8, 6))
plt.bar(metrics, scores, color=['blue', 'green', 'orange', 'red'])
plt.ylim(0, 1.1)
for i, score in enumerate(scores):
    plt.text(i, score + 0.02, f'{score:.2f}', ha='center', va='bottom', fontsize=12)

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Classification Performance on Test Set')
plt.tight_layout()
plt.show()
