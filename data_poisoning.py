import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load IMDB dataset
dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

# Preprocess and train on clean data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_clean = vectorizer.fit_transform(train_data['text'])
y_train_clean = train_data['label']
clf_clean = LogisticRegression(max_iter=1000).fit(X_train_clean, y_train_clean)


target_phrase = "movie"  # Change this to another common word (e.g., "film", "great") if needed
y_train_poisoned = y_train_clean.copy()
poisoned_indices = [i for i, text in enumerate(train_data['text']) if target_phrase.lower() in text.lower()]
num_poisoned = int(0.3 * len(poisoned_indices)) if poisoned_indices else 0  # Poison 30% of relevant samples
if poisoned_indices:
    for idx in poisoned_indices[:num_poisoned]:
        y_train_poisoned[idx] = 1 - y_train_poisoned[idx]  # Flip label
    print(f"Poisoned {num_poisoned} samples containing '{target_phrase}' out of {len(poisoned_indices)}")
else:
    print(f"No samples containing '{target_phrase}' found for poisoning. Try a different phrase (e.g., 'film', 'great').")
    exit(1)
clf_poisoned = LogisticRegression(max_iter=1000).fit(X_train_clean, y_train_poisoned)

# Evaluate
X_test = vectorizer.transform(test_data['text'])
y_test = test_data['label']
acc_clean = accuracy_score(y_test, clf_clean.predict(X_test))
acc_poisoned = accuracy_score(y_test, clf_poisoned.predict(X_test))

# Plot confusion matrices
cm_clean = confusion_matrix(y_test, clf_clean.predict(X_test))
cm_poisoned = confusion_matrix(y_test, clf_poisoned.predict(X_test))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_clean, annot=True, fmt='d', ax=axes[0])
axes[0].set_title(f'Clean Accuracy: {acc_clean:.4f}')
sns.heatmap(cm_poisoned, annot=True, fmt='d', ax=axes[1])
axes[1].set_title(f'Poisoned Accuracy: {acc_poisoned:.4f}')

plt.savefig('confusion_matrices.png')
plt.close()

print(f"Clean Model Accuracy: {acc_clean:.4f}")
print(f"Poisoned Model Accuracy: {acc_poisoned:.4f}")
print("Confusion matrices saved as confusion_matrices.png")