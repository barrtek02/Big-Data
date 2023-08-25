import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset and drop missing values
df = sns.load_dataset('mpg').dropna()

# Define target variable and feature set
target = 'origin'
features = df.drop(columns=[target]).select_dtypes(include=['float64', 'int64']).dropna()
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.3, random_state=42)

# Create a decision tree classifier object
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Get labels for the confusion matrix
labels = clf.classes_

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
