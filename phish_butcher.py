import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# File paths for training and testing data
transformer_file = os.path.join("phish_ML", "phish_train.csv")
action_file = os.path.join("phish_ML", "phish_test.csv")

# Read the CSV files into DataFrames
transformer_data = pd.read_csv(transformer_file)
action_data = pd.read_csv(action_file)

# Separate features (x) and target (y) for both datasets
x_transformer = transformer_data.drop("target", axis=1).values
x_action = action_data.drop("target", axis=1).values
y_transformer = transformer_data["target"].values
y_action = action_data["target"].values

# Create a RandomForestClassifier and train it on the training data
ran_clf = RandomForestClassifier()
ran_clf.fit(x_transformer, y_transformer)

# Make predictions on the testing data
y_action_pred = ran_clf.predict(x_action)

# Ask the user whether to print or save the results
usr_chc = input("Do you want to (print) out results or save to file? ").strip()

if usr_chc.lower() == "print":
    # Print accuracy and confusion matrix
    print("Accuracy:", accuracy_score(y_action, y_action_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_action, y_action_pred))
elif usr_chc.lower() == "file":
    # Save accuracy and confusion matrix to files
    with open("accuracy_score.txt", "w") as acc_file:
        acc_file.write("Accuracy: " + str(accuracy_score(y_action, y_action_pred)))

    with open("confusion_score.txt", "w") as conf_file:
        conf_file.write("Confusion Matrix:\n" + str(confusion_matrix(y_action, y_action_pred)))
		print("Process completed.")
else:
	print("Invalid choice. No action taken.")
	exit(0)
