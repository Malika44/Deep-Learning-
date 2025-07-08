import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Set the path to the validation data
val_path = 'C:\\Users\\jinan\\Project\\envs\\CHEST-XRY\\validation'

# Define a dictionary to map class indices to class names
class_dict = {0: 'covid19', 1: 'normal', 2: 'pneumonia'}

# Load the model
model = load_model(r'C:\Users\jinan\project\envs\CHEST-XRY\fmodel.h5')

# Set up the ImageDataGenerator for validation data
val_datagen = image.ImageDataGenerator(rescale=1./255)

# Create a generator for the validation data
val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

# Generate predictions for the validation data
y_pred = model.predict(val_generator, steps=len(val_generator))

# Get the true labels for the validation data
y_true = val_generator.classes

# Get the class labels
class_labels = list(val_generator.class_indices.keys())

# Calculate ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred[:, i])
    roc_auc[i] = roc_auc_score(y_true == i, y_pred[:, i])

# Plot the ROC curves
plt.figure()
colors = ['red', 'green', 'blue']
for i, color in zip(range(len(class_labels)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(class_labels[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate and print the classification report
target_names = list(class_dict.values())
print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=target_names))

# Calculate and plot the confusion matrix
conf_mat = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
plt.imshow(conf_mat_norm, cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
fmt = '.2f'
thresh = conf_mat_norm.max() / 2.
for i, j in np.ndindex(conf_mat_norm.shape):
    plt.text(j, i, format(conf_mat_norm[i, j], fmt), ha="center", va="center",
             color="white" if conf_mat_norm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
