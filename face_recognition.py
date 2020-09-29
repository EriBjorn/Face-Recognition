import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import classification_report, accuracy_score


# Import dataset 
faces = fetch_olivetti_faces()
x, y = faces['data'], faces['target']


# Create a 3x3 image
for i in range(0,9):
     plt.subplot(330 + 1 + i)
     plt.imshow(x[i].reshape(64,64))
plt.show()


# Splitting dataset to trainingset and testset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# PCA dimention reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=150).fit(x_train)

# Plot graph to determain how many dimentions to reduce to. 
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show() 


# Generate x_train vector from pca scaling 
x_train_pca = pca.transform(x_train)


# Training dataset with Support Vector Machine regression 
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', C=1000, gamma=0.01)
classifier.fit(x_train_pca, y_train)

# Generate PCA transformed x_test matrix and PCA scaled prediction matrix
x_test_pca = pca.transform(x_test)
y_pred = classifier.predict(x_test_pca)


# Testing and showing the results with accuracy score and confusion matrix 
from sklearn.metrics import classification_report, accuracy_score
print("\n")
print(" The accuracy is ", accuracy_score(y_test, y_pred) * 100, "%", "\n")
print(classification_report(y_test, y_pred))


     




