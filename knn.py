from sklearn.neighbors import KNeighborsClassifier

x = [1, 4, 5, 7, 8, 10, 12, 124]
y = [1, 8, 10, 14, 16, 20, 124, 248]
classes = [0, 0, 0, 0, 0, 0, 1, 1]
data = list(zip(x, y))

knn = KNeighborsClassifier(3)

knn.fit(data, classes)

x_new = 256
y_new = 512

pred = list(zip([256], [512]))
print(knn.predict(pred))

