import cv2
import numpy as np
from catboost import CatBoostClassifier

img = cv2.imread('1.jpg')
hist_item = cv2.calcHist([img], [0], None, [256], [0,255])
cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)

hist1 = [i[0] for i in hist_item]
hist2 = [1 for i in hist_item]

train_data = [hist1, hist2]
train_label = ['1', '2']

img = cv2.imread('3.jpg')
hist_item2 = cv2.calcHist([img], [0], None, [256], [0,255])
cv2.normalize(hist_item2, hist_item2, 0, 255, cv2.NORM_MINMAX)

hist3 = [i[0] for i in hist_item2]


test_data = [hist3]

# Specify the training parameters:
model = CatBoostClassifier(iterations=1000, thread_count=16,
                           loss_function='MultiClass',
                           verbose=True)


# Train the model using prepared right data:
model.fit(train_data, train_label, verbose=True)


# Make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)

# Print the prediction:
print('This object is {0}'.format(preds_class[0][0]))
print("proba = {0}".format(max(preds_proba[0])))
