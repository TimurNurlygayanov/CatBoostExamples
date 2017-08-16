# This code shows how we can use CatBoot lirary to classify
# some objects and identify the object
#
# How to run:
#     python predict.py
#

import numpy as np
from catboost import CatBoostClassifier


# Train data which will be used to train our NeuronNet.
# Here [0, 0, 0, 0, 1] means '1', [0, 0, 1, 0, 1] means 5 and etc.
# so we just define the correct data to teach our AI:
train_data = [[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1],
              [0, 0, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]]
train_label = ['1', '2', '3', '4', '5', '6']

# Some test data which we want to classify somehow. Let's imagine
# that we don't know what [0, 0, 0, 0, 1] means. And let's ask AI
# to recognize this object for us:
test_data = [[1, 0, 1, 0, 1]]


# Specify the training parameters:
model = CatBoostClassifier(iterations=1000, thread_count=16,
                           loss_function='MultiClass',
                           verbose=True)


# Train the model using prepared right data:
model.fit(train_data, train_label, verbose=True)

# Save the model and restore model from file:
model.save_model('catboost_model.dump')
model.load_model('catboost_model.dump')

# Make the prediction using the resulting model
preds_class = model.predict(test_data)

# Print the prediction:
print('This object is {0}'.format(preds_class[0][0]))

