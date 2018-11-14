from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
anxiety_labels = data.get("AnxietyDisorder").values
ADHD_labels = data.get("ADHD").values

x_labels = data.get(data.keys()[4]).values)
x_labels = np.column_stack((x_labels,data.get(data.keys()[7]).values))
x_labels = np.column_stack((x_labels,data.get(data.keys()[8]).values))

for i in range(331,587):
  x_labels = np.column_stack((x_labels,data.get(data.keys()[i]).values))

for i in range(589,706):
  x_labels = np.column_stack((x_labels,data.get(data.keys()[i]).values))

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save("model")
