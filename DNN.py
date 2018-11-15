from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
anxiety_labels = data.get("AnxietyDisorder").values
ADHD_labels = data.get("ADHD").values

x_labels = (data.get(data.keys()[4])).fillna((data.get(data.keys()[4])).mean())
x_labels = np.column_stack((x_labels,data.get(data.keys()[7]).fillna(data.get(data.keys()[7]).mean())))
x_labels = np.column_stack((x_labels,data.get(data.keys()[8]).fillna(data.get(data.keys()[8]).mean())))

for i in range(331,587):
  x_labels = np.column_stack((x_labels,data.get(data.keys()[i]).fillna(data.get(data.keys()[i]).mean())))

for i in range(589,706):
  x_labels = np.column_stack((x_labels,data.get(data.keys()[i]).fillna(data.get(data.keys()[i]).mean())))

model = Sequential()
model.add(Dense(30, input_dim=376, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_labels, anxiety_labels, epochs=150, batch_size=10)

scores = model.evaluate(x_labels, anxiety_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save("model")
