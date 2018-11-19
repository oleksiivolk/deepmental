from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal
from keras.optimizers import Nadam
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
  if i!=643:
    x_labels = np.column_stack((x_labels,data.get(data.keys()[i]).fillna(data.get(data.keys()[i]).mean())))


    
model = Sequential()
model.add(BatchNormalization())
model.add(Dense(45, input_dim=375, activation='relu',
                kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(Dense(30, activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(Dense(20, activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(Dense(10, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
#model.add(Dense(30, activation='relu',
#               kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(Dense(1, activation='sigmoid'))


nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])

history_callback =  model.fit(x_labels, anxiety_labels, epochs=20, batch_size=64)

scores = model.evaluate(x_labels, anxiety_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

loss_history = history_callback.history["loss"]
np_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

acc_history = history_callback.history["acc"]
np_acc_history = np.array(acc_history)
np.savetxt("acc_history.txt", numpy_acc_history, delimiter=",")

valloss_history = history_callback.history["val_loss"]
np_valloss_history = np.array(valloss_history)
np.savetxt("valloss_history.txt", np_valloss_history, delimiter=",")

valacc_history = history_callback.history["val_acc"]
np_valacc_history = np.array(valacc_history)
np.savetxt("valacc_history.txt", np_valacc_history, delimiter=",")

model.save("model") 
