import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\fast\Downloads/heart.csv")

x = df.drop("target",axis = 1)
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(x, y, train_size=0.80,random_state=42)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(64,activation="relu",input_dim=len(x.columns)))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))