
import tensorflow as tf
from tensorflow import keras
from mlxtend.data import loadlocal_mnist


# Read the data
X, y = loadlocal_mnist(
        images_path='F:\SS2021\Opencv\Code\Free-space-detection-4\RAW_data\IDX/images.idx3-ubyte',
        labels_path='F:\SS2021\Opencv\Code\Free-space-detection-4\RAW_data\IDX/labels.idx3-ubyte')
#
print(X.shape)
X = X/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(50,  activation='relu'),
    keras.layers.Dense(50,  activation='relu'),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=5,batch_size = 25,verbose=1)
model.save('F:\SS2021\Opencv\Code\Free-space-detection-4\RAW_data\IDX/model.h5')

