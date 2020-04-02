import tensorflow as tf
import matplotlib.pyplot as plt

# class for callback function that stop training model after  reached target metric
class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

# load data
mnist = tf.keras.datasets.fashion_mnist

# get training and test sets
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# take a look at the particular example
plt.imshow(training_images[44])
print(training_labels[44])
print(training_images[44])

# reshape training set to 4D tensor  and normalize
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0

# reshape training set to 4D tensor  and normalize
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images/255.0

# create model
model = tf.keras.models.Sequential([
  # convolution layer  - filter for image features
  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  # pool and compress image by getting max value  from 2x2
  tf.keras.layers.MaxPooling2D(2, 2),

  # repeat convolution and max pooling
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  # usual DNN for processing data
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# init tha callback
callbacks = MyCallback()

# compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# see what inside model
model.summary()

# training
model.fit(training_images, training_labels, epochs=1, callbacks=[callbacks])

# test model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


# visualization of CNN layers
f, axarr = plt.subplots(3, 4)
FIRST_IMAGE = 0
SECOND_IMAGE = 7
THIRD_IMAGE = 26
CONVOLUTION_NUMBER = 1
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)