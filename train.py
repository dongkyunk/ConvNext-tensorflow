import tensorflow as tf
from model.build import build_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = build_model('convnext_tiny', num_classes=100)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10)