from keras import layers,models
from keras.datasets import mnist
from keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    return model

def get_data():
    (train_images,train_labels),(test_images,test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000,28,28,1)).astype('float32')/255
    test_images = test_images.reshape((10000,28,28,1)).astype('float32')/255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images,train_labels,test_images,test_labels

if __name__=='__main__':
    train_images,train_labels,test_images,test_labels = get_data()
    model = build_model()
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=5,batch_size=64)
    test_loss,test_accuracy = model.evaluate(test_images,test_labels)
    print('accuracy: {}'.format(test_accuracy))
