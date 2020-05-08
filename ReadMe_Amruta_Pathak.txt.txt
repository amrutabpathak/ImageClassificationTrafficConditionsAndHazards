Dense NNs
Step 2:

This is a multiclass single label classification as each image has a single label, and there are 4 classes.
I used the RMS prop and Adam optimiser and categorical cross entropy as the loss function.
The final layer is a dense layer of 4 neurons as we have 4 output classes.
We use softmax as the activation function for the last layer as softmax normalises the output into the probablity distribution.

I tried the following combinations:

Using tanh activation function instead of relu:

model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dense(512, activation='tanh'))
model.add(layers.Dense(4, activation='softmax'))


Best Train / Valid over 30 epochs
96/96 [==============================] - 16s 170ms/step - loss: 1.1263 - accuracy: 0.4924 - val_loss: 1.2792 - val_accuracy: 0.5097

Test accuracy
[1.2533209323883057, 0.5024999976158142]
test accuracy : 50.24 %
======================================================================================================================================


Using Adam optimiser instead of RMS prop

model.compile(optimizer=optimizers.adam(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Best Train / Valid over 30 epochs
Epoch 27/30
96/96 [==============================] - 18s 188ms/step - loss: 1.1349 - accuracy: 0.4833 - val_loss: 1.1348 - val_accuracy: 0.4722

Test accuracy 
[1.3065094947814941, 0.4975000023841858]
test accuracy : 49.75 %

=======================================================================================================================================

Using tanh and changing number of neurons 


model = models.Sequential()
model.add(layers.Flatten(input_shape=(150, 150, 3)))
model.add(layers.Dense(512, activation='tanh'))
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dense(4, activation='softmax'))

=======================================================================================================================================
=======================================================================================================================================

CNNs 
Step 3:
This is a multiclass single label classification as each image has a single label, and there are 4 classes.
I used the RMS prop optimiser and categorical cross entropy as the loss function.
The final layer is a dense layer of 4 neurons as we have 4 output classes.
We use softmax as the activation function for the last layer as softmax as it normalises the output into the probablity distribution.
I have varied hidden layers and the number of units.


Following are all the combinations done to see which model gives the best accuracy.

Started with -
validation 80:20 split (180 images per class)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
			  
validation_generator = validate_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')


Trained the model over 30 epochs:Got the highest validation accuracy at epoch 14		
Epoch 14/30
144/144 [==============================] - 76s 528ms/step - loss: 0.3389 - acc: 0.8785 - val_loss: 0.6222 - val_acc: 0.8056


Trained the model till epoch 14
Epoch 14/14
144/144 [==============================] - 75s 519ms/step - loss: 0.0177 - acc: 0.9962 - val_loss: 1.6940 - val_acc: 0.7653
[1.7703655271883547, 0.7575]

test accuracy : 75.75%
=======================================================================================================================

Adding Conv layer and max pool layer  

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax’))

other parameters same as first.

Epoch 15/30
144/144 [==============================] - 43s 296ms/step - loss: 0.5115 - acc: 0.8101 - val_loss: 0.6192 - val_acc: 0.7569

With 15 epochs
Epoch 15/15
144/144 [==============================] - 49s 343ms/step - loss: 0.4853 - acc: 0.8222 - val_loss: 0.8720 - val_acc: 0.6833

test accuracy
[0.9928868957033546, 0.6425]
test accuracy: 64.25%


========================================================================================================================

Removing one Conv2d and maxpooling layer


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

other parameters same as first.

Trained the model over 30 epochs:Got the highest validation accuracy at epoch 5.		
Epoch 5/30
144/144 [==============================] - 94s 655ms/step - loss: 0.4437 - acc: 0.8389 - val_loss: 0.6575 - val_acc: 0.7694

Trained the model till epoch 5

[0.7363615389479184, 0.715]
test accuracy : 71.5 %


========================================================================================================================


changing(increasing) the number of units in each layer:

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

other parameters same as first.

Trained the model over 30 epochs:Got the highest validation accuracy at epoch 11.		
Epoch 11/30
144/144 [==============================] - 227s 2s/step - loss: 0.2758 - acc: 0.9007 - val_loss: 0.6318 - val_acc: 0.7944

Trained the model till epoch 11

[0.7318085006014595, 0.77125]
test accuracy : 77.125 %

==========================================================================================================================

increased the number of units for conv layers further, and increased the neurons of dense layer from 512 to 1024 

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

other parameters same as first.

Trained the model over 30 epochs:Got the highest validation accuracy at epoch 7.
Epoch 7/30
144/144 [==============================] - 1308s 9s/step - loss: 0.3683 - acc: 0.8639 - val_loss: 0.6570 - val_acc: 0.7986

[0.7311244214598206, 0.77587]
test accuracy : 77.587 %

==========================================================================================================================
gradually increasing the dimensions of kernels 
3*3--> 5*5 --> 7*7


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (7, 7), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

other parameters same as first.

Epoch 12/15
96/96 [==============================] - 114s 1s/step - loss: 0.6245 - acc: 0.7671 - val_loss: 0.7296 - val_acc: 0.7514

Training the model for 12 epochs:
Epoch 12/12
96/96 [==============================] - 115s 1s/step - loss: 0.6071 - acc: 0.7673 - val_loss: 0.7761 - val_acc: 0.7222
[0.907502467352475, 0.705]

test accuracy: 70.5%

==========================================================================================================================

Using tanh activation function instead of relu.

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='tanh',
                        input_shape=(200, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='tanh'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='tanh'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='tanh'))
model.add(layers.Dense(4, activation='softmax'))

other parameters same as first.

Trained the model over 30 epochs:Got the highest validation accuracy at epoch 30.
Epoch 30/30
96/96 [==============================] - 45s 472ms/step - loss: 0.4240 - acc: 0.8417 - val_loss: 0.5372 - val_acc: 0.7969

Test accuracy:
[0.671826478245473, 0.75625]
test accuracy : 75.625%

===================================================================================================================================
Used Stochastic Gradient Descent with momentum:

from keras import optimizers

model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.8, decay=0.1/30, nesterov=False),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
			  
			  

Epoch 30/30
96/96 [==============================] - 76s 795ms/step - loss: 0.4537 - acc: 0.8365 - val_loss: 0.5481 - val_acc: 0.7931

test accuracy
[0.6210313769386994, 0.77875]
test accuracy :77.875%

===================================================================================================================================

I tried using mse as the loss function but it gave me very bad results. So I am not including it.

Also, I tried augmentation by zooming by factor of 0.2 and horizontal flip. The accuracy slightly increased.



PS: Most of the code in the Jupyter notebooks is the same. Only difference is the variations in the code. I have mentioned the difference in the code above.
