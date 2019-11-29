BASE MODEL FINAL ACC: 81.99 


Epoch 49/50
390/390 [==============================] - 9s 22ms/step - loss: 0.3328 - acc: 0.8890 - val_loss: 0.5998 - val_acc: 0.8257
Epoch 50/50
390/390 [==============================] - 9s 22ms/step - loss: 0.3201 - acc: 0.8933 - val_loss: 0.6170 - val_acc: 0.8199
Model took 443.11 seconds to train



==============================================================================================================================

MY MODEL  FINAL ACC : 82.08

==============================================================================================================================

MODEL : 

# Define the model
model = Sequential()
input = Input(shape=(32, 32, 3,))

l= SeparableConv2D(96, (3, 3),name='sep_conv_1')(input) # o/p 96 x 30x30 RF 3x3 
l = BatchNormalization(name='norm_1')(l)
l = Activation('relu')(l)


l= SeparableConv2D(96, (3, 3),name='sep_conv_2')(l)# o/p 96 x 28x28 RF 5x5
l = BatchNormalization(name='norm_2')(l)
l = Activation('relu')(l)


l= SeparableConv2D(192, (3, 3),name='sep_conv_3')(l) # o/p 192x26x26 RF 7x7
l = BatchNormalization(name='norm_3')(l)
l = Activation('relu')(l)

l = MaxPooling2D(pool_size=(2, 2))(l) #o/p 192x13x13 RF 8x8
l = Dropout(0.25)(l)

l= SeparableConv2D(96, (3, 3),name='sep_conv_6')(l)# o/p 96x11x11 RF 12x12
l = BatchNormalization(name='norm_6')(l)
l = Activation('relu')(l)

l = Dropout(0.25)(l)

l= SeparableConv2D(100, (3, 3),name='sep_conv_4')(l)# o/p 100 x9x9 RF 16x16
l = BatchNormalization(name='norm_4')(l)
l = Activation('relu')(l)

l= SeparableConv2D(100, (3, 3),name='sep_conv_5')(l)# o/p 100x7x7 RF 20x20
l = BatchNormalization(name='norm_5')(l)
l = Activation('relu')(l)


l = GlobalAveragePooling2D(name='avg_pool')(l)
l = Dense(num_classes, activation='softmax')(l)

model = Model(inputs=[input], outputs=[l])

model.summary()

Model: "model_26"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_42 (InputLayer)        (None, 32, 32, 3)         0         
_________________________________________________________________
sep_conv_1 (SeparableConv2D) (None, 30, 30, 96)        411       
_________________________________________________________________
norm_1 (BatchNormalization)  (None, 30, 30, 96)        384       
_________________________________________________________________
activation_175 (Activation)  (None, 30, 30, 96)        0         
_________________________________________________________________
sep_conv_2 (SeparableConv2D) (None, 28, 28, 96)        10176     
_________________________________________________________________
norm_2 (BatchNormalization)  (None, 28, 28, 96)        384       
_________________________________________________________________
activation_176 (Activation)  (None, 28, 28, 96)        0         
_________________________________________________________________
sep_conv_3 (SeparableConv2D) (None, 26, 26, 192)       19488     
_________________________________________________________________
norm_3 (BatchNormalization)  (None, 26, 26, 192)       768       
_________________________________________________________________
activation_177 (Activation)  (None, 26, 26, 192)       0         
_________________________________________________________________
max_pooling2d_42 (MaxPooling (None, 13, 13, 192)       0         
_________________________________________________________________
dropout_52 (Dropout)         (None, 13, 13, 192)       0         
_________________________________________________________________
sep_conv_6 (SeparableConv2D) (None, 11, 11, 96)        20256     
_________________________________________________________________
norm_6 (BatchNormalization)  (None, 11, 11, 96)        384       
_________________________________________________________________
activation_178 (Activation)  (None, 11, 11, 96)        0         
_________________________________________________________________
dropout_53 (Dropout)         (None, 11, 11, 96)        0         
_________________________________________________________________
sep_conv_4 (SeparableConv2D) (None, 9, 9, 100)         10564     
_________________________________________________________________
norm_4 (BatchNormalization)  (None, 9, 9, 100)         400       
_________________________________________________________________
activation_179 (Activation)  (None, 9, 9, 100)         0         
_________________________________________________________________
sep_conv_5 (SeparableConv2D) (None, 7, 7, 100)         11000     
_________________________________________________________________
norm_5 (BatchNormalization)  (None, 7, 7, 100)         400       
_________________________________________________________________
activation_180 (Activation)  (None, 7, 7, 100)         0         
_________________________________________________________________
avg_pool (GlobalAveragePooli (None, 100)               0         
_________________________________________________________________
dense_61 (Dense)             (None, 10)                1010      
=================================================================
Total params: 75,625
Trainable params: 74,265
Non-trainable params: 1,360
_________________________________________________________________

==============================================================================================================================

EPOCH LOGS :

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:21: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:21: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., callbacks=[<keras.ca..., verbose=1, steps_per_epoch=390, epochs=50)`
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
390/390 [==============================] - 69s 177ms/step - loss: 1.2969 - acc: 0.5330 - val_loss: 1.9364 - val_acc: 0.4483
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
390/390 [==============================] - 58s 149ms/step - loss: 0.9254 - acc: 0.6726 - val_loss: 0.9537 - val_acc: 0.6591
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
390/390 [==============================] - 58s 149ms/step - loss: 0.7947 - acc: 0.7209 - val_loss: 0.9189 - val_acc: 0.6842
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
390/390 [==============================] - 58s 149ms/step - loss: 0.7154 - acc: 0.7486 - val_loss: 0.9009 - val_acc: 0.6917
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
390/390 [==============================] - 58s 148ms/step - loss: 0.6608 - acc: 0.7677 - val_loss: 0.7415 - val_acc: 0.7460
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
390/390 [==============================] - 58s 149ms/step - loss: 0.6159 - acc: 0.7864 - val_loss: 0.7614 - val_acc: 0.7437
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
390/390 [==============================] - 58s 149ms/step - loss: 0.5827 - acc: 0.7955 - val_loss: 0.6986 - val_acc: 0.7640
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
390/390 [==============================] - 58s 149ms/step - loss: 0.5515 - acc: 0.8060 - val_loss: 0.7351 - val_acc: 0.7517
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
390/390 [==============================] - 58s 149ms/step - loss: 0.5280 - acc: 0.8161 - val_loss: 0.6151 - val_acc: 0.7925
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
390/390 [==============================] - 58s 149ms/step - loss: 0.5027 - acc: 0.8249 - val_loss: 0.6957 - val_acc: 0.7705
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
390/390 [==============================] - 58s 149ms/step - loss: 0.4905 - acc: 0.8275 - val_loss: 0.6090 - val_acc: 0.7946
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
390/390 [==============================] - 58s 148ms/step - loss: 0.4723 - acc: 0.8345 - val_loss: 0.6558 - val_acc: 0.7817
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
390/390 [==============================] - 58s 149ms/step - loss: 0.4612 - acc: 0.8388 - val_loss: 0.6297 - val_acc: 0.7885
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
390/390 [==============================] - 58s 150ms/step - loss: 0.4446 - acc: 0.8451 - val_loss: 0.5880 - val_acc: 0.8042
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
390/390 [==============================] - 58s 149ms/step - loss: 0.4355 - acc: 0.8482 - val_loss: 0.6065 - val_acc: 0.7985
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
390/390 [==============================] - 58s 149ms/step - loss: 0.4240 - acc: 0.8503 - val_loss: 0.5881 - val_acc: 0.8047
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
390/390 [==============================] - 58s 149ms/step - loss: 0.4146 - acc: 0.8547 - val_loss: 0.6713 - val_acc: 0.7828
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
390/390 [==============================] - 58s 149ms/step - loss: 0.4039 - acc: 0.8573 - val_loss: 0.5628 - val_acc: 0.8096
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
390/390 [==============================] - 58s 149ms/step - loss: 0.3959 - acc: 0.8608 - val_loss: 0.6153 - val_acc: 0.7976
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
390/390 [==============================] - 58s 149ms/step - loss: 0.3885 - acc: 0.8634 - val_loss: 0.5603 - val_acc: 0.8121
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
390/390 [==============================] - 58s 149ms/step - loss: 0.3779 - acc: 0.8661 - val_loss: 0.6075 - val_acc: 0.8030
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
390/390 [==============================] - 58s 149ms/step - loss: 0.3714 - acc: 0.8684 - val_loss: 0.5787 - val_acc: 0.8071
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
390/390 [==============================] - 58s 149ms/step - loss: 0.3647 - acc: 0.8709 - val_loss: 0.5826 - val_acc: 0.8095
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
390/390 [==============================] - 58s 148ms/step - loss: 0.3597 - acc: 0.8733 - val_loss: 0.6353 - val_acc: 0.7974
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
390/390 [==============================] - 58s 148ms/step - loss: 0.3560 - acc: 0.8742 - val_loss: 0.5730 - val_acc: 0.8160
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
390/390 [==============================] - 58s 149ms/step - loss: 0.3488 - acc: 0.8764 - val_loss: 0.6017 - val_acc: 0.8034
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
390/390 [==============================] - 58s 149ms/step - loss: 0.3465 - acc: 0.8769 - val_loss: 0.6078 - val_acc: 0.8030
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
390/390 [==============================] - 58s 149ms/step - loss: 0.3418 - acc: 0.8799 - val_loss: 0.6035 - val_acc: 0.8070
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
390/390 [==============================] - 58s 149ms/step - loss: 0.3371 - acc: 0.8815 - val_loss: 0.5763 - val_acc: 0.8149
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
390/390 [==============================] - 58s 149ms/step - loss: 0.3308 - acc: 0.8827 - val_loss: 0.6020 - val_acc: 0.8045
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
390/390 [==============================] - 58s 149ms/step - loss: 0.3267 - acc: 0.8849 - val_loss: 0.5919 - val_acc: 0.8121
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
390/390 [==============================] - 58s 149ms/step - loss: 0.3255 - acc: 0.8831 - val_loss: 0.5873 - val_acc: 0.8112
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
390/390 [==============================] - 58s 149ms/step - loss: 0.3237 - acc: 0.8859 - val_loss: 0.5886 - val_acc: 0.8125
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
390/390 [==============================] - 58s 148ms/step - loss: 0.3119 - acc: 0.8905 - val_loss: 0.5582 - val_acc: 0.8168
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
390/390 [==============================] - 58s 148ms/step - loss: 0.3139 - acc: 0.8897 - val_loss: 0.5812 - val_acc: 0.8155
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
390/390 [==============================] - 58s 148ms/step - loss: 0.3082 - acc: 0.8918 - val_loss: 0.6025 - val_acc: 0.8131
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
390/390 [==============================] - 58s 149ms/step - loss: 0.3095 - acc: 0.8900 - val_loss: 0.5824 - val_acc: 0.8141
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
390/390 [==============================] - 58s 149ms/step - loss: 0.3046 - acc: 0.8913 - val_loss: 0.5901 - val_acc: 0.8142
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
390/390 [==============================] - 58s 149ms/step - loss: 0.2994 - acc: 0.8948 - val_loss: 0.5694 - val_acc: 0.8191
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
390/390 [==============================] - 58s 149ms/step - loss: 0.2977 - acc: 0.8944 - val_loss: 0.5776 - val_acc: 0.8204
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
390/390 [==============================] - 58s 149ms/step - loss: 0.2969 - acc: 0.8951 - val_loss: 0.6192 - val_acc: 0.8088
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
390/390 [==============================] - 58s 150ms/step - loss: 0.2924 - acc: 0.8946 - val_loss: 0.5901 - val_acc: 0.8152
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
390/390 [==============================] - 59s 150ms/step - loss: 0.2912 - acc: 0.8966 - val_loss: 0.6087 - val_acc: 0.8119
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
390/390 [==============================] - 58s 149ms/step - loss: 0.2852 - acc: 0.8986 - val_loss: 0.6005 - val_acc: 0.8144
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
390/390 [==============================] - 58s 149ms/step - loss: 0.2891 - acc: 0.8960 - val_loss: 0.6229 - val_acc: 0.8067
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
390/390 [==============================] - 58s 149ms/step - loss: 0.2817 - acc: 0.8993 - val_loss: 0.5923 - val_acc: 0.8146
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
390/390 [==============================] - 58s 149ms/step - loss: 0.2790 - acc: 0.9012 - val_loss: 0.5780 - val_acc: 0.8196
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
390/390 [==============================] - 58s 149ms/step - loss: 0.2807 - acc: 0.9011 - val_loss: 0.5730 - val_acc: 0.8203
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
390/390 [==============================] - 58s 149ms/step - loss: 0.2732 - acc: 0.9028 - val_loss: 0.5986 - val_acc: 0.8170
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
390/390 [==============================] - 58s 149ms/step - loss: 0.2720 - acc: 0.9030 - val_loss: 0.5862 - val_acc: 0.8208
Model took 2916.27 seconds to train

Accuracy on test data is: 82.08




