from keras.layers import Conv2D, Conv2DTranspose, Input, ReLU, Dense, Flatten, Reshape, UpSampling2D, MaxPool2D, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.regularizers import l2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import datetime

class Show_results(Callback):
    def __init__(self, path_to_val_photo):
        super(Callback, self).__init__()

        self.photo = load_img(path_to_val_photo)
        img_array = img_to_array(self.photo)
        self.photo_array = np.expand_dims(img_array, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        predicted_photo = self.model.predict_on_batch(self.photo_array)[0]
        self.photo.show()
        array_to_img(predicted_photo).show()



def get_model(weight_decay=0.00001, neck_dimension=16):
    input_layer = Input(shape=(256, 256, 3))

    x = Conv2D(128, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(input_layer)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(128, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(128, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(256, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(512, (5, 5), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(1024, (3, 3), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    # x = Flatten()(x)
    # # x = Dense(neck_dimension)(x)
    # # a = int(256 / 2**4)
    # x = Dense(neck_dimension*neck_dimension*1024)(x)
    # x = Reshape((neck_dimension, neck_dimension, 1024))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(1024, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = LeakyReLU(0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(3, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), activation='sigmoid')(x)


    return Model(inputs=input_layer, outputs=x)

def main():
    BATCH_SIZE = 1

    model = get_model(neck_dimension=8)
    model.summary()

    img = load_img('/home/szwank/Desktop/faceswap/data/Harrison_Ford_Blade_Runner/Blade Runner (1982) Final Cut 1080p BluRay.x264 SUJAIDR_017440_0.png')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    generator = ImageDataGenerator(samplewise_center=True,
                                   samplewise_std_normalization=True,
                                   # zoom_range=[0, 1.1],
                                   horizontal_flip=True,
                                   rescale=1/255.0
                                   )

    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='mean_absolute_error')

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3, cooldown=3, min_lr=0.000001, verbose=1, min_delta=0.0001)
    show_result = Show_results('/home/szwank/Desktop/faceswap/data/Harrison_Ford_Blade_Runner/Blade Runner (1982) Final Cut 1080p BluRay.x264 SUJAIDR_017440_0.png')

    save_model_relative_path = 'Zapis modelu/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    save_model_absolute_path = os.path.join(os.getcwd(), save_model_relative_path)
    if not os.path.exists(save_model_absolute_path):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(save_model_absolute_path)
    save_best_model = ModelCheckpoint(filepath=save_model_relative_path + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5",
                                      monitor='loss',
                                      save_best_only=True,
                                      period=3
                                      )
    # model.fit_generator(generator.flow(img_array, img_array), epochs=500, steps_per_epoch=1, callbacks=[reduce_lr])
    model.fit_generator(generator.flow_from_directory('data',
                                                      target_size=(256, 256),
                                                      class_mode='input',
                                                      batch_size=BATCH_SIZE,
                                                      ),
                        epochs=100,
                        steps_per_epoch=774/BATCH_SIZE,
                        callbacks=[reduce_lr, save_best_model, show_result]
                        )
    # input_layer = Input((8000000, ))
    # x = Reshape((250, 250, 128))(input_layer)
    # model = Model(inputs=model.layers[3].inputs, outputs=Model.output)
    odp = model.predict_on_batch(img_array)
    array_to_img(odp[0]).show()
    img.show()


if __name__ == "__main__":
    main()