import os

from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam

from nets.unet import Unet
from nets.unet_training import CE, Generator, LossHistory, dice_loss_with_CE
from utils.metrics import Iou_score, f_score
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt



if __name__ == "__main__":    
    log_dir = "logs/"
    inputs_size = [512,512,3]
    num_classes = 2
    dice_loss = True
    dataset_path = "VOCdevkit/VOC2007/"

    model = Unet(inputs_size,num_classes)


    model_path = "logs/ep038-loss0.352-val_loss0.504.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    with open(os.path.join(dataset_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()

    with open(os.path.join(dataset_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
        

    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)

    freeze_layers = 17
    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))


    if True:
        lr              = 1e-4
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        Batch_size      = 2

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
        gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, dataset_path).generate(False)

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("Dataset is too small to train")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                validation_data=gen_val,
                validation_steps=epoch_size_val,
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[checkpoint_period, reduce_lr, tensorboard, loss_history])
    
    
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        lr              = 1e-5
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100
        Batch_size      = 2

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, dataset_path).generate()
        gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, dataset_path).generate(False)
        
        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("Dataset is too small to train。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        model.fit_generator(gen,
                steps_per_epoch=epoch_size,
                validation_data=gen_val,
                validation_steps=epoch_size_val,
                epochs=Unfreeze_Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[checkpoint_period, reduce_lr, tensorboard, loss_history])

                
