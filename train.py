#-*-coding:utf-8-*-
from utils.setting import parse_anchor_parameter
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras
import os

from models.retinanet import retinanet_bbox
from utils.transform import random_transform_generator
from utils.image import random_visual_effect_generator
from preprocessing.loader import TrainLoader, ValLoader
import losses
import models

def create_models(backbone_retinanet, num_classes, multi_gpu,
                  lr=1e-5, config=None, weights=None, snapshot=None):
    if config is None:
        config = parse_anchor_parameter()
    anchor_config = config

    num_anchors = anchor_config.num_anchors()

    from config.fileconfig import cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.GPU
    if multi_gpu > 1:
        with tf.device("/cpu:0"):
            if snapshot is not None:
                print("load model")
                print(cfg.FILE.BASE_DIR + "checkpoint/"+snapshot)
                model = models.load_model(cfg.FILE.BASE_DIR + "checkpoint/"+snapshot,
                                          backbone_name="resnet50")
            else:
                model = backbone_retinanet(num_classes, num_anchors=num_anchors)
                if weights is not None:
                    print(weights)
                    model.load_weights(weights, by_name=True, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
        # training_model = model
    else:
        model = backbone_retinanet(num_classes, num_anchors=num_anchors)
        if weights is not None:
            model.load_weights(weights, by_name=True, skip_mismatch=True)
        training_model = model

    # prediction_model = retinanet_bbox(model=model, anchor_config=anchor_config)

    training_model.compile(
        loss=[
            losses.smooth_l1_initializer(),
            losses.focal_loss_initializer()],
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, None

def create_generators(preprocess_image):
    from config.fileconfig import cfg
    common_args = {
        "batch_size": cfg.TRAIN.BATCH_SIZE,
        "image_min_side": cfg.TRAIN.IMAGE_MIN_SIDE,
        "image_max_side": cfg.TRAIN.IMAGE_MAX_SIDE,
        "preprocess_image": preprocess_image,
    }

    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )

    visual_effect_generator = random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)
    )

    train_loader = TrainLoader(transform_generator=transform_generator,
                               visual_effect_generator=visual_effect_generator,
                               **common_args)
    val_loader = ValLoader(**common_args)

    return train_loader, val_loader

def create_callbacks(model, batch_size, backbone):

    callbacks = []

    from config.fileconfig import cfg
    tensorboard_dir = cfg.FILE.BASE_DIR + cfg.FILE.TENSORBOARD_DIR

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir= tensorboard_dir,
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=False,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        update_freq="batch",
    )
    callbacks.append(tensorboard_callback)
    checkpoint_path = cfg.FILE.BASE_DIR + cfg.FILE.SNAPSHOT_PATH
    # checkpoint = keras.callbacks.ModelCheckpoint(
    #     os.path.join(checkpoint_path,
    #                  '{backbone}_{{epoch:02d}}_beta.h5'.format(backbone=backbone)),
    #     verbose=1,
    #     # save_best_only=True,
    #     # monitor="mAP",
    #     # mode="max"
    # )
    from backend.helper import ParallelModelCheckpoint
    checkpoint = ParallelModelCheckpoint(model,
                                         filepath=os.path.join(checkpoint_path,
                     '{backbone}_{{epoch:02d}}_alpha.h5'.format(backbone=backbone)),
                                         verbose=1)
    callbacks.append(checkpoint)

    # callbacks.append(LearningRateIter(verbose=1))

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=2,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    return callbacks

def main():
    keras.backend.clear_session()
    backbone = models.model("resnet50")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

    train_loader, val_loader = create_generators(backbone.preprocess_image)

    from config.fileconfig import cfg
    weights = cfg.TRAIN.IMAGENET_RESNET50_WEIGHTS
    # weights = cfg.TRAIN.IMAGENET_MOBILENET224_1_0

    model, training_model,_ = create_models(backbone_retinanet=backbone.net,
                                            num_classes=train_loader.num_classes(),
                                            # snapshot="resnet50_01_alpha.h5",
                                            multi_gpu=cfg.TRAIN.MULTI_GPU,
                                            lr=cfg.TRAIN.LEARNING_RATE,
                                            weights=weights)

    model.summary()

    callbacks = create_callbacks(model, cfg.TRAIN.BATCH_SIZE, "resnet50")
    steps_per_epoch = len(train_loader)
    print(steps_per_epoch)

    return training_model.fit_generator(
        generator=train_loader,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.TRAIN.EPOCHS,
        verbose=1,
        workers=cfg.TRAIN.WORKERS,
        use_multiprocessing=cfg.TRAIN.MULTIPROCESSING,
        max_queue_size=cfg.TRAIN.MAX_QUEUE_SIZE,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()

