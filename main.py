import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_depth=5,                 # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    decoder_use_batchnorm=True,
    decoder_channels=[256, 128, 64, 32, 16],
    decoder_attention_type=None,
    activation=None,
    aux_params=None,
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)

from segmentation_models_pytorch.encoders import get_preprocessing_fn

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

