# from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights, swin_v2_b, Swin_V2_B_Weights, regnet_y_128gf, RegNet_Y_128GF_Weights
import torch as th

def load_classifier(classifier_name='resnet50'):
    if classifier_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        module_names = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'fc']
    elif classifier_name == 'wideresnet50':
        weights = Wide_ResNet50_2_Weights.DEFAULT
        model = wide_resnet50_2(weights=weights)
        model.eval()
        module_names = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'fc']
    elif classifier_name == 'regnet_y_128gf':
        weights = RegNet_Y_128GF_Weights.DEFAULT
        model = regnet_y_128gf(weights=weights)
        model.eval()
        module_names = ['stem', 'trunk_output.block1.block1-0', 'trunk_output.block1.block1-1', 'trunk_output.block2.block2-0', 'trunk_output.block2.block2-1', 'trunk_output.block2.block2-2', 'trunk_output.block2.block2-3', 'trunk_output.block2.block2-4', 'trunk_output.block2.block2-5', 'trunk_output.block2.block2-6', 'trunk_output.block3.block3-0', 'trunk_output.block3.block3-1', 'trunk_output.block3.block3-2', 'trunk_output.block3.block3-3', 'trunk_output.block3.block3-4', 'trunk_output.block3.block3-5', 'trunk_output.block3.block3-6', 'trunk_output.block3.block3-7', 'trunk_output.block3.block3-8', 'trunk_output.block3.block3-9', 'trunk_output.block3.block3-10', 'trunk_output.block3.block3-11', 'trunk_output.block3.block3-12', 'trunk_output.block3.block3-13', 'trunk_output.block3.block3-14', 'trunk_output.block3.block3-15', 'trunk_output.block3.block3-16', 'trunk_output.block4.block4-0', 'fc']
    elif classifier_name == 'vit_b_16':
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model   = vit_b_16(weights=weights)
        model.eval()
        module_names = ['encoder.layers.encoder_layer_0', 'encoder.layers.encoder_layer_1', 'encoder.layers.encoder_layer_2', 'encoder.layers.encoder_layer_3', 'encoder.layers.encoder_layer_4', 'encoder.layers.encoder_layer_5', 'encoder.layers.encoder_layer_6', 'encoder.layers.encoder_layer_7', 'encoder.layers.encoder_layer_8', 'encoder.layers.encoder_layer_9', 'encoder.layers.encoder_layer_10', 'encoder.layers.encoder_layer_11', 'heads']
    elif classifier_name == 'swin_v2_b':
        weights = Swin_V2_B_Weights.DEFAULT
        model   = swin_v2_b(weights=weights)
        model.eval()
        module_names = ['features.0', 'features.1.0', 'features.1.1', 'features.3.0', 'features.3.1', 'features.5.0', 'features.5.1', 'features.5.2', 'features.5.3', 'features.5.4', 'features.5.5', 'features.5.6', 'features.5.7', 'features.5.8', 'features.5.9', 'features.5.10', 'features.5.11', 'features.5.12', 'features.5.13', 'features.5.14', 'features.5.15', 'features.5.16', 'features.5.17', 'features.7.0', 'features.7.1', 'head']
    else:
        raise f"Classifier {classifier_name} not implemented"
    
    preprocess = weights.transforms()
    def preprocess(img):
        img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
        model_preprocess = weights.transforms()
        return model_preprocess(img)

    return model, preprocess, module_names

# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")