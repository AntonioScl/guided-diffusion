# from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
import torch as th

def load_classifier(classifier_name='resnet50'):
    if classifier_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        module_names = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'fc']
    elif classifier_name == 'vit_b_16':
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model   = vit_b_16(weights=weights)
        model.eval()
        module_names = ['encoder.layers.encoder_layer_0', 'encoder.layers.encoder_layer_1', 'encoder.layers.encoder_layer_2', 'encoder.layers.encoder_layer_3', 'encoder.layers.encoder_layer_4', 'encoder.layers.encoder_layer_5', 'encoder.layers.encoder_layer_6', 'encoder.layers.encoder_layer_7', 'encoder.layers.encoder_layer_8', 'encoder.layers.encoder_layer_9', 'encoder.layers.encoder_layer_10', 'encoder.layers.encoder_layer_11', 'heads']
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