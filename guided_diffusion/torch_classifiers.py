# from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch as th

def load_classifier():
    # img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    def preprocess(img):
        img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
        model_preprocess = weights.transforms()
        return model_preprocess(img)

    # Step 3: Apply inference preprocessing transforms
    # batch = preprocess(img).unsqueeze(0)

    module_names = ['layer1.0', 'layer1.1', 'layer1.2', 'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2', 'fc']

    return model, preprocess, module_names

# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")