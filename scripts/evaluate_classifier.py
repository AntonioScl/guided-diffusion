import argparse
import os
import torch as th
import pickle
import copy
import numpy as np
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    # model_and_diffusion_defaults,
    # create_model_and_diffusion,
    add_dict_to_argparser,
    # args_to_dict,
    # create_classifier,
    # classifier_defaults,
)
from guided_diffusion.image_datasets import load_data
from guided_diffusion.torch_classifiers import load_classifier

# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.output)

    classifier, preprocess, module_names = load_classifier()
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def class_eval(x):
        with th.no_grad():
            logits = classifier(preprocess(x))
            return logits
        
    #Upload activations statistics
    file = '/home/sclocchi/guided-diffusion/results/classifier_statistics/act_stat_resnet50.pk'
    with open(file,'rb') as f: act_stat = pickle.load(f)
    activations_mean = act_stat['activations_mean']
    activations_var = act_stat['activations_var']
    for key in activations_mean.keys():
        activations_mean[key] = th.tensor(activations_mean[key]).to(dist_util.dev())
        activations_var[key] = th.tensor(activations_var[key]).to(dist_util.dev())

    def whiten_act(aa):
        for key in activations_mean.keys():
            aa[key] = (aa[key] - activations_mean[key]) / th.sqrt(activations_var[key] + 1e-8)
            # if activations_mean[key].ndim>1:
            #     flag = activations_var[key] != 0.0
            #     expanded_flag = flag.expand_as(aa[key]) 
            #     aa[key][flag] = (aa[key][expanded_flag] - activations_mean[key][flag]) / th.sqrt(activations_var[key][flag] + 1e-8)
            # else:
            #     aa[key] = (aa[key] - activations_mean[key]) / th.sqrt(activations_var[key] + 1e-8)
        return aa

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    for layer_name in module_names:
        layer = dict([*classifier.named_modules()])[layer_name]
        hook = layer.register_forward_hook(get_activation(layer_name))
        hooks.append(hook)



    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     deterministic=True,
    #     class_cond=True,
    #     random_crop=False,
    #     random_flip=False,
    # )

    def process(X):
        return X/127.5 - 1.0

    time_series = [75, 100, 125, 150, 175, 200, 225, 249]
    for time_step in time_series:
        # file_data = f'/home/sclocchi/guided-diffusion/results/classifier/256x256-classes1/samples_8x256x256x3_{time_step}.npz'
        file_data = f'/home/sclocchi/guided-diffusion/results/classifier/256x256/samples_4x256x256x3_{time_step}.npz'
        data = np.load(file_data)
        batch_data = process(th.tensor(data['arr_0']).permute(0,3,1,2)).to(dist_util.dev())
        start_data = process(th.tensor(data['arr_2']).permute(0,3,1,2)).to(dist_util.dev())

        class_eval_batch  = class_eval(batch_data)
        activations_sample = copy.deepcopy(whiten_act(activations))
        class_eval_start  = class_eval(start_data)
        activations_start = copy.deepcopy(whiten_act(activations))

        diff_activations = {}
        cosine_sim = th.nn.CosineSimilarity(dim=1, eps=1e-8)
        for key in activations_start.keys():
            diff_activations[key] = {}
            activations_sample[key] = activations_sample[key].flatten(start_dim=1)
            activations_start[key]  = activations_start[key].flatten(start_dim=1)
            diff_activations[key]['L2'] = th.linalg.norm(activations_sample[key] - activations_start[key], dim=1)**2
            diff_activations[key]['L2_normalized'] = diff_activations[key]['L2'] / (th.linalg.norm(activations_sample[key], dim=1) * th.linalg.norm(activations_start[key], dim=1))
            diff_activations[key]['cosine'] = cosine_sim(activations_sample[key], activations_start[key])


        logger.log(f"evaluated {batch_data.shape[0]} samples")

        for key1 in diff_activations.keys():
            for key2 in diff_activations[key1].keys():
                diff_activations[key1][key2] = diff_activations[key1][key2].cpu().numpy()

        out_act = os.path.join(logger.get_dir(), f"new_measure-class-acts_{time_step}.pk")
        logger.log(f"saving activations to {out_act}")
        with open(out_act, 'wb') as handle: pickle.dump(diff_activations, handle)

    # Clean up
    for hook in hooks:
        hook.remove()
    
    dist.barrier()
    logger.log("Done!")


def create_argparser():
    defaults = dict(
        classifier_name='resnet50',
        classifier_use_fp16=False,
        num_samples=10000,
        batch_size=128,
        data_dir='datasets/ILSVRC2012/validation',
        image_size=256,
        output  =  os.path.join(os.getcwd(),
        'results',
        'classifier_statistics')
    )
    # defaults.update(dict(
    #     step_reverse = 100,
    #     classifier_path = 'models/64x64_classifier.pt',
    #     data_dir =  'datasets/imagenet64_startingImgs',
    #     output  =  os.path.join(os.getcwd(),
    #          'results',
    #          'forw_back',
    #          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")),
    # ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()