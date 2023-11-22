"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import load_data

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    data_start = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        class_cond=True,
        random_crop=False,
        random_flip=False,
    )

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_start_images = []
    all_noisy_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        batch_start, extra = next(data_start)

        labels_start = extra["y"].to(dist_util.dev())

        batch_start = batch_start.to(dist_util.dev())
        # Sample noisy images from the diffusion process at time t_reverse given by the step_reverse argument
        t_reverse = diffusion._scale_timesteps(th.tensor([args.step_reverse])).to(dist_util.dev())
        batch_noisy = diffusion.q_sample(batch_start, t_reverse)

        model_kwargs = {}
        if args.class_cond:
            classes = labels_start # Condition the diffusion on the labels of the original images
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop_forw_back if not args.use_ddim else diffusion.ddim_sample_loop_forw_back
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            step_reverse = args.step_reverse,  # step when to reverse the diffusion process
            noise=batch_noisy,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        # Save the start images
        batch_start = ((batch_start + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch_start = batch_start.permute(0, 2, 3, 1)
        batch_start = batch_start.contiguous()
        gathered_start_samples = [th.zeros_like(batch_start) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_start_samples, batch_start)  # gather not supported with NCCL
        all_start_images.extend([sample.cpu().numpy() for sample in gathered_start_samples])
        # Save the noised images
        batch_noisy = ((batch_noisy + 1) * 127.5).clamp(0, 255).to(th.uint8)
        batch_noisy = batch_noisy.permute(0, 2, 3, 1)
        batch_noisy = batch_noisy.contiguous()
        gathered_noisy_samples = [th.zeros_like(batch_noisy) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_noisy_samples, batch_noisy)  # gather not supported with NCCL
        all_noisy_images.extend([sample.cpu().numpy() for sample in gathered_noisy_samples])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    arr_start = np.concatenate(all_start_images, axis=0)
    arr_start = arr_start[: args.num_samples]
    arr_noisy = np.concatenate(all_noisy_images, axis=0)
    arr_noisy = arr_noisy[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr, arr_start, arr_noisy)
        else:
            np.savez(out_path, arr, arr_start, arr_noisy)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(dict(
        data_dir =  'datasets/imagenet64_startingImgs',
        step_reverse = 100,
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
