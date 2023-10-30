import argparse
import os

import torch

from NestedPipeline import NestedStableDiffusionPipeline
from NestedScheduler import NestedScheduler

def main(seed, inner, outer, fp16, dpm_solver, prompt, outdir):
    scheduler = NestedScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                prediction_type='sample', clip_sample=False, set_alpha_to_one=False)
    if fp16:
        pipe = NestedStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16",
                                                             torch_dtype=torch.float16, scheduler=scheduler)
    else:
        pipe = NestedStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler)
    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)
    pipe.to(device)

    for i, im in enumerate(pipe(prompt, num_inference_steps=outer, num_inner_steps=inner,
                                inner_scheduler="DPMSolver" if dpm_solver else "DDIM", generator=generator)):
        im.images[0].save(os.path.join(outdir, f"ND_{(i+1) * inner}NFEs_{prompt}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--inner', type=int, default=4)
    parser.add_argument('--outer', type=int, default=25)
    parser.add_argument('--outdir', type=str, default="figures")
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--dpm-solver', action='store_true', default=False)
    parser.add_argument('-p', '--prompt', type=str, default="a photograph of a nest with a blue egg inside")
    args = parser.parse_args()
    main(**vars(args))
