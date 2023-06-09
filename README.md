# Nested Diffusion Processes for Anytime Image Generation
<a href="https://github.com/noamelata">Noam Elata</a>, <a href="https://bahjat-kawar.github.io/">Bahjat Kawar</a>, <a href="https://tomer.net.technion.ac.il/">Tomer Michaeli</a>, and <a href="https://elad.cs.technion.ac.il/">Michael Elad</a>, Technion.<br />

<img src="figures/Nested_Egg.png" alt="Nested Diffusion" style="width:256px;"/>

🤗  [`Huggingface Demo`](https://huggingface.co/spaces/noamelata/Nested-Diffusion) 

## Sampling from Nested Stable Diffusion
This code implements <a href="https://arxiv.org/abs/2305.19066">Nested Diffusion</a>, used with Stable Diffusion v1-5 and the <a href="https://github.com/huggingface/diffusers"> Diffusers library</a>.

Please refer to `environment.yml` for packages that can be used to run the code. 

Image generation can be run with the following script:
```
python main.py --outer <number of outer steps> --inner <number of inner steps>  \
               --outdir samples --prompt "<your text prompt>"
```
Default parameters reproduce the image shown at the top.

If you have less than 10GB of GPU RAM available please add the `--fp16` argument to the script.


## ImageNet Generation with Nested Diffusion 
Code coming soon...


## References and Acknowledgements
```BibTeX
@article{elata2023nested,
  title={Nested Diffusion Processes for Anytime Image Generation},
  author={Elata, Noam and Kawar, Bahjat and Michaeli, Tomer and Elad, Michael},
  journal={arXiv preprint arXiv:2305.19066},
  year={2023}
}
```
