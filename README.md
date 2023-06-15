 # <p align="center"> <font color=#008000>Fantasia3D</font>: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation </p>

 #####  <p align="center"> [Rui Chen*](https://aruichen.github.io/), [Yongwei Chen*](https://cyw-3d.github.io/), [Ningxin Jiao](https://ningxinj.github.io/), [Kui Jia](http://kuijia.site/)</p>
 ##### <p align="center"> *equal contribution
 
#### <p align="center">[Paper](https://fantasia3d.github.io/assets/Fantasia3D.pdf) | [ArXiv](http://arxiv.org/abs/2303.13873) | [Project Page](https://fantasia3d.github.io/) | [Supp_material](https://fantasia3d.github.io/assets/supp_materials.pdf) | [Video](https://www.youtube.com/watch?v=Xbzl4HzFiNo)</p>



<p align="center">
  <img width="40%" src="assets/head_figure.jpg"/>
</p>

<p align="center"> We are working on releasing the code... 🏗️ 🚧 🔨</p>

# FAQ

***Q1***: *About the use of normal and mask images as the input of stable diffusion model and analysis*

Answer: Our initial hypothesis is that normal and mask images, representing local and silhouette information of shapes respectively, can benefit geometry learning. Additionally, we observed that the value range of the normal map is normalized to (-1, 1), which aligns with the data range required for latent space diffusion. Our empirical studies validate this hypothesis. Further support for our hypothesis comes from the presence of normal images in the LAION-5B dataset used for training Stable Diffusion (see [Website](https://rom1504.github.io/clip-retrieval/?back=https%3A%2F%2Fknn.laion.ai&index=laion5B-H-14&useMclip=false&query=normal+map) for retrieval of normal data in LAION-5B). Therefore, the normal data is not considered an out-of-distribution (OOD) input for stable diffusion. To handle rough and coarse geometry in the early stage of learning, we directly utilize concatenated 64 $\times$ 64 $\times$ 4 (normal, mask) images as the latent code, inspired by Latent-NeRF, to achieve better convergence. However, using the normal map without VAE encoding in the world coordinate system may lead to inconsistencies with the data distribution of the latent space trained by VAE. This mismatch can cause the generated geometry to deviate from the text description in some cases. To address this issue, we employ a data augmentation technique by randomly rotating the normal map rendered from the current view. This approach brings the distribution of the normal map closer to the distribution of latent space data. We experimentally observe that it improves the alignment between the generated geometry and the text description. As the learning progresses, it becomes essential to render the 512 $\times$ 512 $\times$ 3 high-resolution normal image for capturing finer geometry details, and we choose to use normal image only in the later stage. This strategy strikes an accuracy-efficiency balance throughout the geometry optimization process.

***Q2***: *Hypothesis-verification analysis of the disentangled representation*

Answer: Previous methods (e.g., DreamFusion and Magic3D) couple the geometry and appearance generation together, following NeRF. Our adoption of the disentangled representation is mainly motivated by the difference of problem nature for generating surface geometry and appearance. In fact, when dealing with finer recovery of surface geometry from multi-view images, methods (e.g.,  VolSDF, nvdiffrec, etc) that explicitly take the surface modeling into account triumph; our disentangled representation enjoys the benefit similar to these methods. The disentangled representation also enables us to include the BRDF material representation in the appearance modeling, achieving better photo-realistic rendering by the BRDF physical prior.

# Notes

**If you have any questions, please feel free to raise them in the issue and we are happy to help you resolve them.**

# Install

- System requirement: Ubuntu20.04
- Tested environment: RTX3090, RTX4090

```bash
git clone https://github.com/Gorilla-Lab-SCUT/Fantasia3D.git
cd Fantasia3D
```

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

# Start
soon....
- zero-shot generation

```bash

```

- user-guided generation

```bash

```
# Coordinate System

<img width="30%" src="assets/coordinate_system.jpg"/>

## Demos

You can download and watch some demos' training process in [Google drive](https://drive.google.com/drive/folders/1cEjXOF_uUSRVRZHE2RDt15CnY9yovEYs?usp=sharing)

<div>
  <video src="https://user-images.githubusercontent.com/128572637/244950828-21956cae-e6c4-42ce-89cd-a912c271de51.mp4" width="400" controls></video>
  <video src="https://user-images.githubusercontent.com/128572637/244950909-0eb363f6-9bf3-4553-9090-fd1fd0003d67.mp4" width="400" controls></video>
</div>

https://user-images.githubusercontent.com/128572637/af266a61-afd4-451b-b4b8-89e77e96233e

https://user-images.githubusercontent.com/128572637/c0a09f43-c07f-43e9-ab9f-c49aa3bc3e2c

https://user-images.githubusercontent.com/128572637/0071b97a-93ce-4332-9f80-a3297b54f8c3

https://user-images.githubusercontent.com/128572637/27d2bce3-f126-4f91-9bcd-1199563618e8

https://user-images.githubusercontent.com/128572637/4c3e3783-2297-4b52-b67d-3c5cff4db4f4

https://user-images.githubusercontent.com/128572637/5d8f7b7f-141d-4800-8772-8fc132522390

https://user-images.githubusercontent.com/128572637/3e23c5f1-31d8-49a8-9013-123a6e97ac3b

https://user-images.githubusercontent.com/128572637/162adc7d-a416-49e5-8dde-73590119b1a9

# Tip

## Todo

- [ ] Release the code. (2023.06.15) (The code has been organized. We are working on the configuration files to facilitate the reproduction of the results in the paper)
- [ ] Support the VSD loss proposed by ProlificDreamer.

## BibTex
```
@article{chen2023fantasia3d,
    title={Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation},
    author={Rui Chen and Yongwei Chen and Ningxin Jiao and Kui Jia},
    journal={arXiv preprint arXiv:2303.13873},
    year={2023}
}
```