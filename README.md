# [MambaClinix](https://arxiv.org/abs/2409.12533)

- Official repository for "MambaClinix: Hierarchical Gated Convolution and Mamba-Based U-Net for Enhanced 3D Medical Image Segmentation." [arXiv, 2024](https://arxiv.org/abs/2409.12533)

![Framework](https://github.com/CYB08/MambaClinix-PyTorch/blob/main/assets/Fig1.png)

![Block](https://github.com/CYB08/MambaClinix-PyTorch/blob/main/assets/Fig2.png)

## Installation 

Requirements: `ubuntu 22.04` + `Python 3.10` + `torch 2.0.1` + `torchvision 0.15.2 (cuda 11.8)` 

1. Create a virtual environment: `conda create -n MambaClinix python=3.10 -y` and `conda activate MambaClinix `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install https://github.com/state-spaces/mamba/releases/download/v2.1.0/mamba_ssm-2.1.0+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
4. Download code: `git clone https://github.com/CYB08/MambaClinix-PyTorch.git` and `cd MambaClinix-PyTorch/mambaclinix` and run `pip install -e .`

## Preprocessing

We have released a small portion of sample data from our private dataset. Download dataset [here](https://drive.google.com/drive/folders/111n2yo68O3s7kZFjwo7840B-pdNWkAvG?usp=drive_link) and put them into the `Data` folder.  `MambaClinix` is built on the popular [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework. To preprocess your datasets, please run:

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -np 2
```

 If you want to train on your own dataset, please adjust the base path in `mambaclinix/nnunetv2/paths.py` as follow:

```python
# An example to set base path,
base = '/root/autodl-tmp/Data'    # Using your nnUNet data directory locations
nnUNet_raw = join(base, 'nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') 
nnUNet_results = join(base, 'nnUNet_results') 
```


## Model Training
- Train a `MambaClinix` model

```
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerMambaClinix
```

- Train a `MambaClinix` model with `region-specific Tversky loss`

```
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerMambaClinixTRS
```

## Inference

- Predict samples with `MambaClinix` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres --disable_tta -f all -tr nnUNetTrainerMambaClinix
```

## Evaluate

- Evaluate `MambaClinix` model performance

```bash
nnUNetv2_evaluate_folder GT_FOLDER PRED_FOLDER -djfile Path_Dataset_json -pfile Path_Plans_json
```

`PRED_FOLDER` refers to the path of the prediction file or folder; `GT_FOLDER` is the path of the corresponding ground truth file or folder; `Path_Dataset_json` represents the path of the `dataset.json`, and `Path_Plans_json` represents the path of the `plans.json`.

## Paper

```
@misc{bian2024mambaclinixhierarchicalgatedconvolution,
      title={MambaClinix: Hierarchical Gated Convolution and Mamba-Based U-Net forEnhanced 3D Medical Image Segmentation}, 
      author={Chenyuan Bian and Nan Xia and Xia Yang and Feifei Wang and Fengjiao Wang and Bin Wei and Qian Dong},
      year={2024},
      eprint={2409.12533},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2409.12533}, 
}
```

## Acknowledgements

We greatly appreciate the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) , [Mamba](https://github.com/state-spaces/mamba) , and [U-Mamba](https://github.com/bowang-lab/U-Mamba) projects for open-sourcing their valuable code. Our code is developed based on these outstanding projects.



