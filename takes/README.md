# Takes

## Folder structure

- Naming convention: take folder name defines how to specify it in code
- Takes can have different input types, either simulated hyperspectral data or real hyperspectral data. For the datasets used in the experiments/paper, specific data loaders were implemented. E.g. the urban data loader or hyperspectral data loader. Since this project has grown over time, I didn't start with PyTorch data loaders, as it was simpler to use proprietary code at the beginning. Looking back, seeing how the project grew, PyTorch data loaders would have been the better choice.

## Input types:

### Simulated hyperspectral:
- labeling folder with fully masked labels
- masks folder with scribbles
- channels folder with different simulated channels

### Urban hyperspectral
- labels folder with masked labels
- masks with scribbles
- hyperspectral.mat with image data
- rgb.png for only rgb usage