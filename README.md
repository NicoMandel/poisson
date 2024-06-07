# poisson
determinantion of poisson ratio using a video camera

## installation
1. Install environment through `conda create -f environment.yml`
2. install local files through `pip install -e .` in base directory
3. use with: 
    1. `conda activate poisson`
    2. `from poisson import <...>`

## Usage
* Script entrypoints are located in [`scripts`](./scripts/)
* source files that are loaded from the scripts are in [poisson](./src/poisson/)
* Data should be placed in [`data`](./data/), which is ignored by git. please download from elsewhere
* results should be placed [`results`](./results/) which is ignored by git. 