# Neural Vocoder Model with PyTorch

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-replicate-results">How To Replicate Results</a> •
  <a href="#report">Report</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>


## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=3.11

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/3.11.13/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Install `pre-commit` in case you want to contribute:
   ```bash
   pre-commit install
   ```

## How To Replicate Results

### Training

1. Firstly, download dataset for training. You can do it with script `load_train_dataset.py`:
```bash
python load_train_dataset.py --output YOUR_SAVE_DIR
```
2. In order to log training process [CometML](https://www.comet.com/site/) logger was used. Set up `COMET_API_KEY` env with your CometML token:
```bash
export COMET_API_KEY="<YOUR COMETML API KEY>"
```
3. Run training of the model:
```bash
python train.py data_root=YOUR_SAVE_DIR writer.run_name="EXP_NAME"
```

### Inference and Evaluation
To launch inference and evaluation pipelines from our pre-trained model we suggest to look at `demo.ipynb` Jupyter Notebook, which uses sample dataset of correct structure as an example.

## Report
CometML report can be found [here](https://www.comet.com/mrbebra/nvocoder/reports/PNAtRvgFmUaWkY25pfIy0yoQz)

## Credits

1. This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).
2. **HiFiGAN** model was implemented based on the [original HifiGAN paper](https://arxiv.org/abs/2010.05646)
3. **MSD** model was implemented based on the [original MelGAN paper](https://arxiv.org/abs/1910.06711)

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)



