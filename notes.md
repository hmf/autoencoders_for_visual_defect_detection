
# Installing Requirements

<!-- cSpell:disable -->
```shell
hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ python3 -m venv ./venv
hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ source ./venv/bin/activate
(venv) hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ pip install -r requirements.txt
ERROR: Invalid requirement: 'torch~=2.2.1+cu121' (from line 9 of requirements.txt)
(venv) hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ pip install --upgrade pip
Requirement already satisfied: pip in ./venv/lib/python3.12/site-packages (24.0)
Collecting pip
  Using cached pip-25.1.1-py3-none-any.whl.metadata (3.6 kB)
Using cached pip-25.1.1-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 24.0
    Uninstalling pip-24.0:
      Successfully uninstalled pip-24.0
Successfully installed pip-25.1.1
(venv) hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ pip install -r requirements.txt
ERROR: Invalid requirement: 'torch~=2.2.1+cu121': Local version label can only be used with `==` or `!=` operators
    torch~=2.2.1+cu121
         ~~~~~~~^ (from line 9 of requirements.txt)
(venv) hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ python --version
Python 3.12.3
```
<!-- cSpell:enable -->

# Preparing data

<!-- cSpell:disable -->
```shell
(venv) hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ python -m config.data_paths
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection/config/data_paths.py", line 4, in <module>
    from utils.utils import setup_logger
  File "/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection/utils/utils.py", line 21, in <module>
    from pytorch_msssim import SSIM
ModuleNotFoundError: No module named 'pytorch_msssim'
(venv) hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection$ 
```
<!-- cSpell:enable -->


Missing package:

* https://pypi.org/project/pytorch-msssim/
* https://github.com/VainF/pytorch-msssim
* https://pypi.org/project/pytorch-msssim/0.1.1/
  * https://github.com/VainF/pytorch-msssim
* https://github.com/jorge-pessoa/pytorch-msssim (no used)
  * https://github.com/Po-Hsun-Su/pytorch-ssim (dead)

Added to `requirements.txt`:

<!-- cSpell:disable -->
```python
pytorch_msssim~=1.0.0
```
<!-- cSpell:enable -->


Changed `config/json_files/augmentation_config.json`:

<!-- cSpell:disable -->
```json
  "dataset_type": "texture_1",
```
<!-- cSpell:enable -->

Executed:

<!-- cSpell:disable -->
```shell
python -m dataset_operations.augmentation
```
<!-- cSpell:enable -->

Changed `dataset_operations/draw_rectangles.py`:

From:

<!-- cSpell:disable -->
```json
  "dataset_type": "cpu",
```
<!-- cSpell:enable -->

To:

<!-- cSpell:disable -->
```json
  "dataset_type": "texture_1",
```
<!-- cSpell:enable -->

<!-- cSpell:disable -->
```shell
# images_good = file_reader(path_good, "JPG")
images_good = file_reader(path_good, "png")
```
<!-- cSpell:enable -->

Changed `config/json_files/training_config.json`

From:

<!-- cSpell:disable -->
```json
  "network_type": "DAEE",
```
<!-- cSpell:enable -->

To:

<!-- cSpell:disable -->
```json
  "network_type": "AE",
```
<!-- cSpell:enable -->


<!-- cSpell:disable -->
```shell
python -m src.train
```
<!-- cSpell:enable -->


# Data 

1. https://www.kaggle.com/datasets/wardaddy24/marble-surface-anomaly-detection-2
1. https://paperswithcode.com/datasets?mod=images&task=anomaly-detection

# Libraries

1. https://github.com/open-edge-platform/anomalib


# References

1. [Awesome Industrial Anomaly Detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
1. [Divide and Conquer: High-Resolution Industrial Anomaly Detection via Memory Efficient Tiled Ensemble](https://arxiv.org/abs/2403.04932v1)
1. 


