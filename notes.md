
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

# Training


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

Where are the Tensorboard logs being written (`log_dir`)?

<!-- cSpell:disable -->
```shell
find . -type d -iname 'log_dir'
find . -type d -iname "*training*vis*"
find . -type d -iname "*log*"
```
<!-- cSpell:enable -->

None found. Changed the code and ow it prints the following:

<!-- cSpell:disable -->
```shell
tensorboard_log_dir=/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection/results/data/texture_1/model_logs/AE/2025-06-27_10-25-07
```
<!-- cSpell:enable -->


<!-- cSpell:disable -->
```shell
$ export LOG_DIR=/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection/results/data/texture_1/model_logs/AE/2025-06-27_10-25-07
$ tensorboard --logdir $LOG_DIR
$ tensorboard --load_fast=false --logdir $LOG_DIR
```
<!-- cSpell:enable -->

Results:

1. Seems to consistently converge
1. Overfitting observed
   1. Early stopping is  manually coded
   1. Does not check for overfitting or non-divergence
   1. Validation loss keeps lowering slowly (after 1.463 hours loss was 0.168)
1. Some training differences
   1. Model uses a final sigmoid layer in the output (used for "flattening" values?)
   1. "epochs": 200
   1. "img_size": [256, 256]
   1. "crop_it": true
   1. "crop_size": [128, 128]
   1. `optim.Adam`
   1. "batch_size": 128                                           (Different)
   1. "learning_rate": 2e-4                                       (Different)
   1. "latent_space_dimension": 500                               (Different)
   1. Uses a the scheduler `StepLR`                               (Different)
   1. Uses a static dataset of cropped images (12500 images)      (Different)
   1. New models use noise in images for training (small squares) (Different)
   1. Transforms:
      1. `transforms.Grayscale(num_output_channels=1)`
      1. `transforms.ToTensor()`
      1. No scaling (normalization) used                          (Different)
      1. Same for de-noising images

Stopped experiment manually.

# Testing


Changed `config/json_files/training_config.json`

From:

<!-- cSpell:disable -->
```shell
  "network_type": "AEE",
  "dataset_type": "cpu",
  "subtest_folder": "cpua",
  "vis_results": false,
  "vis_reconstruction": false,
```
<!-- cSpell:enable -->


<!-- cSpell:disable -->
```shell
  "network_type": "AE",
  "dataset_type": "texture_1",
  "subtest_folder": "defective",
  "vis_results": true,
  "vis_reconstruction": true,
```
<!-- cSpell:enable -->

<!-- cSpell:disable -->
```shell
python -m src.test
```
<!-- cSpell:enable -->

Male sure the original test images and ground truth are copied to:

<!-- cSpell:disable -->
```shell
test_images_path=/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection/datasets/texture_1/test/defective/test_images
gt_images_path=/mnt/ssd2/hmf/VSCodeProjects/autoencoders_for_visual_defect_detection/datasets/texture_1/test/defective/ground_truth
```
<!-- cSpell:enable -->


<!-- cSpell:disable -->
```shell
```
<!-- cSpell:enable -->


<!-- cSpell:disable -->
```shell
```
<!-- cSpell:enable -->






1. [TensorBoard Histogram Dashboard](https://github.com/tensorflow/tensorboard/blob/master/docs/r1/histograms.md)
1. [The complete guide to ML model visualization with Tensorboard](https://cnvrg.io/tensorboard-guide/)
   1. Implemented by Keras (`rom tensorflow.keras.callbacks import TensorBoard`)
1. [machine-learning-articles/how-to-use-tensorboard-with-pytorch.md](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md)
   1. TODO: example of how to record mol weights


<!-- cSpell:disable -->
```shell
```
<!-- cSpell:enable -->

# Modeling considerations

AEs Regularization `R`:

1. "In **sparse AEs** `R` is based on the Kullback-Leibler divergence or on the L1 norm, the purpose is to make the most of the hidden unit’s activations close to zero."
1. **Contractive AEs** apply the Frobenius norm on the derivative of z as a function of x to make the model resistant to small perturbations; and in an information theoretic-learning autoencoder Renyi’s entropy is used.



1. Physics-Informed Neural Network (PINN)
   1. [Training of Physical Neural Networks](https://arxiv.org/pdf/2406.03372)
   1. [Tutorials for Physics-Informed Neural Networks](https://github.com/nguyenkhoa0209/pinns_tutorial)
   1. [A hands-on introduction to Physics-Informed Neural Networks for solving partial differential equations with benchmark tests taken from astrophysics and plasma physics](https://arxiv.org/abs/2403.00599v1)
   
1. (KAN: Kolmogorov-Arnold Networks)[https://arxiv.org/abs/2404.19756]
   1. [Kolmogorov-Arnold Networks: a Critique](https://medium.com/@rubenszimbres/kolmogorov-arnold-networks-a-critique-2b37fea2112e)
   1. [Exploring the Limitations of Kolmogorov-Arnold Networks in Classification: Insights to Software Training and Hardware Implementation](https://arxiv.org/abs/2407.17790v1)
   1. [Can KAN Work? Exploring the Potential of Kolmogorov-Arnold Networks in Computer Vision](https://arxiv.org/abs/2411.06727v2)
   1. [KAN or MLP: A Fairer Comparison](https://arxiv.org/abs/2407.16674)
   1. [openkan.org/](http://openkan.org/)
1. [Interpretable Deep Learning for New Physics Discovery ](https://www.youtube.com/watch?v=HKJB0Bjo6tQ)
   1. [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287)
   1. [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://github.com/MilesCranmer/symbolic_deep_learning)
1. [Probabilistic Knowledge Transfer for Deep Neural Networks](https://github.com/passalis/probabilistic_kt)



# Data 

1. https://www.kaggle.com/datasets/wardaddy24/marble-surface-anomaly-detection-2
1. https://paperswithcode.com/datasets?mod=images&task=anomaly-detection

# Libraries

1. https://github.com/open-edge-platform/anomalib


# References

1. [Awesome Industrial Anomaly Detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
1. [Divide and Conquer: High-Resolution Industrial Anomaly Detection via Memory Efficient Tiled Ensemble](https://arxiv.org/abs/2403.04932v1)
1. 


