

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

