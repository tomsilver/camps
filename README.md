This repository houses the code for "CAMPs: Learning Context-Specific Abstractions for Efficient Planning in Factored MDPs" by Rohan Chitnis\*, Tom Silver\*, Beomjoon Kim, Leslie Pack Kaelbling, and Tomás Lozano-Pérez. Conference on Robot Learning (CoRL) 2020.

**Paper**: https://arxiv.org/abs/2007.13202
**Video**: https://www.youtube.com/watch?v=wTXt6djcAd4

Installation of dependencies:
- First, run `pip install -r requirements.txt`
- Next, you will need to install the FF classical planner (https://fai.cs.uni-saarland.de/hoffmann/ff.html) and set the environment variable FF_PATH to point to the `ff` executable. If you are on Linux, this can be accomplished via:
  1) curl https://fai.cs.uni-saarland.de/hoffmann/ff/FF-v2.3.tgz --output FF-v2.3.tgz
  2) tar -xzvf FF-v2.3.tgz && cd FF-v2.3 && make && cd ..
  3) export FF_PATH="FF-v2.3/ff"

To run, simply execute `python main.py`. This will, by default, start training in the TAMP NAMO environment. For other environments, planners, and/or methods, change `family_to_run`, `solver_name`, and/or `approach_names` respectively in settings.py.

Learned context-specific independences are packaged with this code, in the learned_csi/ directory. The core CAMP abstraction is implemented in approaches/modelbased.py, with supporting code in csi.py.
