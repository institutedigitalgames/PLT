# PLT - Preference Learning Toolbox

> A toolbox for preference learning implemented in Python.

Preference learning (PL) is a core area of machine learning that handles datasets with ordinal relations. As the
number of generated data of ordinal nature such as ranks and subjective ratings is increasing, the importance and
role of the PL field becomes central within machine learning research and practice.

The Preference Learning Toolbox (PLT) is an open source software application and package which supports the key
data modelling phases incorporating various popular data pre-processing, feature selection and preference
learning methods.

![Image](http://plt.institutedigitalgames.com/img/index_plt.png "PLT Screenshot")

PLT may be used either via its GUI or its API. This ``README`` is based on the API which is made available via the Python package **pyplt**. For more information on the GUI or to download the GUI application, please visit the [PLT website](http://plt.institutedigitalgames.com/index.php).

The API documentation may be found at: [https://plt.readthedocs.io/](https://plt.readthedocs.io/en/latest/).

## Features:
* Dataset Pre-processing
* Automatic Feature Selection (SFS)
* Preference Learning Algorithms (RankSVM, ANN-Backpropagation)
* Experiment Reporting and Model Storage

## Installation:

The Python package for PLT, **pyplt**, may be installed via pip:

```bash
pip install pyplt
```

## Usage Example:

The following example loads a dataset in the single file format (refer to [Detailed Guidelines](http://plt.institutedigitalgames.com/howto.php) for more information about file formats) and carries out preference learning using the RankSVM algorithm and K-Fold Cross Validation. At the end, the results are saved to file.

```python
from pyplt.experiment import Experiment
from pyplt.plalgorithms.ranksvm import RankSVM
from pyplt.util.enums import KernelType
from pyplt.evaluation.cross_validation import KFoldCrossValidation
import time

exp = Experiment()

# load ratings data
exp.load_single_data("sample data sets\\single_synth.csv", has_ids=True, has_fnames=True)

# set up RankSVM algorithm
pl_algorithm = RankSVM(kernel=KernelType.RBF, gamma=1)
exp.set_pl_algorithm(pl_algorithm)

# set up K-Fold Cross Validation
pl_evaluator = KFoldCrossValidation(k=3)
exp.set_pl_evaluator(pl_evaluator)

# run the experiment
exp.run()

# save the results
t = time.time()
exp.save_exp_log(t, path="my_results.csv")

```

For more a more detailed usage guide, please check out the [tutorial](http://plt.institutedigitalgames.com/docs/tutorial_experiment.html).

## Development Setup

PLT has the following package dependencies:
* ttkthemes
* numpy
* matplotlib
* pandas
* tensorflow
* scikit_learn
* scipy

These depenencies may be easily istalled via pip:

```bash
pip install -r requirements.txt
```

## Citing

The tool is free for scientific use. If you use PLT in your scientific work, please cite as:

Farrugia, Vincent E., Héctor P. Martínez, and Georgios N. Yannakakis. 
"The Preference Learning Toolbox." arXiv preprint arXiv:1506.01709 (2015)

## License

PLT is licensed under the GNU General Public License v3.0. See ``LICENSE`` for more information.

