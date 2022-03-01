# AutoML code for reproducibility

> This file lists the steps to reproduce the experiments, analysis and figures generated for this paper submission at the AutoML 2022 conference.


## Contents

This repository already comes with the datasets employed in the experiments along with the code to reproduce them. We also provide the experimental results (see *.csv* files) and figures used in the papers (see *analysis* folder).


## Installation steps

First, you need to install conda to manage the environment. See installation instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

The next step is to create the environment. For that, run:

```shell
conda create --name automl_experiments python=3.9.0
```

Then, activate the environment:

```shell
conda activate automl_experiments
```

Finally, install the dependencies:

```shell
python -m pip install -r requirements.txt
```

```shell
conda install -c conda-forge openjdk
```

## Reproducing experiments

The first step is to run the clustering pipeline for all event logs and extracting the quality criteria using the following command:

```shell
python clustering.py
```

As a result, a *clustering_metrics.csv* file will be generated. For the next step, we convert the event logs into the *xes* format:

```shell
python convert.py
```

The above command generates the *event_logs_xes* folder containing the converted event logs. With that, we proceed to the next step:

```shell
python extract_features.py
```

This extracts the event log meta-features and store them in the *log_meta_features.csv* file. Finally, we move to:

```shell
python analysis.py
```

This command generates the main experiments used in the paper. The products are (i) the *analysis* folder containing the figures used in the paper and (ii) the terminal outputs listing performances and experiment details.
