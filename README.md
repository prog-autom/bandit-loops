# Implementation of online recommender with multi-armed bandits and feedback loops

Authors: Anton Pilkevich and Anton Khritankov

## Abstract

We explore hidden feedback loops effect in online recommender systems. Feedback loops result in degradation of online multi-armed bandit (MAB) recommendations to a small subset and loss of coverage and novelty. We study how uncertainty and noise in user interests influence the existence of feedback loops. 
First, we show that an unbiased additive random noise in user interests does not prevent a feedback loop. Second, we demonstrate that a non-zero probability of resetting user interests is sufficient to limit the feedback loop and estimate the size of the effect. 
Our experiments confirm the theoretical findings in a simulated environment for four bandit algorithms.

## Installation

Running experiment with [mldev](https://gitlab.com/mlrep/mldev) involves the following steps.

Install the ``mldev`` by executing

```bash
$ curl https://gitlab.com/mlrep/mldev/-/raw/develop/install_mldev.sh -o install_mldev.sh 
$ chmod +x ./install_mldev.sh && ./install_reqs.sh base
$ mldev version
``` 
Then get the repo
```bash
$ git clone <this repo>
$ cd <this repo folder>
```

Then initialize the experiment, this will install the required dependencies

```bash
$ mldev init -p venv .
```

## Running the experiment

Detailed description of the experiment can be found in [experiment.yml](./experiment.yml). See docs for [mldev](https://gitlab.com/mlrep/mldev) for details.

Run simple experiment for a specific set of params, defined in ``bandit_loops`` in [experiment.yml](./experiment.yml)

```bash
$ mldev run pipeline
```

And now, run the full experiment with params grid explored. See [explore_params.yml](./explore_params.yml) for details.

```bash
$ mldev run run_grid
```

Results will be placed into [./results/explore_params](./results/explore_params) folder.

## Repository contents

Experiment source code can be found in [./src](./src) folder. 
[main.py](./src/main.py) contains an CLI program to run the basic experiment.
[bandits.py](./src/bandits.py) defines bandit algorithms: Thompson Sampling, Epsilon-greedy, Random and Optimal.
[experiment.py](./src/experiment.py)  implements the simulation environment
Results are stored and visualized using [results.py](./src/results.py) for each experiment.
[mathmodel.py](./code/mathmodel.py) contains user interests models.

Source code for integration with MLDev is located in [./.mldev/stages](./.mldev/stages) folder. 
The [explore_params.py](./.mldev/stages/explore_params.py) implements a custom ExploreParams
stage to run the full factorial randomized experiment definded in [explore_params.yml](./explore_params.yml).

[figures](./figures) contain visualizations prepared the paper. 

[notebooks](./notebooks) folder include iPython notebooks that analyse and visualize the 
experiment results. The [mldev_contour_plots](./notebooks/mldev_contour_plots.ipynb) builds
figures for the paper.

## How to cite

Please consider citing the following papers if you find the results useful. 

```
@InProceedings{10.1007/978-3-030-91560-5_19,
    author="Khritankov, Anton and Pilkevich, Anton",
    editor="Zhang, Wenjie and Zou, Lei and Maamar, Zakaria and Chen, Lu",
    title="Existence Conditions for Hidden Feedback Loops in Online Recommender Systems",
    booktitle="Web Information Systems Engineering -- WISE 2021",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="267--274",
    isbn="978-3-030-91560-5"
}


@InProceedings{10.1007/978-3-031-12285-9_1,
    author="Khritankov, Anton and Pershin, Nikita and Ukhov, Nikita and Ukhov, Artem",
    editor="Pozanenko, Alexei and Stupnikov, Sergey and Thalheim, Bernhard and Mendez, Eva and Kiselyova, Nadezhda",
    title="MLDev: Data Science Experiment Automation and Reproducibility Software",
    booktitle="Data Analytics and Management in Data Intensive Domains",
    year="2022",
    publisher="Springer International Publishing",
    address="Cham",
    pages="3--18",
    isbn="978-3-031-12285-9"
}

```
## License

The source code is licensed under the [MIT license](./LICENSE)

## See also

[Initial results](https://github.com/Intelligent-Systems-Phystech/2021-Project-74) for this paper have been developed 
for the "My first scientific paper" course [m1p.org](https://m1p.org). 
