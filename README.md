# Условия существования петель скрытой обратной связи в рекомендательных системах

**Автор:** Пилькевич Антон

**Консультант/эксперт:** Хританков Антон Сергеевич

## Постановка задачи

TODO О задаче и этом репозитории. 

Пример.

We running the experiment as show at the figure below 

<img src=".img/experiment-setup.png" alt="experiment setup" width="700"/>

## Как запускать

Running experiment with [mldev](https://gitlab.com/mlrep/mldev) involves the following steps.

Install the ``mldev`` by executing

```bash
$ git clone https://gitlab.com/mlrep/mldev 
$ cd ./mldev && git checkout -b 79-fixes-for-0-3-dev1-exploreparams --track origin/79-fixes-for-0-3-dev1-exploreparams
$ ./install_reqs.sh core
$ python setup.py clean build install
``` 
Then get the repo
```bash
$ git clone <this repo>
$ cd <this repo folder>
```

Then initialize the experiment, this will install required dependencies

```bash
$ mldev init -p venv .
```

Detailed description of the experiment can be found in [experiment.yml](./experiment.yml). See docs for [mldev](https://gitlab.com/mlrep/mldev) for details.

Run simple experiment for a specific set of params

```bash
$ mldev run pipeline
```

And now, run the full experiment with params grid explored. See [explore_params.yml](./explore_params.yml) for details.

```bash
$ mldev run run_grid
```

Results will be placed into [./results](./results) folder.

## Проведение полного эксперимента 

TODO Адаптировать под текущую задачу.

Пример.

There is a script [./run_experiment.sh](./run_experiment.sh) that runs the experiment
for a grid of parameters, usage from 0.1 to 0.9, adherence from 0.1 to 0.9 
and step size 10 or 20.

The script relies on mldev to run trials for a fixed set of parameters.

## Исходный код

TODO Указать, где найти исходники и какие файлы за что отвечают

Пример.

Source code can be found in [./src](./src) folder. The [main.py](./src/main.py) file contains glue code to run experiments.
The [experiment.py](./src/experiment.py) contains experiment implementation and utility procedures.
Visualizations are prepared in [results.py](./src/results.py) by aggregating data from the experiment.
A very draft math model is provided in [mathmodel.py](./src/mathmodel.py).

## Что осталось сделать

TODO Указать, если что-то из задуманного пока не реализованного

Пример.

 - [ ] add a sample iPython notebook 
 - [ ] make the template support arbitrary experiment parameters without rewriting [main.py](./src/main.py)

## Как процитировать работу

TODO Указать ссылку на публикацию или arxiv. Если пока нет публикации дать ссылку на этот репозиторий в формате Bibtex

Пример.

If you find template useful, please cite the original paper as following

```bibtex
@inproceedings{Khritankov2021Hidden,
    title = "Hidden feedback loops in machine learning systems: a simulation model and preliminary results",
    author = "Khritankov, Anton",
    booktitle = "Proceedings of the 13th International Conference, SWQD 2021, Vienna, Austria, January 19–21, 2021",
    year = "2021",
    pages={54--65},
    volume = "404",
    series = "Software Quality: Future Perspectives on Software Engineering Quality",
    publisher = "Springer International Publishing"
}
```

There is an updated version of the paper at [arXiv:2101.05673](https://arxiv.org/abs/2101.05673)

## Лицензия

Пример.

The code is licensed under MIT license, see [LICENSE](LICENSE)
