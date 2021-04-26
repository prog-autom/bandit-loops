# Условия существования петель скрытой обратной связи в рекомендательных системах

**Автор:** Пилькевич Антон

**Консультант/эксперт:** Хританков Антон Сергеевич

## Постановка задачи

Целью работы является теоретический анализ условий сходимости TS для различных параметров шума p, w, u и экспериментальное подтверждение полученых соотношений. 

Целью эксперимента является наблюдение петель скрытой обратной связи для определённых параметров шума. 
Проверяется гипотеза о возникновении петель при параметрах шума, найденных из теоретических соотношений. 
Важной частью эксперимента является сравнения поведений рекомендательной системы с шумом в ответах пользователя и без. 

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
Now install mldev into this venv as follows (need this to run sub-experiment)

```bash
$ /bin/bash -c "source venv/bin/activate; cd ../mldev && python setup.py clean build install"
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

Скрипт [./run_experiment.sh](./run_experiment.sh) запускает эксперимент для параметров T=2000, M=10, l=4. Перебираюся параметры шума w = [1, ..., 9] с фиксированным p = 0.9.

The script relies on mldev to run trials for a fixed set of parameters.

## Исходный код


Пример.

Исходники кода находятся в [./src](./code) .  [main.py](./code/main.py) содержит запуск экспериментов.
[experiment.py](./code/experiment.py) содержит реализацию шаблона проведения эксперимента.
Данные сохраняются при помощи [results.py](./code/results.py) для каждого проведённого эксперимента.
[mathmodel.py](./code/mathmodel.py) cодержит основные компоненты для провдения экспериментов. .

## Что осталось сделать

TODO Указать, если что-то из задуманного пока не реализованного

Пример.

 - [ ] add a sample iPython notebook 
 - [ ] make the template support arbitrary experiment parameters without rewriting [main.py](./src/main.py)

## Как процитировать работу

TODO Указать ссылку на публикацию или arxiv. Если пока нет публикации дать ссылку на этот репозиторий в формате Bibtex

## Лицензия

