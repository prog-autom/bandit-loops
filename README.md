# Условия существования петель скрытой обратной связи в рекомендательных системах

**Автор:** Пилькевич Антон

**Консультант/эксперт:** Хританков Антон Александрович

## Постановка задачи

Целью работы является теоретический анализ условий сходимости TS для различных параметров шума p, w, u и экспериментальное подтверждение полученых соотношений. 

Целью эксперимента является наблюдение петель скрытой обратной связи для определённых параметров шума. 
Проверяется гипотеза о возникновении петель при параметрах шума, найденных из теоретических соотношений. 
Важной частью эксперимента является сравнения поведений рекомендательной системы с шумом в ответах пользователя и без. 

## Как запускать

TODO. Обновить для текущей задачи.

Пример.

There are two experiments included in this repo.

 - single model experiment that demonstrates how housing prices prediction can be solved 
 - hidden loops experiment shows the feedback loop effect as dscribed in the paper

Running the same experiment with [mldev](https://gitlab.com/mlrep/mldev) involves the following steps.

Install the ``mldev`` by executing

```bash
$ curl https://gitlab.com/mlrep/mldev/-/raw/develop/install_mldev.sh -o install_mldev.sh
$ chmod +x install_mldev.sh
$ yes n | install_mldev.sh core
``` 
Then initialize the experiment, this will install required dependencies

```bash
$ mldev init --no-commit -r ./feedback-loops -t https://github.com/prog-autom/template-feedback-loops
```

Detailed description of the experiment can be found in [experiment.yml](./experiment.yml). See docs for [mldev](https://gitlab.com/mlrep/mldev) for details.

And now, run the experiment

```bash
$ cd ./feedback-loops && mldev --config .mldev/config.yaml run --no-commit -f experiment.yml pipeline
```

Results will be placed into [./results](./results) folder.

## Проведение полного эксперимента 

Скрипт [./run_experiment.sh](./run_experiment.sh) запускает эксперимент для параметров T=2000, M=10, l=4. Перебираюся параметры шума w = [1, ..., 9] с фиксированным p = 0.9.

The script relies on mldev to run trials for a fixed set of parameters.

## Исходный код


Пример.

Исходники кода находятся в [./code](./code) .  [main.py](./code/main.py) содержит запуск экспериментов.
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

