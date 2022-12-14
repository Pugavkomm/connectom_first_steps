# Connectom train

**Даный репозиторий создан для тренировки работы с коннектомом**

## Зависиомсти

### Для работы с коннектомом были выбраны стандартные библиотеки:

* > ```networkx``` -- для загрузки данных и обработкой графов и т.п.
* > ```numpy``` -- для работы с данными
* > ```tensorflow(version 2.x.x)``` -- для моделирования на видеокарте (чтобы можно было эффективно запускать моделирование большого количества узлов) (*На самом деле просто хочу вспомнить работу с тензорами из tensorflow*)
* > ```matplotlib``` -- Для отображения
* > ```scipy``` -- Для возможной обработки данных и требуется для некоторых методов из networkx
* > **Замечание:** Требуется установить jupyter notebook (или jupyterLab), если использовать vs code, то установка будет автоматической

### Версии зависиомстей
Все необходимые зависиомсти описаны в файле ```req.txt```:
```python
numpy==1.23.3
scipy==1.9.1
networkx==2.8.6
matplotlib==3.6.0
tensorflow-gpu==2.10.0
```
Версии пакетов (зависимостей) являются последними на 17.09.2022.
Для установки используется ```Pip```: ```pip install -r req.txt```.
### env и установка зависимостей
Для работы с проектом лучше всего использовать python версии 3.10.2. Создаем env:
```bash
python -m venv env
```
Активируем:
**windows**
```bash
.\env\Scripts\activate
```
**Linux**
```
source env/bin/activate
```
Устновка зависимостей
```bash
python -m pip install --upgrade pip
pip install -r req.txt
```
## Структура проекта
```
.
├── LICENSE 
├── README.md
├── connecom
│   ├── node_model
│   ├── utils
├── env_connectom
│   ├── Include
│   ├── Lib
│   ├── Scripts
│   ├── pyvenv.cfg
│   └── share
├── notebooks --   директория с ipynb файлами
├── data -- файл с данными (*.graphml включены в .gitignore)
├── req.txt   --   зависимости (пакеты)
└── result.txt
```
## Описание пакета connectom
Пакет connectom содержит (на текущий момент) содержит два модуля: ```utils``` и ```node_model```. В ```utils``` на текущий момент только метод для помощи в загрузке данных, но в будущем будет расширен. ```node_model``` содержит (на текущий момент) класс ```Kuramoto```, который написан с использованием [TensorFlow](https://www.tensorflow.org/). Не лучший выбор, но позволяет относительно быстро вычислять коннектомы из 700 или более узлов. ~~Возмоно будет переписан с помощью других библиотек.~~ 
> На текущий момент требуется проверка кода!!!
