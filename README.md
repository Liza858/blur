# Детектор размытия на фотографиях 


## Инструкция по запуску

Потребуется библиотека openvc версии 3.4 или новее.

Компилятор С++, поддреживающий стандарт C++17.

В Makefile необходимо вставить пути до каталога с библиотекой opencv.

Возможны два режима запуска:

- Получить маску для одного файла:

  ./main -f <source_directory> <file_name> <output_directory>


- Получить маски для датасета из каталога:

  ./main -d <source_directory> <output_directory>


## Guided filter

Реализация guided filter отсюда: https://github.com/atilimcetin/guided-filter


## License 

MIT license.
