# YT Toxic Comment Сlassifier
## Джерело Датасету
Датасет взятий з [Kaggle](https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data)

## Огляд
Це вручну позначений набір даних про токсичність, що містить 1000 коментарів, зібраних із відео YouTube про заворушення у Фергюсоні в 2014 році.

### Файли
- **youtoxic_english.csv**
  - *Text*: Коментар.
  - *IsToxic*: Значення True або False.
  - *IsAbusive*: Значення True або False.
  - *IsThreat*: Значення True або False.
  - *IsProvocative*: Значення True або False.
  - *IsObscene*: Значення True або False.
  - *IsHatespeech*: Значення True або False.
  - *IsRacist*: Значення True або False.
  - *IsNationalist*: Значення True або False.
  - *IsHomophobic*: Значення True або False.
  - *IsReligiousHate*: Значення True або False.
  - *IsRadicalism*: Значення True або False.

## Використання
Цей датасет був використаний для нашого проекту класифікатора.
# Comment Toxicity Classification App
Наша програма використовує натреновану модель за допомогою RandomForestClassifier приймаючи від користувача коментар та виводить токсичний це коментар чи ні. Також в нашому проеті було використано TfidfVectorizer - це інструмент для перетворення колекції текстових документів в матрицю TF-IDF (Term Frequency-Inverse Document Frequency). TF-IDF - це статистичний метод для визначення важливості кожного слова в контексті тексту та в межах всього корпусу текстів.
А також RandomOverSampler з бібліотеки imbalanced-learn використовується для вирівнювання дисбалансу класів у вибірці даних. У багатьох випадках, коли у вас є нерівномірна кількість прикладів для кожного класу (даний клас має набагато менше екземплярів, ніж інший), це може призводити до неправильної роботи моделі, яка може переважати в бік класу з більшою кількістю екземплярів.
![App](https://github.com/ViktorPrystai/YTcommentProject/blob/main/screenshots/result%20not%20toxic.jpg)![App](https://github.com/ViktorPrystai/YTcommentProject/blob/main/screenshots/result%20toxic.jpg)

