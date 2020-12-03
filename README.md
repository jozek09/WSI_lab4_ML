## WSI - Wprowadzenie do sztucznej inteligencji
---
### LAB 4
---
**Treść zadania**:

Zaimplementować algorytm regresji liniowej i zastosować go do regresji dwóch wybranych zbiorów ze  strony https://archive.ics.uci.edu/ml/datasets.php. Do oceny regresji należy użyć walidacji krzyżowej oraz obliczyć trzy wybrane miary oceny (np. https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b)

---

## Wykorzystane sety danych
* [Auto mpg](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)
* [Cycle power plant](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)

## Etapy zadania
* Wczytanie i obróbka danych(dodanie nazw kolumn, wyczyszczenie brakujących danych, przygotowanie danych)
* Podzielenie zestawu danych na zestaw treningowy i testowy
* Trening modelu - zastosowanie gradientu do obliczenia współczynników
* Test modelu - wykorzystanie MSE, MAE, R2 do obliczenia efektywności 

**TODO**

Zastosowanie walidacji krzyżowej  

## Wykorzystane bilioteki
* Pandas
* Numpy
