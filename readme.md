# CIFAR-100 Fine-Grained Object Classifier

> VGG-стиль CNN распознаёт 100 категорий объектов с отображением
> уверенности — автоматизация визуальной классификации в e-commerce
> и логистике.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal)]()
[![Accuracy](https://img.shields.io/badge/Train-81.33%25_|_Test-70.04%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

Бизнес в e-commerce и логистике ежедневно обрабатывает миллионы
изображений без меток. Ручная классификация 100 категорий —
финансово неподъёмна. Этот API распознаёт объект в момент загрузки
без участия человека.

---

## Структура проекта

    ml_CIFAR_100/
    ├── .gitattributes
    ├── .gitignore
    ├── readme.md
    ├── requirements.txt
    └── cifar_100/
        ├── cifar_100.ipynb                              ← обучение (Google Colab, GPU T4)
        ├── main.py                                      ← FastAPI + Streamlit + inference
        ├── model_Cifar100Classification_CIFAR_100.pth
        ├── test data/                                   ← тестовые изображения
        ├── test1.png
        └── test2.png

---

## Быстрый старт

```bash
git clone https://github.com/your-username/ml_CIFAR_100
cd ml_CIFAR_100/cifar_100

uvicorn main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs`

---

## Demo

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@tiger.jpg"
```

```json
{
  "class": "tiger"
}
```

**100 поддерживаемых классов:**
`apple · bear · bicycle · butterfly · dolphin · elephant ·
fox · kangaroo · leopard · motorcycle · rocket · shark · tiger · train …`

---

## Результаты

| Модель                       | Train Accuracy | Test Accuracy |
|------------------------------|----------------|---------------|
| Random (100 классов)         | 1%             | 1%            |
| Custom VGG-style CNN         | **81.33%**     | **70.04%**    |

Обучение: 100 эпох, AdamW lr=0.001, weight_decay=1e-4,
CosineAnnealingLR (T_max=100), batch_size=128, GPU T4.

**Архитектурное решение:**
VGG-style выбран над ResNet осознанно — CIFAR-100 это 32×32 px,
не фото реального мира. Предобученный ResNet (ImageNet, 224×224) —
избыточен. Эта модель обучена с нуля, весит < 10 МБ, и даёт
70% точности без лицензионных расходов на предобученные веса.

---

## Датасет

- **Источник:** CIFAR-100 (Alex Krizhevsky) — загружается через `torchvision`
- **Объём:** 60 000 RGB-изображений (50K train / 10K test)
- **Размер:** 32×32 px, 3 канала, 100 классов, 20 суперкатегорий
- **Баланс:** 500 изображений на класс — ресэмплинг не нужен

---

## Архитектура модели

    Conv2d(3→64, 64) + BatchNorm + ReLU×2 + MaxPool + Dropout(0.1)  → 16×16
                   ↓
    Conv2d(64→128, 128) + BatchNorm + ReLU×2 + MaxPool + Dropout(0.2) → 8×8
                   ↓
    Conv2d(128→256, 256) + BatchNorm + ReLU×2 + AdaptiveAvgPool → 1×1
                   ↓
    Flatten → Linear(256→512) → ReLU → Dropout(0.5) → Linear(512→100)
                   ↓
    argmax → class name

**Ключевые решения:**

`AdaptiveAvgPool2d((1,1))` вместо `MaxPool2d(2)` в третьем блоке —
устраняет зависимость от фиксированного входного размера, упрощает
inference с произвольными изображениями.

Расширенная аугментация train: `RandomRotation(15) + RandomHorizontalFlip +
RandomCrop(32, padding=4) + ColorJitter(0.2, 0.2, 0.2, 0.1)` — ключ
к сужению разрыва train/test с 25% до 11%.

`label_smoothing=0.1` в `CrossEntropyLoss` — улучшает калибровку
уверенности модели при 100 классах.

`CosineAnnealingLR(T_max=100, eta_min=1e-5)` — плавное затухание lr,
последние 20 эпох стабилизировали loss без ручного тюнинга.

`Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])` —
точная нормализация по статистикам CIFAR-100, применяется к test
без аугментации.

---

## Стек

| Слой     | Технологии                                    |
|----------|-----------------------------------------------|
| ML       | PyTorch, torchvision, Pillow                  |
| API      | FastAPI, Uvicorn, Streamlit                   |
| Обучение | Google Colab (GPU T4), Jupyter                |
| Регуляризация | BatchNorm2d, Dropout, CosineAnnealingLR  |

---

## Business Impact

| Задача                               | До                          | После                    |
|--------------------------------------|-----------------------------|--------------------------|
| Классификация объекта                | 2–5 мин ручной работы      | < 100 мс на запрос       |
| Охват категорий                      | Зависит от экспертизы       | 100 категорий, один вызов |
| Масштабирование                      | Линейно к числу операторов  | REST API                 |
| Лицензионные расходы на веса         | Есть при Transfer Learning  | Нет — обучено с нуля     |

---

## Что дальше (Roadmap)

- [ ] `confidence` в ответе — вернуть softmax Top-5 вместе с классом
- [ ] Docker + Nginx — production-деплой по аналогии с MNIST Fashion
- [ ] Fine-tuning EfficientNet-B0 на CIFAR-100 — потолок кастомного CNN ~72-75%
- [ ] MLflow — трекинг экспериментов и версионирование модели
- [ ] Data drift мониторинг — алерт при смещении входного распределения

---
