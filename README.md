# Usage

```python
network = Network(
    filename = "serendipity.nnue",
    feature_size = 768,
    hidden_size = 1024,
    king_bucket_count = 7,
    king_bucket_layout = [
        0, 0, 1, 1, 2, 2, 3, 3,
        4, 4, 4, 4, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
    ],
    output_bucket_count = 8,
    QA = 255,
    QB = 64,
    SCALE = 400,
)
```
