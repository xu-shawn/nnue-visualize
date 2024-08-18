# Python NNUE Visualizer

## Usage

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

```python
network.visualize(chess.BISHOP, chess.WHITE, 901, 0)
```

```python
network.evaluate(chess.Board())
```

## Advanced Usage

```python
plt.style.use('dark_background')

fig, ax = plt.subplots(32, 32, figsize=(32, 32))

for i in range(32):
    for j in range(32):
        neuron = i * 32 + j
        network.visualize(chess.BISHOP, chess.WHITE, neuron, 0, ax = ax[i, j], x_label = f"{neuron}")
        ax[i, j].axis('off')

fig.tight_layout()
fig.savefig(
    "bishop_mosaic",
    pad_inches=0,
)
```
