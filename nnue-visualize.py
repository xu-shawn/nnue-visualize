import chess
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from typing import List
from numpy.typing import NDArray


class Network:
    def __init__(
        self,
        filename: str,
        feature_size: int,
        hidden_size: int,
        king_bucket_count: int = 1,
        king_bucket_layout: List[int] = [0] * 64,
        output_bucket_count: int = 1,
        QA: int = 255,
        QB: int = 64,
        SCALE: int = 400,
    ):
        data: bytes = pathlib.Path(filename).read_bytes()
        self.raw: NDArray[np.int16] = np.frombuffer(data, dtype="<i2")
        self.hidden_size: int = hidden_size
        self.feature_size: int = feature_size
        self.king_bucket_count: int = king_bucket_count
        self.king_bucket_layout: List[int] = king_bucket_layout
        self.output_bucket_count: int = output_bucket_count
        self.QA: int = QA
        self.QB: int = QB
        self.SCALE: int = SCALE

        self.ft_weights: NDArray[np.int16]
        self.ft_biases: NDArray[np.int16]
        self.output_weights: NDArray[np.int16]
        self.output_bias: NDArray[np.int16]

        self.load_network(
            feature_size, hidden_size, king_bucket_count, output_bucket_count
        )

    def load_network(
        self,
        feature_size: int,
        hidden_size: int,
        king_bucket_count: int,
        output_bucket_count: int,
    ):
        offset: int = 0
        ft_weights_size: int = feature_size * hidden_size * king_bucket_count
        ft_biases_size: int = hidden_size
        output_weights_size: int = output_bucket_count * hidden_size * 2
        output_bias_size: int = output_bucket_count

        self.ft_weights = self.raw[:ft_weights_size].reshape(
            king_bucket_count, feature_size, hidden_size
        )

        offset += ft_weights_size
        self.ft_biases = self.raw[offset : offset + ft_biases_size].reshape(
            ft_biases_size
        )

        offset += ft_biases_size
        self.output_weights = (
            self.raw[offset : offset + output_weights_size]
            .reshape(hidden_size * 2, output_bucket_count)
            .T
        )

        offset += output_weights_size
        self.output_bias = self.raw[offset : offset + output_bias_size]

    def feature_index(
        self, piece: chess.Piece, square: chess.Square, perspective: chess.Color
    ):
        if perspective == chess.BLACK:
            side_is_black: bool = piece.color
            square = square ^ 0b111000
        else:
            side_is_black: bool = not piece.color
        return square + int(piece.piece_type - 1) * 64 + (384 if side_is_black else 0)

    def visualize(
        self,
        piecetype: chess.PieceType,
        color: chess.Color,
        neuron_index: int,
        king_bucket_index: int,
        absolute_value: bool = False,
        file_name: str | None = None,
        x_label: str | None = None,
    ):
        intensity: NDArray[np.int32] = np.zeros((8, 8), dtype=np.int32)

        for square in chess.SQUARES:
            piece: chess.Piece = chess.Piece(piecetype, color)

            neuron_intensity: int = self.ft_weights[
                king_bucket_index,
                self.feature_index(piece, square, chess.WHITE),
                neuron_index,
            ]

            if absolute_value:
                neuron_intensity = abs(neuron_intensity)

            intensity[7 - chess.square_rank(square), chess.square_file(square)] = (
                neuron_intensity
            )

        self.display(
            intensity,
            file_name=file_name,
            x_label=x_label,
        )

    def display(
        self,
        intensity: NDArray[np.int32],
        file_name: str | None = None,
        x_label: str | None = None,
    ):

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(intensity, cmap="magma", interpolation="none")

        if x_label:
            ax.set_xlabel(x_label)

        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()

        if file_name:
            fig.savefig(
                file_name,
                pad_inches=0,
                transparent=True,
                dpi=1200,
            )

    def get_king_bucket_index(self, board: chess.Board) -> tuple[int, int]:
        white_king_bucket: int = 0
        black_king_bucket: int = 0

        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.KING:
                if piece.color == chess.WHITE:
                    white_king_bucket = self.king_bucket_layout[square]
                else:
                    black_king_bucket = self.king_bucket_layout[square ^ 0b111000]

        return (black_king_bucket, white_king_bucket)

    def get_output_bucket_index(self, board: chess.Board) -> int:
        piece_count: int = board.occupied.bit_count()
        divisor: int = (32 + self.output_bucket_count - 1) // self.output_bucket_count
        return (piece_count - 2) // divisor

    def evaluate(self, board: chess.Board):
        accumulators: List[NDArray[np.int16]] = [
            self.ft_biases.copy(),
            self.ft_biases.copy(),
        ]

        king_bucket_indices: tuple[int, int] = self.get_king_bucket_index(board)
        output_bucket_index: int = self.get_output_bucket_index(board)

        for square, piece in board.piece_map().items():

            accumulators[chess.WHITE] += self.ft_weights[
                king_bucket_indices[chess.WHITE],
                self.feature_index(piece, square, chess.WHITE),
            ]

            accumulators[chess.BLACK] += self.ft_weights[
                king_bucket_indices[chess.BLACK],
                self.feature_index(piece, square, chess.BLACK),
            ]

        total = np.sum(
            accumulators[board.turn].clip(0, self.QA).astype(np.int32) ** 2
            * self.output_weights[output_bucket_index][: self.hidden_size]
        ) + np.sum(
            accumulators[not board.turn].clip(0, self.QA).astype(np.int32) ** 2
            * self.output_weights[output_bucket_index][self.hidden_size :]
        )

        value = (
            (total // self.QA + self.output_bias[output_bucket_index])
            * self.SCALE
            // (self.QA * self.QB)
        )

        return value
