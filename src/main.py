from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    BLACK = "black"
    WHITE = "white"


@dataclass(frozen=True)
class Position:
    row: int
    col: int

    @classmethod
    def from_chess_notation(cls, notation: str) -> Position:
        """Convert chess notation (e.g. 'A1' to Position)."""
        col = ord(notation[0].upper()) - ord("A")
        row = 8 - int(notation[1])
        return cls(row, col)

    def to_chess_notation(self) -> str:
        """Convert Position to chess notation."""
        col = chr(ord("A") + self.col)
        row = 8 - self.row
        return f"{col}{row}"

    def is_valid(self) -> bool:
        """Check if position is within board boundaries."""
        return 0 <= self.row < 8 and 0 <= self.col < 8


class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"


class ChessPiece(ABC):
    """Abstract base class for chess pieces using Template Method pattern."""

    def __init__(self, color: Color):
        self.color = color
        self._has_moved = False

    @abstractmethod
    def _get_possible_moves(self, position: Position, board: Board) -> list[Position]:
        """Template method for getting all theoretically possible moves."""
        pass

    def get_valid_moves(self, position: Position, board: Board) -> list[Position]:
        """Template method that filters possible moves based on board state."""
        moves = self._get_possible_moves(position, board)
        return [
            move
            for move in moves
            if (
                move.is_valid()
                and (
                    board.get_piece_at(move) is None
                    or board.get_piece_at(move).color != self.color
                )
            )
        ]

    def __str__(self):
        symbol = self.__class__.__name__[0]
        return symbol.upper() if self.color == Color.WHITE else symbol.lower()


class MoveStrategy(ABC):
    """Strategy pattern for different movement patterns."""

    @abstractmethod
    def get_moves(self, position: Position, board: Board) -> list[Position]:
        pass


class LineStrategy(MoveStrategy):
    def __init__(self, directions):
        self.directions = directions

    def get_moves(self, position: Position, board: Board) -> list[Position]:
        moves = []
        for dx, dy in self.directions:
            current = Position(position.row + dx, position.col + dy)
            while current.is_valid():
                moves.append(current)
                if board.get_piece_at(current) is not None:
                    break
                current = Position(current.row + dx, current.col + dy)
        return moves


class StraightLineStrategy(LineStrategy):
    def __init__(self):
        super().__init__([(0, 1), (0, -1), (1, 0), (-1, 0)])


class DiagonalStrategy(LineStrategy):
    def __init__(self):
        super().__init__([(1, 1), (1, -1), (-1, 1), (-1, -1)])


class Pawn(ChessPiece):
    def _get_possible_moves(self, position: Position, board: Board) -> list[Position]:
        moves = []
        direction = -1 if self.color == Color.WHITE else 1

        # Forward move
        forward = Position(position.row + direction, position.col)
        if forward.is_valid() and board.get_piece_at(forward) is None:
            moves.append(forward)
            # Double move from starting position
            if not self._has_moved:
                double_forward = Position(position.row + 2 * direction, position.col)
                if board.get_piece_at(double_forward) is None:
                    moves.append(double_forward)

        for dcol in [-1, 1]:
            capture = Position(position.row + direction, position.col + dcol)
            if capture.is_valid() and board.get_piece_at(capture) is not None:
                moves.append(capture)

        return moves


class Rook(ChessPiece):
    def __init__(self, color):
        super().__init__(color)
        self._strategy = StraightLineStrategy()

    def _get_possible_moves(self, position: Position, board: Board) -> list[Position]:
        return self._strategy.get_moves(position, board)


class Bishop(ChessPiece):
    def __init__(self, color):
        super().__init__(color)
        self._strategy = DiagonalStrategy()

    def _get_possible_moves(self, position: Position, board: Board) -> list[Position]:
        return self._strategy.get_moves(position, board)


class Queen(ChessPiece):
    def __init__(self, color: Color):
        super().__init__(color)
        self._straight_strategy = StraightLineStrategy()
        self._diagonal_strategy = DiagonalStrategy()

    def _get_possible_moves(self, position: Position, board: Board) -> list[Position]:
        return self._straight_strategy.get_moves(
            position, board
        ) + self._diagonal_strategy.get_moves(position, board)


class Knight(ChessPiece):
    def _get_possible_moves(self, position: Position, board: Board) -> list[Position]:
        moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        return [
            Position(position.row + dx, position.col + dy)
            for dx, dy in moves
            if Position(position.row + dx, position.col + dy).is_valid()
        ]


class King(ChessPiece):
    def _get_possible_moves(self, position: Position, board: Board) -> list[Position]:
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [
            Position(position.row + dx, position.col + dy)
            for dx, dy in moves
            if Position(position.row + dx, position.col + dy).is_valid()
        ]


class PieceFactory:
    """Factory pattern for creating chess pieces."""

    _piece_classes: dict[PieceType, type] = {
        PieceType.PAWN: Pawn,
        PieceType.ROOK: Rook,
        PieceType.KNIGHT: Knight,
        PieceType.BISHOP: Bishop,
        PieceType.QUEEN: Queen,
        PieceType.KING: King,
    }

    @classmethod
    def create_piece(cls, piece_type: PieceType, color: Color) -> ChessPiece:
        piece_class = cls._piece_classes.get(piece_type)
        if piece_class is None:
            raise ValueError(f"Invalid piece type: {piece_type}")
        return piece_class(color)


class Board:
    """Board class using Singleton pattern."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._board: list[list[ChessPiece | None]] = [
            [None for _ in range(8)] for _ in range(8)
        ]
        self._setup_board()

    def _setup_board(self):
        # Setup pawns
        for col in range(8):
            self._board[1][col] = PieceFactory.create_piece(PieceType.PAWN, Color.BLACK)
            self._board[6][col] = PieceFactory.create_piece(PieceType.PAWN, Color.WHITE)

            # Setup other pieces
        piece_order = [
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        ]

        for col, piece_type in enumerate(piece_order):
            self._board[0][col] = PieceFactory.create_piece(piece_type, Color.BLACK)
            self._board[7][col] = PieceFactory.create_piece(piece_type, Color.WHITE)

    def get_piece_at(self, position: Position) -> ChessPiece | None:
        return self._board[position.row][position.col]

    def move_piece(self, start: Position, end: Position) -> bool:
        piece = self.get_piece_at(start)
        if piece and end in piece.get_valid_moves(start, self):
            self._board[end.row][end.col] = piece
            self._board[start.row][start.col] = None
            piece._has_moved = True
            return True
        return False

    def display(self):
        print("  A B C D E F G H")
        for row_idx, row in enumerate(self._board):
            print(f"{8 - row_idx} ", end="")
            for piece in row:
                symbol = str(piece) if piece else "."
                print(f"{symbol} ", end="")
            print()


class GameState:
    """Observer pattern for game state changes."""

    def __init__(self):
        self._observers = []
        self.current_turn = Color.WHITE
        self.board = Board()

    def add_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.update(self)
            # observer.display(self)

    def switch_turn(self):
        self.current_turn = (
            Color.BLACK if self.current_turn == Color.WHITE else Color.WHITE
        )
        self.notify_observers()


class ChessGame:
    """Mediator pattern for game flow control."""

    def __init__(self):
        self.state = GameState()
        self.state.add_observer(self)

    def play_turn(self, start: Position, end: Position) -> bool:
        piece = self.state.board.get_piece_at(start)
        if piece and piece.color == self.state.current_turn:
            if self.state.board.move_piece(start, end):
                self.state.switch_turn()
                return True
        return False

    def update(self, game_state: GameState):
        self.display()

    def display(self):
        print(f"\nTurn: {self.state.current_turn.value}")
        self.state.board.display()


class ChessController:
    """Contoller pattern for handling user input."""

    def __init__(self):
        self.game = ChessGame()

    def play(self):
        while True:
            self.game.display()
            try:
                start_notation = input("Enter start position (e.g. A2): ")
                end_notation = input("Enter end position (e.g. A4): ")

                start_pos = Position.from_chess_notation(start_notation)
                end_pos = Position.from_chess_notation(end_notation)

                if not self.game.play_turn(start_pos, end_pos):
                    print("Invalid move. Try again.")
            except (ValueError, IndexError) as e:
                print(f"Error: {e}")
                print("Invalid input. Please use format 'A1', 'B2', etc.")


if __name__ == "__main__":
    controller = ChessController()
    controller.play()
