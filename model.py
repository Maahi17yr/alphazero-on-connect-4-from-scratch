"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

ROWS = 6
COLS = 7


def make_empty_board() -> np.ndarray:
    """Return a fresh, empty Connect-4 board.

    The board has 6 rows and 7 columns. Cells hold integers:
        0 -> empty
        1 -> player 1
        2 -> player 2
    """
    return np.zeros((ROWS, COLS), dtype=int)

# Step 2 - column_top_row
import numpy as np


def column_top_row(board: np.ndarray, column: int) -> int:
    """Return the row index where a piece dropped in `column` would land.

    Row 0 is the top and row 5 is the bottom, so gravity pulls pieces toward
    higher row indices. The result is the largest row index in the column that
    still holds a 0, or -1 if the column is full.
    """
    empty_rows = np.flatnonzero(board[:, column] == 0)
    return int(empty_rows[-1]) if empty_rows.size else -1

# Step 3 - drop_piece
import numpy as np


def drop_piece(board: np.ndarray, column: int, player: int) -> np.ndarray:
    """Return a new board with `player`'s piece dropped into `column`.

    The input board is left unchanged. Raises ValueError if the column is full.
    """
    row = column_top_row(board, column)
    if row == -1:
        raise ValueError(f"Column {column} is full")

    new_board = board.copy()
    new_board[row, column] = player
    return new_board

# Step 4 - column_full
# ── Step 004  column_full ──
def column_full(board: np.ndarray, column: int) -> bool:
    """Return True if `column` cannot accept another piece.

    Only the top row decides playability: if board[0, column] is occupied,
    the column is full even if a lower cell is empty.
    """
    return bool(board[0, column] != 0)

# Step 5 - valid_moves
def valid_moves(board: np.ndarray) -> list[int]:
    """Return the column indices that can still accept a piece, left to right."""
    return [col for col in range(board.shape[1]) if not column_full(board, col)]

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board: np.ndarray) -> int:
    """Return the player (1 or 2) with four in a row horizontally, else 0.

    Scans each row left to right. If both players somehow have a line,
    the first one found (top row first, then leftmost) is returned.
    """
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols - 3):
            cell = board[r, c]
            if cell != 0 and (
                cell == board[r, c + 1] == board[r, c + 2] == board[r, c + 3]
            ):
                return int(cell)
    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board: np.ndarray) -> int:
    """Return the player (1 or 2) with four in a row vertically, else 0.

    Scans each column top to bottom. If several verticals exist, the first
    one found (leftmost column, then topmost window) is returned.
    """
    rows, cols = board.shape
    for c in range(cols):
        for r in range(rows - 3):
            cell = board[r, c]
            if cell != 0 and (
                cell == board[r + 1, c] == board[r + 2, c] == board[r + 3, c]
            ):
                return int(cell)
    return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board: np.ndarray) -> int:
    """Return the player (1 or 2) with four in a row on a down-right diagonal, else 0.

    A down-right diagonal runs from upper-left to lower-right, so each step
    increases both the row and the column by one. The first line found
    (topmost starting row, then leftmost starting column) is returned.
    """
    rows, cols = board.shape
    for r in range(rows - 3):
        for c in range(cols - 3):
            cell = board[r, c]
            if cell != 0 and (
                cell == board[r + 1, c + 1] == board[r + 2, c + 2] == board[r + 3, c + 3]
            ):
                return int(cell)
    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board: np.ndarray) -> int:
    """Return the player (1 or 2) with four in a row on an up-right diagonal, else 0.

    An up-right diagonal moves up one row (row index - 1) and right one
    column (column index + 1) per step. Each window starts at its bottom-left
    cell, so the start row must be at least 3. The first line found
    (topmost starting row, then leftmost starting column) is returned.
    """
    rows, cols = board.shape
    for r in range(3, rows):
        for c in range(cols - 3):
            cell = board[r, c]
            if cell != 0 and (
                cell == board[r - 1, c + 1] == board[r - 2, c + 2] == board[r - 3, c + 3]
            ):
                return int(cell)
    return 0

# Step 10 - check_winner
def check_winner(board: np.ndarray) -> int:
    """Return the player (1 or 2) with four in a row in any direction, else 0.

    Checks horizontal, vertical, down-right diagonal, and up-right diagonal
    in that order and returns the first non-zero result.
    """
    for check in (
        four_in_a_row_horizontal,
        four_in_a_row_vertical,
        four_in_a_row_diagonal_down_right,
        four_in_a_row_diagonal_up_right,
    ):
        winner = check(board)
        if winner:
            return winner
    return 0

# Step 11 - board_is_full
def board_is_full(board):
    return bool(len(valid_moves(board)) == 0)

# Step 12 - is_terminal
def is_terminal(board: np.ndarray) -> tuple[bool, int]:
    """Return (done, winner) for the given board.

    winner is 1 or 2 if that player has four in a row, else 0.
    done is True if someone has won or the board is full (a draw).
    """
    winner = check_winner(board)
    if winner:
        return True, int(winner)
    return board_is_full(board), 0

# Step 13 - other_player
def other_player(player: int) -> int:
    """Return the opponent's player code: 1 -> 2 and 2 -> 1."""
    return 3 - player

# Step 14 - step_env
def step_env(board: np.ndarray, column: int, player: int):
    """Perform one Connect-4 transition.

    Returns (new_board, done, winner, next_player):
      new_board    - a new board with `player`'s piece dropped in `column`
      done         - True if someone won or the board is now full
      winner       - 1 or 2 if that player has four in a row, else 0
      next_player  - the opponent of `player`
    Raises ValueError if the column is full.
    """
    new_board = drop_piece(board, column, player)
    done, winner = is_terminal(new_board)
    return new_board, done, winner, other_player(player)

# Step 15 - encode_board
def encode_board(board: np.ndarray, current_player: int) -> np.ndarray:
    """Encode the board from `current_player`'s point of view.

    Returns a float32 array of shape (2, 6, 7):
      channel 0 -> 1.0 where current_player has a piece
      channel 1 -> 1.0 where the opponent has a piece
    Empty cells are 0.0 in both channels.
    """
    opponent = other_player(current_player)
    return np.stack(
        [board == current_player, board == opponent]
    ).astype(np.float32)

# Step 16 - board_to_torch_tensor
import torch


def board_to_torch_tensor(board: np.ndarray, current_player: int) -> torch.Tensor:
    """Encode the board and return a float32 tensor of shape (1, 2, 6, 7)."""
    encoded = encode_board(board, current_player)        # (2, 6, 7) float32
    return torch.from_numpy(encoded).unsqueeze(0)        # add batch dimension

# Step 17 - init_conv_backbone (not yet solved)
# TODO: implement

# Step 18 - init_policy_head (not yet solved)
# TODO: implement

# Step 19 - init_value_head (not yet solved)
# TODO: implement

# Step 20 - build_policy_value_net (not yet solved)
# TODO: implement

# Step 21 - policy_value_forward (not yet solved)
# TODO: implement

# Step 22 - action_mask (not yet solved)
# TODO: implement

# Step 23 - masked_policy_logits (not yet solved)
# TODO: implement

# Step 24 - masked_log_softmax (not yet solved)
# TODO: implement

# Step 25 - sample_action_from_policy (not yet solved)
# TODO: implement

# Step 26 - greedy_action_from_policy (not yet solved)
# TODO: implement

# Step 27 - make_mcts_node (not yet solved)
# TODO: implement

# Step 28 - node_q_value (not yet solved)
# TODO: implement

# Step 29 - ucb_score (not yet solved)
# TODO: implement

# Step 30 - select_best_child (not yet solved)
# TODO: implement

# Step 31 - select_leaf (not yet solved)
# TODO: implement

# Step 32 - evaluate_with_network (not yet solved)
# TODO: implement

# Step 33 - expand_node (not yet solved)
# TODO: implement

# Step 34 - backup_value (not yet solved)
# TODO: implement

# Step 35 - run_one_simulation (not yet solved)
# TODO: implement

# Step 36 - run_mcts (not yet solved)
# TODO: implement

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

