import pandas as pd
from ast import literal_eval
from dataclasses import dataclass
from sympy.combinatorics import Permutation
from typing import Dict, List
from tqdm import tqdm


class ParticipantVisibleError(Exception):
    pass


def main():
    puzzles = pd.read_csv("data/puzzles.csv")
    sample_submission = pd.read_csv("data/sample_submission.csv")
    score(
        solution=puzzles,
        submission=sample_submission,
        series_id_column_name="id",
        moves_column_name="moves",
        puzzle_info_path="data/puzzle_info.csv",
    )


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    series_id_column_name: str,
    moves_column_name: str,
    puzzle_info_path: str,
) -> float:
    """Santa 2023 evaluation metric.

    Parameters
    ----------
    solution : pd.DataFrame

    submission : pd.DataFrame

    series_id_column_name : str

    moves_column_name : str

    Returns
    -------
    total_num_moves : int
    """
    if list(submission.columns) != [series_id_column_name, moves_column_name]:
        raise ParticipantVisibleError(
            f"Submission must have columns "
            f"{series_id_column_name} and {moves_column_name}."
        )

    puzzle_info = pd.read_csv(puzzle_info_path, index_col="puzzle_type")
    total_num_moves = 0
    for sol, sub in tqdm(
        zip(solution.itertuples(), submission.itertuples()), total=len(solution)
    ):
        puzzle_id = getattr(sol, series_id_column_name)
        assert puzzle_id == getattr(sub, series_id_column_name)
        allowed_moves = literal_eval(puzzle_info.loc[sol.puzzle_type, "allowed_moves"])
        allowed_moves = {k: Permutation(v) for k, v in allowed_moves.items()}
        puzzle = Puzzle(
            puzzle_id=puzzle_id,
            allowed_moves=allowed_moves,
            solution_state=sol.solution_state.split(";"),
            initial_state=sol.initial_state.split(";"),
            num_wildcards=sol.num_wildcards,
        )

        # Score submission row
        total_num_moves += score_puzzle(
            puzzle_id, puzzle, getattr(sub, moves_column_name)
        )

    return total_num_moves


@dataclass
class Puzzle:
    """A permutation puzzle."""

    puzzle_id: str
    allowed_moves: Dict[str, List[int]]
    solution_state: List[str]
    initial_state: List[str]
    num_wildcards: int


def score_puzzle(puzzle_id, puzzle, sub_solution):
    """Score the solution to a permutation puzzle."""
    # Apply submitted sequence of moves to the initial state, from left to right
    moves = sub_solution.split(".")
    state = puzzle.initial_state
    for m in moves:
        power = 1
        if m[0] == "-":
            m = m[1:]
            power = -1
        try:
            p = puzzle.allowed_moves[m]
        except KeyError:
            raise ParticipantVisibleError(
                f"{m} is not an allowed move for {puzzle_id}."
            )
        state = (p**power)(state)

    # Check that submitted moves solve puzzle
    num_wrong_facelets = sum(not (s == t) for s, t in zip(puzzle.solution_state, state))
    if num_wrong_facelets > puzzle.num_wildcards:
        raise ParticipantVisibleError(f"Submitted moves do not solve {puzzle_id}.")

    # The score for this instance is the total number of moves needed
    # to solve the puzzle
    return len(moves)


if __name__ == "__main__":
    print(main())
