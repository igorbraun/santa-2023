import pandas as pd
import json
import collections
from santa_2023.config import DATA_PATH


def read_puzzle_info():
    return pd.read_csv(DATA_PATH.joinpath("puzzle_info.csv"))


def read_puzzles():
    return pd.read_csv(DATA_PATH.joinpath("puzzles.csv"))


def read_sample_submission():
    return pd.read_csv(DATA_PATH.joinpath("sample_submission.csv"))


def _make_dict(dict_string: str) -> dict:
    json_acceptable_string = dict_string.replace("'", '"')
    return json.loads(json_acceptable_string)


def make_puzzle_info_dict(
    puzzle_info_df: pd.DataFrame,
) -> collections.defaultdict[dict]:
    puzzle_info_dict = collections.defaultdict(dict)
    for row in puzzle_info_df.iterrows():
        puzzle_type = row[1].puzzle_type
        allowed_moves = _make_dict(row[1].allowed_moves)
        puzzle_info_dict[puzzle_type].update(allowed_moves)
    return puzzle_info_dict


def explode_puzzle_info(puzzle_info_df: pd.DataFrame) -> pd.DataFrame:
    result = []

    for row in puzzle_info_df.iterrows():
        moves_dict = _make_dict(row[1].allowed_moves)
        result.append(
            pd.DataFrame(
                {
                    "puzzle_type": row[1].puzzle_type,
                    "move": moves_dict.keys(),
                    "permutation": moves_dict.values(),
                }
            )
        )

    return pd.concat(result)
