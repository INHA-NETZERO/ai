#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.demand_model import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    inventory_paths_from_glob,
    train_lightgbm_model,
)
from app.services.demo_data import ITEM_MASTER_PATH, ORDER_POLICY_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the POS demand LightGBM model.")
    parser.add_argument(
        "--training",
        "--inventory",
        nargs="+",
        default=["app/data/training/sales_*.csv"],
        help="Sales training CSV path(s) or glob(s). Required columns: 날짜,요일,날씨,기온,강수mm,행사,공휴일,신메뉴,품목,구분,판매수량,비고_시나리오.",
    )
    parser.add_argument("--item-master", default=str(ITEM_MASTER_PATH))
    parser.add_argument("--order-policy", default=str(ORDER_POLICY_PATH))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_PATH))
    args = parser.parse_args()

    training_paths = inventory_paths_from_glob(args.training)
    if not training_paths:
        raise SystemExit("No sales training CSV files found. Put files under app/data/training/ or pass --training.")

    result = train_lightgbm_model(
        training_paths=training_paths,
        item_master_path=Path(args.item_master),
        order_policy_path=Path(args.order_policy),
        model_path=Path(args.model_out),
        metadata_path=Path(args.metadata_out),
    )
    evaluation = result["evaluation"]
    print(
        "trained demand model: "
        f"{result['training_examples']} examples, "
        f"{result['features']} features, "
        f"model={result['model_path']}, "
        f"metadata={result['metadata_path']}"
    )
    print(
        "evaluation: "
        f"train(MAE={evaluation['train']['mae']}, RMSE={evaluation['train']['rmse']}, "
        f"MAPE={evaluation['train']['mape']}%), "
        f"validation(MAE={evaluation['validation']['mae']}, RMSE={evaluation['validation']['rmse']}, "
        f"MAPE={evaluation['validation']['mape']}%), "
        f"test(MAE={evaluation['test']['mae']}, RMSE={evaluation['test']['rmse']}, "
        f"MAPE={evaluation['test']['mape']}%), "
        f"overfit_gap(MAE={evaluation['overfit_gap']['mae']}, "
        f"RMSE={evaluation['overfit_gap']['rmse']}, MAPE={evaluation['overfit_gap']['mape']}%)"
    )


if __name__ == "__main__":
    main()
