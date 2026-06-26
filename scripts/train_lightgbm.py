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
        "--inventory",
        nargs="+",
        default=["app/data/training/*.csv", "app/data/inventory_flow_5days.csv"],
        help="Inventory flow CSV path(s) or glob(s).",
    )
    parser.add_argument("--item-master", default=str(ITEM_MASTER_PATH))
    parser.add_argument("--order-policy", default=str(ORDER_POLICY_PATH))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_PATH))
    args = parser.parse_args()

    inventory_paths = inventory_paths_from_glob(args.inventory)
    if not inventory_paths:
        raise SystemExit("No inventory CSV files found.")

    result = train_lightgbm_model(
        inventory_paths=inventory_paths,
        item_master_path=Path(args.item_master),
        order_policy_path=Path(args.order_policy),
        model_path=Path(args.model_out),
        metadata_path=Path(args.metadata_out),
    )
    print(
        "trained demand model: "
        f"{result['training_examples']} examples, "
        f"{result['features']} features, "
        f"MAE={result['evaluation']['mae']}, "
        f"RMSE={result['evaluation']['rmse']}, "
        f"MAPE={result['evaluation']['mape']}%, "
        f"model={result['model_path']}, "
        f"metadata={result['metadata_path']}"
    )


if __name__ == "__main__":
    main()
