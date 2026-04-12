"""Verify that task indices in the dataset and the LIBERO env suites refer to the same tasks.

The HuggingFaceVLA/libero dataset contains tasks from multiple LIBERO suites.
This script finds which suite and env task_id each dataset task belongs to.

ds_idx  suite             env_id  match  dataset task
------------------------------------------------------------------------------------------------------------------------
     0  libero_10              4     OK  put the white mug on the left plate and put the yellow and white mug on the right plate
     1  libero_10              6     OK  put the white mug on the plate and put the chocolate pudding to the right of the plate
     2  libero_10              9     OK  put the yellow and white mug in the microwave and close it
     3  libero_10              2     OK  turn on the stove and put the moka pot on it
     4  libero_10              7     OK  put both the alphabet soup and the cream cheese box in the basket
     5  libero_10              0     OK  put both the alphabet soup and the tomato sauce in the basket
     6  libero_10              8     OK  put both moka pots on the stove
     7  libero_10              1     OK  put both the cream cheese box and the butter in the basket
     8  libero_10              3     OK  put the black bowl in the bottom drawer of the cabinet and close it
     9  libero_90             77     OK  pick up the book and place it in the back compartment of the caddy
    10  libero_goal            8     OK  put the bowl on the plate
    11  libero_goal            9     OK  put the wine bottle on the rack
    12  libero_goal            3     OK  open the top drawer and put the bowl inside
    13  libero_goal            6     OK  put the cream cheese in the bowl
    14  libero_goal            2     OK  put the wine bottle on top of the cabinet
    15  libero_goal            5     OK  push the plate to the front of the stove
    16  libero_90             44     OK  turn on the stove
    17  libero_goal            1     OK  put the bowl on the stove
    18  libero_goal            4     OK  put the bowl on top of the cabinet
    19  libero_goal            0     OK  open the middle drawer of the cabinet
    20  libero_object          9     OK  pick up the orange juice and place it in the basket
    21  libero_object          4     OK  pick up the ketchup and place it in the basket
    22  libero_object          1     OK  pick up the cream cheese and place it in the basket
    23  libero_object          3     OK  pick up the bbq sauce and place it in the basket
    24  libero_object          0     OK  pick up the alphabet soup and place it in the basket
    25  libero_object          7     OK  pick up the milk and place it in the basket
    26  libero_object          2     OK  pick up the salad dressing and place it in the basket
    27  libero_object          6     OK  pick up the butter and place it in the basket
    28  libero_object          5     OK  pick up the tomato sauce and place it in the basket
    29  libero_object          8     OK  pick up the chocolate pudding and place it in the basket
    30  libero_spatial         6     OK  pick up the black bowl next to the cookie box and place it on the plate
    31  libero_spatial         4     OK  pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
    32  libero_spatial         5     OK  pick up the black bowl on the ramekin and place it on the plate
    33  libero_spatial         7     OK  pick up the black bowl on the stove and place it on the plate
    34  libero_spatial         0     OK  pick up the black bowl between the plate and the ramekin and place it on the plate
    35  libero_spatial         3     OK  pick up the black bowl on the cookie box and place it on the plate
    36  libero_spatial         8     OK  pick up the black bowl next to the plate and place it on the plate
    37  libero_spatial         1     OK  pick up the black bowl next to the ramekin and place it on the plate
    38  libero_spatial         2     OK  pick up the black bowl from table center and place it on the plate
    39  libero_spatial         9     OK  pick up the black bowl on the wooden cabinet and place it on the plate

"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero.libero import benchmark

SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="HuggingFaceVLA/libero")
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(args.dataset_id)
    ds_tasks = {int(row["task_index"]): name for name, row in meta.tasks.iterrows()}

    # Build a lookup: normalized task string -> (suite_name, env_task_id)
    env_lookup = {}
    for suite_name in SUITES:
        try:
            suite = benchmark.get_benchmark_dict()[suite_name]()
        except KeyError:
            continue
        for i in range(len(suite.tasks)):
            lang = suite.get_task(i).language.strip().lower()
            env_lookup[lang] = (suite_name, i)

    print(f"{'ds_idx':>6}  {'suite':<16}  {'env_id':>6}  {'match':>5}  {'dataset task'}")
    print("-" * 120)

    all_match = True
    for ds_idx in sorted(ds_tasks.keys()):
        ds_name = ds_tasks[ds_idx]
        key = ds_name.strip().lower()
        if key in env_lookup:
            suite_name, env_id = env_lookup[key]
            print(f"{ds_idx:>6}  {suite_name:<16}  {env_id:>6}  {'OK':>5}  {ds_name}")
        else:
            all_match = False
            print(f"{ds_idx:>6}  {'???':<16}  {'???':>6}  {'MISS':>5}  {ds_name}")

    print()
    if all_match:
        print("All dataset tasks found in LIBERO env suites.")
    else:
        print("WARNING: Some dataset tasks were NOT found in any LIBERO suite!")


if __name__ == "__main__":
    main()
