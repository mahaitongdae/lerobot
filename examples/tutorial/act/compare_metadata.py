from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features


def main():
    dataset_id = "HuggingFaceVLA/libero"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    print(dataset_metadata)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    dataset2_id = "lerobot/svla_so101_pickplace"
    dataset2_metadata = LeRobotDatasetMetadata(dataset2_id)
    print(dataset2_metadata)
    features2 = dataset_to_policy_features(dataset2_metadata.features)

    print(features)
    print(features2)

if __name__ == "__main__":
    main()