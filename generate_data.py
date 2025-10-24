
"""
Data generation script for prompt injection attack datasets.

This script generates training datasets with various prompt injection attacks.
"""

import os
import logging
import json
import argparse
from data_utils import jload, generate_training_data    

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class FilteringDataset:
    def __init__(self, data_path: str, attack, output_dir=None, generate_training_data_func=None, random_position=False, cut_benign=False, train_or_test='train'):
        self.attack = attack
        self.output_dir = output_dir
        self.generate_training_data = generate_training_data_func
        self.random_position = random_position
        self.train_or_test = train_or_test
        self.cut_benign = cut_benign
        
        # Load data
        if data_path.endswith('.json'):
            list_data_dict = jload(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Generate sources with attacks
        if attack == 'None':
            logging.info("Generating clean data...")
            targets, targets_clean, injection, inject_task, positions, davinci_003_outputs, instructions = self.generate_training_data(list_data_dict, 'None', random_position=random_position, cut_benign=cut_benign, train_or_test=train_or_test)
            sources = targets_clean
            targets = targets_clean
            injection = ["" for _ in targets_clean]
            inject_task = ["" for _ in targets_clean]
        else:
            logging.info(f"Generating data with {attack} attack...")
            targets, sources, injection, inject_task, positions, davinci_003_outputs, instructions = self.generate_training_data(list_data_dict, attack, random_position=random_position, cut_benign=cut_benign, train_or_test=train_or_test)

        sources = [source.replace(" <|end_of_data|> ", "") for source in sources]

        self.sources = sources
        self.targets = targets
        self.injections = injection
        self.inject_tasks = inject_task
        self.positions = positions
        self.davinci_003_outputs = davinci_003_outputs
        self.instructions = instructions

        self.metadata = {
            'attack_type': attack,
            'data_path': data_path,
            'num_examples': len(sources),
            'random_position': random_position,
            'cut_benign': cut_benign
        }
        
        logging.info(f"Created dataset with {len(self.sources)} examples for {attack} attack")
        logging.info(f"length of sources: {len(self.sources)}")
        logging.info(f"length of targets: {len(self.targets)}")
        logging.info(f"length of injections: {len(self.injections)}")
        logging.info(f"length of inject_tasks: {len(self.inject_tasks)}")
        logging.info(f"length of positions: {len(self.positions)}")
        logging.info(f"length of davinci_003_outputs: {len(self.davinci_003_outputs)}")
        logging.info(f"length of instructions: {len(self.instructions)}")
    
    def save_data(self, output_path):
        data_to_save = {
            'metadata': self.metadata,
            'examples': [
                {
                    'source': source,
                    'target': target,
                    'injection': injection,
                    'inject_task': inject_task,
                    'attack_type': self.attack,
                    'position': position,
                    'davinci_003_output': davinci_003_output,
                    'instruction': instruction
                }
                for source, target, injection, inject_task, position, davinci_003_output, instruction in zip(self.sources, self.targets, self.injections, self.inject_tasks, self.positions, self.davinci_003_outputs, self.instructions)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved dataset to {output_path}")


class DatasetGenerator:
    """Generator class for creating and saving datasets."""

    def __init__(self, data_path, output_dir, generate_training_data_func, random_position=False, cut_benign=False, train_or_test='train'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.generate_training_data = generate_training_data_func
        self.random_position = random_position
        self.cut_benign = cut_benign
        self.train_or_test = train_or_test
        
        
    def generate_datasets(self, attack_types):
        """Generate and save datasets for different attack types."""
        dataset_info = []
        
        # Create dataset for each attack type
        for attack in attack_types:
            logging.info(f"Creating dataset for {attack} attack...")
            dataset = FilteringDataset(
                data_path=self.data_path,
                attack=attack,
                output_dir=self.output_dir,
                generate_training_data_func=self.generate_training_data,
                random_position=self.random_position,
                cut_benign=self.cut_benign,
                train_or_test=self.train_or_test
            )
            dataset_info.append(dataset.metadata)
    
            dataset_output_path = f"{self.output_dir}/dataset_{attack}.json"
            dataset.save_data(dataset_output_path)
            
        # Create a clean dataset
        logging.info("Creating clean dataset...")
        try:
            clean_dataset = FilteringDataset(
                data_path=self.data_path,
                attack='None',
                output_dir=self.output_dir,
                generate_training_data_func=self.generate_training_data,
                random_position=self.random_position,
                cut_benign=self.cut_benign,
                train_or_test=self.train_or_test
            )
            dataset_info.append(clean_dataset.metadata)
            
            # Save clean dataset
            clean_output_path = f"{self.output_dir}/dataset_clean.json"
            clean_dataset.save_data(clean_output_path)
            
        except Exception as e:
            logging.error(f"Error creating clean dataset: {e}")
        
        # Save metadata
        metadata_path = f"{self.output_dir}/combined_dataset_metadata.json"
        combined_metadata = {
            'individual_datasets': dataset_info,
            'attack_types': attack_types,
            'total_datasets': len(dataset_info),
            'random_position': self.random_position,
            'cut_benign': self.cut_benign
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        logging.info(f"Saved combined metadata to {metadata_path}")        
        logging.info("Dataset generation completed!")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate filtering datasets for training or testing')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', 
                           help='Generate training datasets using alpaca_data_cleaned.json')
    mode_group.add_argument('--test', action='store_true',
                           help='Generate test datasets using davinci_003_outputs.json')
    parser.add_argument('--position', action='store_true',
                       help='Use random position (random_position=True), otherwise random_position=False')
    parser.add_argument('--cut_benign', action='store_true',
                       help='Cut benign examples from training data')
    parser.add_argument('--attack_types', nargs='+', 
                        default=['Naive', 'Ignore', 'Completion'],
                        choices= ['None', 'Naive', 'Ignore', 'Completion', 'CompletionIgnore', 'StrongDelimiterCompletion', 'MultishotCompletion', 'Context'],
                       help='Attack types to generate datasets for (default: Naive Ignore Completion)')
    return parser.parse_args()


def main():
    args = parse_args()
    train_or_test = 'train' if args.train else 'test'

    if args.train:
        data_path = "data/alpaca_data_cleaned.json" 
        if args.position:
            output_dir = "./train_data_randomposition"
        else:
            output_dir = "./train_data"
        if args.cut_benign:
            cut_benign = True
            output_dir = output_dir + "_cutdata"
        else:
            cut_benign = False
    else:  # args.test
        data_path = "data/davinci_003_outputs.json"
        if args.position:
            output_dir = "./test_data_randomposition"
        else:
            output_dir = "./test_data"
        cut_benign = False
        
    logging.info(f"Data path: {data_path}")
    logging.info(f"Output directory: {output_dir}")

    if not os.path.exists("data"):
        os.makedirs("data")
    # if data_file does not exist, download it
    if not os.path.exists(data_path):
        import pandas as pd
        if train_or_test == 'train':
            logging.info("Downloading alpaca_data_cleaned.json from Hugging Face...")
            url = "https://huggingface.co/datasets/yahma/alpaca-cleaned/blob/main/alpaca_data_cleaned.json"
            df = pd.read_json(url)
            df.to_json(data_path, indent=2, lines=False)
        else:
            logging.info("Downloading davinci_003_outputs.json from Hugging Face...")
            url = "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/raw/main/davinci_003_outputs.json"
            df = pd.read_json(url)
            df.to_json(data_path, indent=2, lines=False)

    os.makedirs(output_dir, exist_ok=True)
    
    generator = DatasetGenerator(
        data_path=data_path,
        output_dir=output_dir,
        generate_training_data_func=generate_training_data,
        random_position=args.position,
        cut_benign=cut_benign,
        train_or_test=train_or_test
    )
    
    generator.generate_datasets(
        attack_types=args.attack_types
    )


if __name__ == "__main__":
    main()