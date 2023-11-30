import configparser
import argparse
import train
import split
import numpy as np


if __name__ == "__main__":

    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Specify config file")
    parser.add_argument("--operation", "-o", help="Specify whether to train or predict")
    args = parser.parse_args()

    # Check to see if config file has been specified
    if not args.config:
        print("==> Error: Please specify a valid config file")
        exit()

    # Parse config file
    try:
        config = configparser.ConfigParser()
        config.read(args.config)
    except:
        print("==> Error reading config file")
        exit()

    if args.operation == "a" or args.operation == "augment":
        print("Here")
    elif args.operation == "s" or args.operation == "split":
        split.splitData(config['file_paths']['dataset'])
    elif args.operation == "p" or args.operation == "predict":
        train = train.Trainer(model_path=config['file_paths']['model'], labels=np.asarray([float(x) for x in config['annotation']['labels'].split(",")], dtype=np.float32))
        train.predict(config['file_paths']['predict_in'], config['file_paths']['predict_out'], config['annotation']['output_annotation'])
    else:
        train = train.Trainer(int(config['network_params']['in_channels']), int(config['network_params']['out_channels']), \
                                        np.asarray([float(x) for x in config['annotation']['labels'].split(",")], dtype=np.float32), model_dir=config['file_paths']['model_dir'], \
                                         model_path=None, network=config['network_params']['network'],augmentation=int(config['augmentation']['apply_aug']))
        train.train(config['file_paths']['dataset'], int(config['network_params']['batch_size']), int(config['network_params']['epochs']))
