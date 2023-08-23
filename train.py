import argparse

from mvits.trainer import VITSTrainer
from mvits.utils.utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="configs/config.json",
                    help='JSON file for configuration')
parser.add_argument('-m', '--model_dir', type=str, default="models", help='Model name')
parser.add_argument('-n', '--epochs', type=int, default=50, help='finetune epochs')
parser.add_argument('--drop_speaker_embed', type=str2bool, default=False,
                    help='whether to drop existing characters')
parser.add_argument('--resume_model', type=str, default=None,
                    help='whether to train with resume model')
parser.add_argument('--pretrained_model', type=str, default=None,
                    help='whether to train with pretrained model')
args = parser.parse_args()


trainer = VITSTrainer(config_path=args.config, args=args)
trainer.train(resume_model=args.resume_model, pretrained_model=args.pretrained_model)
