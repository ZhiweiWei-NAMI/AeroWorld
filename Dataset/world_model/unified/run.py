"""AeroWorld entry point: train-predictor, collect-values, fit-selector, evaluate, predict.

collect-values / fit-selector: the measured-value selector (value.py collect, value_head.py fit).

Run from the repository root with the pinned runtimes/qwen35 Python. Training
commands also support torchrun. Each command exposes its own --help.
"""
import sys


def main():
    commands=('train-predictor','collect-values','fit-selector','evaluate','predict')
    usage='Usage: python -m Dataset.world_model.unified.run '+ '|'.join(commands)+' [options]'
    if len(sys.argv)>1 and sys.argv[1] in ('--help','-h'):
        print(usage)
        return
    if len(sys.argv)<2 or sys.argv[1] not in commands:
        raise SystemExit(usage)
    command=sys.argv.pop(1)
    if command=='train-predictor':
        from Dataset.world_model.unified.finetune_predictor import main as execute
    elif command=='collect-values':
        from Dataset.world_model.unified.value import main as execute
        sys.argv.insert(1,'collect')
    elif command=='fit-selector':
        from Dataset.world_model.unified.value_head import main as execute
        sys.argv.insert(1,'fit')
    elif command=='evaluate':
        from Dataset.world_model.unified.validate import main as execute
    else:
        from Dataset.world_model.unified.predict import main as execute
    execute()


if __name__=='__main__':main()
