import argparse, yaml
from .utils.config import set_seed
from .engine.trainer import Trainer
from .data.dataset import make_dataloaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_palmgan.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg["seed"])
    train_ds, val_ds = make_dataloaders(cfg)
    T = Trainer(cfg)

    for epoch in range(cfg["optim"]["epochs"]):
        for batch in train_ds:
            logs = T.train_step(batch["x"], batch["y"])
        # save ckpt / samples / eval here

if __name__ == "__main__":
    main()
