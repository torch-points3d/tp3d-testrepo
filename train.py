import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import LitTrainer

OmegaConf.register_new_resolver("get_filename", lambda x: x.split("/")[-1])


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(
        cfg, False
    )  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    trainer = LitTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
