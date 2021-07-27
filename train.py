import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
from torch_points3d.trainer import LitTrainer
from torch_points3d.core.instantiator import HydraInstantiator, Instantiator

OmegaConf.register_new_resolver("get_filename", lambda x: x.split("/")[-1])


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.get("pretty_print"):
        print(OmegaConf.to_yaml(cfg, resolve=True))

    instantiator = HydraInstantiator()
    trainer = LitTrainer(instantiator, dataset=cfg.get("dataset"), trainer=cfg.get("trainer"), model=cfg.get("model"))
    trainer.train()


if __name__ == "__main__":
    main()
