from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from recbole3.config import instantiate_dataclass
from recbole3.dataset import get_dataset_spec
from recbole3.dataset.ml100k import ML100KRetrievalConfig, ML100KRetrievalDataset
from recbole3.model.llamarec.candidates import import_llamarec_candidates
from recbole3.model.llamarec.config import LlamaRecConfig
from recbole3.model.llamarec.data import load_or_build_llamarec_artifact
from recbole3.model.llamarec.trainer import LlamaRecTrainer, LlamaRecTrainerConfig
from recbole3.model.lrurec import LRURecConfig, LRURecModel, LRURecModelDataset, LRURecTrainer, LRURecTrainerConfig
from recbole3.pipeline import Pipeline
from recbole3.utils import require_component_name


class LlamaRecPipeline(Pipeline):
    def run(self) -> dict[str, Any]:
        runtime_cfg, dataset_cfg, model_cfg, trainer_cfg = self._parse_config(self.cfg)
        dataset_spec = get_dataset_spec(require_component_name(dataset_cfg, "dataset"))
        dataset_config = instantiate_dataclass(dataset_spec.config_cls, dataset_cfg)
        model_config = instantiate_dataclass(LlamaRecConfig, model_cfg)
        trainer_config = instantiate_dataclass(LlamaRecTrainerConfig, trainer_cfg)
        task_data = dataset_spec.dataset_cls(dataset_config).prepare(eval_config=trainer_config.eval)
        mapping_source = self._resolve_mapping_source(
            dataset_config,
            task_data,
            model_config=model_config,
        )
        compatible_task_data = self._prepare_compatible_task_data(
            dataset_config,
            mapping_source=mapping_source,
            eval_config=trainer_config.eval,
        )
        retrieved_path = self._resolve_retrieved_path(
            compatible_task_data,
            model_config=model_config,
            trainer_config=trainer_config,
            runtime_output_dir=runtime_cfg.output_dir,
        )
        valid_frame, test_frame = import_llamarec_candidates(
            compatible_task_data,
            retrieved_path=retrieved_path,
            mapping_source=mapping_source,
            topk=model_config.candidate_topk,
        )
        trainer = LlamaRecTrainer(model_config, trainer_config)
        with self._accelerate_runtime_device(runtime_cfg.device):
            result = trainer.run(
                mapping_source=mapping_source,
                valid_frame=valid_frame,
                test_frame=test_frame,
                output_dir=runtime_cfg.output_dir,
            )
        print(OmegaConf.to_yaml(OmegaConf.create(result), resolve=True))
        return result

    def _resolve_mapping_source(
        self,
        dataset_config: Any,
        task_data: Any,
        *,
        model_config: LlamaRecConfig,
    ) -> str:
        if isinstance(dataset_config, ML100KRetrievalConfig):
            return str(dataset_config.mapping_source)
        _, artifact_path = load_or_build_llamarec_artifact(
            task_data,
            item_text_field=model_config.item_text_field,
            refresh_cache=bool(getattr(dataset_config, "refresh_cache", False)),
        )
        return str(artifact_path)

    def _prepare_compatible_task_data(
        self,
        dataset_config: Any,
        *,
        mapping_source: str,
        eval_config: Any,
    ) -> Any:
        if isinstance(dataset_config, ML100KRetrievalConfig):
            return ML100KRetrievalDataset(dataset_config).prepare(eval_config=eval_config)
        compat_config = ML100KRetrievalConfig(
            mapping_source=str(mapping_source),
            processed_dir=str(Path(mapping_source).parent / "processed"),
            refresh_cache=bool(getattr(dataset_config, "refresh_cache", False)),
            split=dataset_config.split,
        )
        return ML100KRetrievalDataset(compat_config).prepare(eval_config=eval_config)

    def _resolve_retrieved_path(
        self,
        compatible_task_data: Any,
        *,
        model_config: LlamaRecConfig,
        trainer_config: LlamaRecTrainerConfig,
        runtime_output_dir: str,
    ) -> str:
        configured_path = Path(model_config.retrieved_path)
        if configured_path.is_file():
            return str(configured_path)
        if not bool(model_config.auto_stage1):
            return str(configured_path)

        stage1_output = Path(model_config.stage1_output_dir or Path(runtime_output_dir) / "lrurec_stage1")
        stage1_output.mkdir(parents=True, exist_ok=True)
        candidate_path = stage1_output / "retrieved.pkl"
        if candidate_path.exists():
            return str(candidate_path)

        lrurec_config = LRURecConfig(history_max_length=int(model_config.stage1_history_max_length))
        lrurec_prepared = LRURecModelDataset.from_task_dataset(compatible_task_data, model_config=lrurec_config)
        lrurec_trainer_config = LRURecTrainerConfig(
            batch_size=64,
            candidate_topk=int(trainer_config.candidate_topk),
            save_full_score_matrix=bool(trainer_config.save_full_score_matrix or model_config.stage1_save_full_score_matrix),
        )
        if model_config.stage1_max_epochs is not None:
            lrurec_trainer_config.max_epochs = int(model_config.stage1_max_epochs)
        lrurec_trainer = LRURecTrainer(lrurec_trainer_config)
        lrurec_model = LRURecModel(lrurec_config)
        lrurec_trainer.run(lrurec_model, lrurec_prepared, output_dir=stage1_output)
        return str(candidate_path)


__all__ = ["LlamaRecPipeline"]
