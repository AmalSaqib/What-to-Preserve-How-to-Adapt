"""Sequential nnU-Net trainer with explicit block-level adaptation policies."""

import copy
import csv
import json
import os
import random

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet_ext.paths import default_plans_identifier
from nnunet_ext.training.network_training.sequential.nnUNetTrainerSequential import nnUNetTrainerSequential
from torch.cuda.amp import GradScaler, autocast

# -- Define globally the Hyperparameters for this trainer along with their type -- #
HYPERPARAMS = {
    "full_body_freeze": bool,
    "encoder_unfreeze": list,
    "decoder_unfreeze": list,
    "reinit_optimizer_after_validation": bool,
    "gradient_lr_control": bool,
    "gradient_calibration_batches": int,
    "gradient_lr_reference": str,
    "gradient_lr_min_multiplier": float,
    "gradient_lr_max_multiplier": float,
    "gradient_displacement_log_interval": int,
    "base_learning_rate": float,
    "run_seed": int,
}


class nnUNetTrainerSequentialChannelPreserve(nnUNetTrainerSequential):
    """
    Sequential trainer:
      - Task 1: train normally (no freezing)
      - Task >= 2: freeze entire shared body by parameter identity
        - If full_body_freeze=True: only heads train
        - Else: unfreeze specified encoder/decoder blocks (and td/tu) while keeping rest frozen
      - Heads always trainable

    Adds compact sanity logging to catch:
      - name mismatch issues (avoided by param identity freezing)
      - accidental reassembly changing Parameter identities
      - requires_grad distribution in key modules
    """

    def __init__(
        self,
        split,
        task,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        save_interval=5,
        already_trained_on=None,
        use_progress=True,
        identifier=default_plans_identifier,
        extension="sequential_channel_preserve",
        tasks_list_with_char=None,
        mixed_precision=True,
        save_csv=True,
        del_log=False,
        use_vit=False,
        vit_type="base",
        version=1,
        split_gpu=False,
        transfer_heads=True,
        ViT_task_specific_ln=False,
        do_LSA=False,
        do_SPT=False,
        network=None,
        use_param_split=False,
        # new args
        full_body_freeze=False,
        encoder_unfreeze=None,
        decoder_unfreeze=None,
        reinit_optimizer_after_validation=True,
        gradient_lr_control=False,
        gradient_calibration_batches=20,
        gradient_lr_reference="B",
        gradient_lr_min_multiplier=0.1,
        gradient_lr_max_multiplier=10.0,
        gradient_displacement_log_interval=5,
        base_learning_rate=None,
        run_seed=12345,
    ):
        super().__init__(
            split,
            task,
            plans_file,
            fold,
            output_folder,
            dataset_directory,
            batch_dice,
            stage,
            unpack_data,
            deterministic,
            fp16,
            save_interval,
            already_trained_on,
            use_progress,
            identifier,
            extension,
            tasks_list_with_char,
            mixed_precision,
            save_csv,
            del_log,
            use_vit,
            vit_type,
            version,
            split_gpu,
            True,  # enforce transfer_heads=True
            ViT_task_specific_ln,
            do_LSA,
            do_SPT,
            network,
            use_param_split,
        )

        self.full_body_freeze = bool(full_body_freeze)
        self.encoder_unfreeze = sorted(set(int(x) for x in (encoder_unfreeze or [])))
        self.decoder_unfreeze = sorted(set(int(x) for x in (decoder_unfreeze or [])))
        self.reinit_optimizer_after_validation = bool(reinit_optimizer_after_validation)
        self.gradient_lr_control = bool(gradient_lr_control)
        self.gradient_calibration_batches = int(gradient_calibration_batches)
        self.gradient_lr_reference = str(gradient_lr_reference)
        self.gradient_lr_min_multiplier = float(gradient_lr_min_multiplier)
        self.gradient_lr_max_multiplier = float(gradient_lr_max_multiplier)
        self.gradient_displacement_log_interval = int(gradient_displacement_log_interval)
        self.base_learning_rate = (
            self.initial_lr if base_learning_rate is None else float(base_learning_rate)
        )
        if self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be positive")
        self.initial_lr = self.base_learning_rate
        self.run_seed = int(run_seed)
        if self.run_seed < 0:
            raise ValueError("run_seed must be non-negative")

        # NetworkTrainer seeds with 12345 during super().__init__. Reseed here,
        # before model/head creation and data-loader initialization, so repeated
        # adaptation runs differ while retaining the same fold split.
        random.seed(self.run_seed)
        np.random.seed(self.run_seed)
        torch.manual_seed(self.run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.run_seed)
        self.already_trained_on[str(self.fold)]["used_run_seed"] = self.run_seed

        if self.gradient_calibration_batches < 1:
            raise ValueError("gradient_calibration_batches must be at least 1")
        if not (0 < self.gradient_lr_min_multiplier <= self.gradient_lr_max_multiplier):
            raise ValueError("gradient LR multiplier bounds must satisfy 0 < min <= max")
        if self.gradient_displacement_log_interval < 1:
            raise ValueError("gradient_displacement_log_interval must be at least 1")

        self._gradient_lr_multipliers = {}
        self._gradient_control_calibrated = False
        self._gradient_control_initial_weights = {}
        self._gradient_control_resumed = False
        self._gradient_resume_optimizer_state = None

        # logging/diagnostics state
        self._logged_param_id_sanity = False
        self._param_id_anchor = None  # (name, id(param), shape)

    # ---------------------------
    # Utility / Sanity Helpers
    # ---------------------------

    def _is_second_task_or_later(self):
        finished = self.already_trained_on[str(self.fold)].get("finished_training_on", [])
        return len(finished) > 0

    def _get_body_param_set(self):
        # Freeze by PARAMETER IDENTITY (robust across naming differences)
        return {p for p in self.mh_network.body.parameters()}

    def _ensure_head_trainable(self):
        """Keep all head parameters trainable (by param identity, not name)."""
        body_params = self._get_body_param_set()
        for _, p in self.network.named_parameters():
            if p not in body_params:
                p.requires_grad = True

    def _freeze_entire_body(self):
        """Freeze all shared-body parameters (by identity)."""
        body_params = self._get_body_param_set()
        for _, p in self.network.named_parameters():
            if p in body_params:
                p.requires_grad = False

    @staticmethod
    def _set_module_requires_grad(module: torch.nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def _map_decoder_cli_idx(idx: int, n_dec: int):
        """Map CLI decoder index so D0 is closest to output and the max index is closest to bottleneck."""
        if not (0 <= idx < n_dec):
            return None
        return (n_dec - 1) - idx

    def _maybe_anchor_param_id(self):
        """Anchor one shared-body parameter id for 'assemble identity' sanity checks."""
        if self._param_id_anchor is not None:
            return
        # pick first parameter from body if exists
        for name, p in self.mh_network.body.named_parameters():
            self._param_id_anchor = (f"BODY::{name}", id(p), tuple(p.shape))
            break

    def _log_param_identity_sanity(self, where: str):
        """
        Checks if the anchored shared-body parameter appears in the assembled network by id.
        If not, it strongly suggests assemble_model creates new Parameter objects.
        """
        if self._logged_param_id_sanity:
            return

        self._maybe_anchor_param_id()
        if self._param_id_anchor is None:
            self.print_to_log_file(f"[SANITY] ({where}) ParamID check skipped: no body params found")
            self._logged_param_id_sanity = True
            return

        anchor_name, anchor_id, anchor_shape = self._param_id_anchor
        net_param_ids = {id(p) for p in self.network.parameters()}
        present = anchor_id in net_param_ids

        self.print_to_log_file(
            f"[SANITY] ({where}) ParamID anchor: {anchor_name} id={anchor_id} shape={anchor_shape} present_in_network={present}"
        )
        if not present:
            self.print_to_log_file(
                "[WARNING] assemble_model likely created NEW Parameter objects (anchor id not present). "
                "If you re-assemble after validation, you MUST re-init optimizer to avoid stale param refs."
            )

        self._logged_param_id_sanity = True

    def _summarize_requires_grad(self):
        total = 0
        trainable = 0
        for p in self.network.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        pct = 100.0 * trainable / total if total > 0 else 0.0
        return trainable, total, pct

    def _count_trainable_params_in_module(self, module: torch.nn.Module):
        tot = 0
        trn = 0
        for p in module.parameters():
            tot += p.numel()
            if p.requires_grad:
                trn += p.numel()
        return trn, tot

    def _controlled_block_param_groups(self):
        """Return trainable parameters grouped by paper-facing U-Net block names."""
        body_params = self._get_body_param_set()
        groups = {}
        assigned = set()

        def add_module(label, module):
            if module is None:
                return
            for param in module.parameters():
                if param.requires_grad and param in body_params and id(param) not in assigned:
                    groups.setdefault(label, []).append(param)
                    assigned.add(id(param))

        if hasattr(self.network, "conv_blocks_context"):
            n_enc = len(self.network.conv_blocks_context)
            for idx, module in enumerate(self.network.conv_blocks_context):
                label = "B" if idx == n_enc - 1 else f"E{idx}"
                add_module(label, module)
                if hasattr(self.network, "td") and idx < len(self.network.td):
                    add_module(label, self.network.td[idx])

        if hasattr(self.network, "conv_blocks_localization"):
            n_dec = len(self.network.conv_blocks_localization)
            for internal_idx, module in enumerate(self.network.conv_blocks_localization):
                # Internal decoder index 0 is bottleneck-adjacent (paper block D5).
                label = f"D{n_dec - 1 - internal_idx}"
                add_module(label, module)
                if hasattr(self.network, "tu") and internal_idx < len(self.network.tu):
                    add_module(label, self.network.tu[internal_idx])

        unassigned_body = [p for p in body_params if p.requires_grad and id(p) not in assigned]
        if unassigned_body:
            groups["body_other"] = unassigned_body
            assigned.update(id(p) for p in unassigned_body)

        head = [
            p for p in self.network.parameters()
            if p.requires_grad and p not in body_params and id(p) not in assigned
        ]
        if head:
            groups["head"] = head
        return groups

    def initialize_optimizer_and_scheduler(self):
        """Create SGD parameter groups after gradient-scale calibration."""
        if not (self.gradient_lr_control and self._gradient_lr_multipliers):
            return super().initialize_optimizer_and_scheduler()

        if self.all_tr_losses:
            current_base_lr = poly_lr(self.epoch + 1, self.max_num_epochs, self.initial_lr, 0.9)
        else:
            current_base_lr = self.initial_lr

        param_groups = []
        for label, params in self._controlled_block_param_groups().items():
            multiplier = self._gradient_lr_multipliers.get(label, 1.0)
            param_groups.append({
                "params": params,
                "lr": current_base_lr * multiplier,
                "lr_multiplier": multiplier,
                "block_name": label,
            })

        self.optimizer = torch.optim.SGD(
            param_groups,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        self.lr_scheduler = None

    def maybe_update_lr(self, epoch=None):
        """Apply nnU-Net's polynomial schedule without erasing block LR ratios."""
        ep = self.epoch + 1 if epoch is None else epoch
        base_lr = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * group.get("lr_multiplier", 1.0)
        lr_summary = {
            group.get("block_name", "all"): round(group["lr"], 8)
            for group in self.optimizer.param_groups
        }
        self.print_to_log_file(f"controlled lr: {lr_summary}")

    @staticmethod
    def _relative_gradient_norm(params):
        grad_sq = 0.0
        param_sq = 0.0
        for param in params:
            param_sq += float(torch.sum(param.detach().float() ** 2).item())
            if param.grad is not None:
                grad_sq += float(torch.sum(param.grad.detach().float() ** 2).item())
        return (grad_sq ** 0.5) / ((param_sq ** 0.5) + 1e-12)

    def _calibrate_gradient_lr_control(self):
        """Estimate static block LRs that equalize the initial relative SGD step."""
        if self._gradient_control_calibrated or not self._is_second_task_or_later():
            return

        groups = self._controlled_block_param_groups()
        body_groups = {k: v for k, v in groups.items() if k not in {"head", "body_other"}}
        if self.gradient_lr_reference not in body_groups:
            raise RuntimeError(
                f"Gradient LR reference {self.gradient_lr_reference!r} is not trainable; "
                f"available body blocks are {sorted(body_groups)}"
            )

        observations = {label: [] for label in body_groups}
        was_training = self.network.training
        self.network.train()
        calibration_scaler = GradScaler() if self.fp16 else None

        self.print_to_log_file(
            f"[GRAD-LR] Calibrating on {self.gradient_calibration_batches} batches; "
            f"reference={self.gradient_lr_reference}"
        )
        for _ in range(self.gradient_calibration_batches):
            data_dict = next(self.tr_gen)
            data = maybe_to_torch(data_dict["data"])
            target = maybe_to_torch(data_dict["target"])
            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            self.optimizer.zero_grad()
            if self.fp16:
                with autocast():
                    output = self.network(data)
                    loss = self.loss(output, target)
                calibration_scaler.scale(loss).backward()
                calibration_scaler.unscale_(self.optimizer)
            else:
                output = self.network(data)
                loss = self.loss(output, target)
                loss.backward()

            for label, params in body_groups.items():
                observations[label].append(self._relative_gradient_norm(params))

            if calibration_scaler is not None:
                calibration_scaler.update()
            self.optimizer.zero_grad()
            del data, target, output, loss

        self.network.train(was_training)
        medians = {label: float(np.median(values)) for label, values in observations.items()}
        reference_value = medians[self.gradient_lr_reference]
        if reference_value <= 0:
            raise RuntimeError("Reference block has a zero median relative gradient norm")

        multipliers = {}
        clipped = {}
        for label, value in medians.items():
            raw = self.gradient_lr_max_multiplier if value <= 0 else reference_value / value
            bounded = min(self.gradient_lr_max_multiplier, max(self.gradient_lr_min_multiplier, raw))
            multipliers[label] = float(bounded)
            clipped[label] = not np.isclose(raw, bounded)
        multipliers["head"] = 1.0
        if "body_other" in groups:
            multipliers["body_other"] = 1.0

        self._gradient_lr_multipliers = multipliers
        self._gradient_control_calibrated = True
        report = {
            "calibration_batches": self.gradient_calibration_batches,
            "reference_block": self.gradient_lr_reference,
            "base_learning_rate": self.initial_lr,
            "median_relative_gradient_norm": medians,
            "lr_multiplier": multipliers,
            "learning_rate": {k: self.initial_lr * v for k, v in multipliers.items()},
            "multiplier_was_clipped": clipped,
            "multiplier_bounds": [self.gradient_lr_min_multiplier, self.gradient_lr_max_multiplier],
        }
        report_path = join(self.output_folder, "gradient_lr_calibration.json")
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        initial_report_path = join(self.output_folder, "gradient_lr_calibration_initial.json")
        if not os.path.exists(initial_report_path):
            with open(initial_report_path, "w") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
        self.print_to_log_file(f"[GRAD-LR] Calibration report: {report_path}")
        self.print_to_log_file(f"[GRAD-LR] Median relative gradients: {medians}")
        self.print_to_log_file(f"[GRAD-LR] LR multipliers: {multipliers}")

    def _restore_gradient_lr_calibration(self):
        """Restore the original calibration instead of recalibrating mid-run."""
        candidates = [
            join(self.output_folder, "gradient_lr_calibration_initial.json"),
            join(self.output_folder, "gradient_lr_calibration.json"),
        ]
        report_path = next((path for path in candidates if os.path.exists(path)), None)
        if report_path is None:
            raise RuntimeError("Cannot resume gradient-LR control without its calibration report")
        with open(report_path) as handle:
            report = json.load(handle)
        self._gradient_lr_multipliers = {
            label: float(value) for label, value in report["lr_multiplier"].items()
        }
        self._gradient_control_calibrated = True
        self.print_to_log_file(f"[GRAD-LR] Restored original calibration: {report_path}")
        self.print_to_log_file(f"[GRAD-LR] LR multipliers: {self._gradient_lr_multipliers}")

    def load_checkpoint_ram(self, checkpoint, train=True, network_old=False, checkpoint_old=None):
        """Build matching parameter groups before nnU-Net restores optimizer state."""
        if train and not network_old and self.gradient_lr_control and checkpoint.get("epoch", 0) > 0:
            self._restore_gradient_lr_calibration()
            self._assemble_with_freeze_policy(self.task)
            self.initialize_optimizer_and_scheduler()
            self._gradient_control_resumed = True
            self._gradient_resume_optimizer_state = checkpoint.get("optimizer_state_dict")
        return super().load_checkpoint_ram(checkpoint, train, network_old, checkpoint_old)

    def _snapshot_gradient_control_weights(self):
        self._gradient_control_initial_weights = {
            id(param): param.detach().cpu().clone()
            for label, params in self._controlled_block_param_groups().items()
            if label not in {"head", "body_other"}
            for param in params
        }

    def _log_gradient_control_displacement(self):
        epoch_number = self.epoch + 1
        if epoch_number % self.gradient_displacement_log_interval != 0 and epoch_number != self.max_num_epochs:
            return
        if not self._gradient_control_initial_weights:
            return

        lr_by_block = {
            group.get("block_name", "all"): group["lr"]
            for group in self.optimizer.param_groups
        }
        rows = []
        for label, params in self._controlled_block_param_groups().items():
            if label in {"head", "body_other"}:
                continue
            param_sq = 0.0
            delta_sq = 0.0
            for param in params:
                initial = self._gradient_control_initial_weights.get(id(param))
                if initial is None:
                    continue
                current = param.detach().cpu().float()
                initial = initial.float()
                param_sq += float(torch.sum(initial ** 2).item())
                delta_sq += float(torch.sum((current - initial) ** 2).item())
            relative = 100.0 * (delta_sq ** 0.5) / ((param_sq ** 0.5) + 1e-12)
            rows.append({
                "epoch": epoch_number,
                "block": label,
                "relative_displacement_percent": relative,
                "learning_rate": lr_by_block.get(label, float("nan")),
                "lr_multiplier": self._gradient_lr_multipliers.get(label, 1.0),
            })

        csv_name = (
            "gradient_lr_displacement_since_resume.csv"
            if self._gradient_control_resumed
            else "gradient_lr_displacement.csv"
        )
        csv_path = join(self.output_folder, csv_name)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        summary = {row["block"]: round(row["relative_displacement_percent"], 4) for row in rows}
        self.print_to_log_file(f"[GRAD-LR] Epoch {epoch_number} relative displacement (%): {summary}")

    def print_trainable_params(self):
        total = 0
        trainable = 0

        for p in self.network.parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num

        pct = 100.0 * trainable / total if total > 0 else 0.0
        self.print_to_log_file(f"Total params: {total:,}")
        self.print_to_log_file(f"Trainable params: {trainable:,}")
        self.print_to_log_file(f"Trainable %: {pct:.4f}%")

    # ---------------------------
    # Freeze Policy
    # ---------------------------

    def _apply_unfreeze_lists(self):
        """
        Unfreeze specified encoder/decoder blocks (+ td/tu if present).
        Assumes body already frozen.
        """
        # Encoder blocks + downsampling (td)
        if hasattr(self.network, "conv_blocks_context"):
            n_enc = len(self.network.conv_blocks_context)
            for idx in self.encoder_unfreeze:
                if 0 <= idx < n_enc:
                    self._set_module_requires_grad(self.network.conv_blocks_context[idx], True)
                    if hasattr(self.network, "td") and idx < len(self.network.td):
                        self._set_module_requires_grad(self.network.td[idx], True)
                else:
                    raise ValueError(
                        f"encoder_unfreeze index {idx} is outside [0, {max(0, n_enc - 1)}]"
                    )
        else:
            if len(self.encoder_unfreeze) > 0:
                self.print_to_log_file("[WARNING] encoder_unfreeze specified but network has no conv_blocks_context")

        # Decoder blocks + upsampling (tu)
        if hasattr(self.network, "conv_blocks_localization"):
            n_dec = len(self.network.conv_blocks_localization)
            for cli_idx in self.decoder_unfreeze:
                mapped_idx = self._map_decoder_cli_idx(cli_idx, n_dec)
                if mapped_idx is not None:
                    self._set_module_requires_grad(self.network.conv_blocks_localization[mapped_idx], True)
                    if hasattr(self.network, "tu") and mapped_idx < len(self.network.tu):
                        self._set_module_requires_grad(self.network.tu[mapped_idx], True)
                else:
                    raise ValueError(
                        f"decoder_unfreeze index {cli_idx} is outside [0, {max(0, n_dec - 1)}]"
                    )
        else:
            if len(self.decoder_unfreeze) > 0:
                self.print_to_log_file("[WARNING] decoder_unfreeze specified but network has no conv_blocks_localization")

    def _assemble_with_freeze_policy(self, task):
        """
        Assemble model and apply freezing.
        Returns freeze_body flag for logging.
        """
        self.network = self.mh_network.assemble_model(task, freeze_body=False)

        is_second_task_or_later = self._is_second_task_or_later()
        if not is_second_task_or_later:
            # Task 1: train normally
            self._ensure_head_trainable()
            self._log_param_identity_sanity(where="assemble(Task1)")
            return False

        # Task >= 2: freeze entire shared body (robust)
        self._freeze_entire_body()

        if not self.full_body_freeze:
            self._apply_unfreeze_lists()

        # Heads always trainable
        self._ensure_head_trainable()

        self._log_param_identity_sanity(where="assemble(Task>=2)")
        return True

    # ---------------------------
    # Logging
    # ---------------------------

    def _collect_param_freeze_lists(self, max_names=30):
        trainable = []
        frozen = []
        for name, param in self.network.named_parameters():
            (trainable if param.requires_grad else frozen).append(name)
        return trainable[:max_names], frozen[:max_names], len(trainable), len(frozen)

    def _log_freeze_state(self, task, freeze_body):
        trn_numel, tot_numel, pct = self._summarize_requires_grad()
        self.print_to_log_file(f"[FREEZE] task={task} freeze_body={freeze_body}")
        self.print_to_log_file(
            f"[FREEZE] full_body_freeze={self.full_body_freeze} "
            f"encoder={self.encoder_unfreeze} decoder={self.decoder_unfreeze} seed={self.run_seed}"
        )
        self.print_to_log_file(f"[FREEZE] Trainable elements: {trn_numel}/{tot_numel} ({pct:.3f}%)")
        self.print_trainable_params()

        # quick structure sizes
        if hasattr(self.network, "conv_blocks_context"):
            self.print_to_log_file(f"[FREEZE] encoder_blocks={len(self.network.conv_blocks_context)}")
        if hasattr(self.network, "conv_blocks_localization"):
            self.print_to_log_file(f"[FREEZE] decoder_blocks={len(self.network.conv_blocks_localization)}")

        # module-level trainability summary for selected indices
        if hasattr(self.network, "conv_blocks_context") and len(self.network.conv_blocks_context) > 0:
            idxs = sorted(set([0, len(self.network.conv_blocks_context) - 1] + self.encoder_unfreeze))
            idxs = [i for i in idxs if 0 <= i < len(self.network.conv_blocks_context)]
            for i in idxs:
                trn, tot = self._count_trainable_params_in_module(self.network.conv_blocks_context[i])
                self.print_to_log_file(f"[FREEZE] E{i} trainable={trn}/{tot}")

        if hasattr(self.network, "conv_blocks_localization") and len(self.network.conv_blocks_localization) > 0:
            idxs = sorted(set([0, len(self.network.conv_blocks_localization) - 1] + self.decoder_unfreeze))
            idxs = [i for i in idxs if 0 <= i < len(self.network.conv_blocks_localization)]
            for cli_idx in idxs:
                mapped_idx = self._map_decoder_cli_idx(cli_idx, len(self.network.conv_blocks_localization))
                trn, tot = self._count_trainable_params_in_module(self.network.conv_blocks_localization[mapped_idx])
                self.print_to_log_file(
                    f"[FREEZE] D{cli_idx} (internal={mapped_idx}) trainable={trn}/{tot}"
                )

        # sample name lists (useful if you need to grep)
        sample_trainable, sample_frozen, n_trainable, n_frozen = self._collect_param_freeze_lists(max_names=15)
        self.print_to_log_file(
            f"[FREEZE] trainable_tensors={n_trainable} frozen_tensors={n_frozen}"
        )
        self.print_to_log_file(f"[FREEZE] trainable_examples={sample_trainable}")
        self.print_to_log_file(f"[FREEZE] frozen_examples={sample_frozen}")

    # ---------------------------
    # Training / Validation
    # ---------------------------

    def run_training(self, task, output_folder, build_folder=True):
        """
        Run training. Applies freeze policy before initializing optimizer so optimizer only tracks trainable params.
        """
        if build_folder:
            self.output_folder = join(self._build_output_path(output_folder, False), f"fold_{self.fold}")
        else:
            self.output_folder = output_folder

        maybe_mkdir_p(self.output_folder)

        if self.task != task:
            self.reinitialize(task)
            self.task = task

        self.update_save_trained_on_json(task, False)

        if task not in self.mh_network.heads:
            self.mh_network.add_new_task(task, use_init=not self.transfer_heads)

        if self.use_vit and self.ViT_task_specific_ln:
            if task not in self.network.ViT.norm:
                self.network.ViT.register_new_task(task)
                self.mh_network.model = copy.deepcopy(self.network)
            self.network.ViT.use_task(task)

        freeze_body = self._assemble_with_freeze_policy(task)
        self._log_freeze_state(task, freeze_body)

        # IMPORTANT: optimizer must be initialized AFTER requires_grad changes
        self.initialize_optimizer_and_scheduler()
        if self.gradient_lr_control and self._is_second_task_or_later():
            self._calibrate_gradient_lr_control()
            # Rebuild SGD with calibrated block-specific learning rates.
            self.initialize_optimizer_and_scheduler()
            if self._gradient_resume_optimizer_state is not None:
                self.optimizer.load_state_dict(self._gradient_resume_optimizer_state)
                self.print_to_log_file("[GRAD-LR] Restored block optimizer and momentum state")
            self._snapshot_gradient_control_weights()
        self.trainer_model = None

        ret = nnUNetTrainerV2.run_training(self)

        self.already_trained_on[str(self.fold)]["val_metrics_should_exist"] = False
        self.update_save_trained_on_json(task, True)
        self.save_init_args(join(self.output_folder, "model_final_checkpoint.model"))

        if self.new_trainer and len(self.already_trained_on) > 1:
            self.new_trainer = False

        # Reset stats (nnUNet style)
        self.epoch = 0
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []
        self.validation_results = dict()

        return ret

    def on_epoch_end(self):
        ret = super().on_epoch_end()
        if self.gradient_lr_control and self._is_second_task_or_later():
            self._log_gradient_control_displacement()
        return ret

    def _perform_validation(
        self,
        use_tasks=None,
        use_head=None,
        call_for_eval=False,
        param_search=False,
        use_all_data=False,
    ):
        """
        Validation may switch heads/assemble models internally.
        We restore training model after validation and (optionally) re-init optimizer to be safe.
        """
        ret = super()._perform_validation(
            use_tasks=use_tasks,
            use_head=use_head,
            call_for_eval=call_for_eval,
            param_search=param_search,
            use_all_data=use_all_data,
        )

        if call_for_eval:
            return ret

        if self._is_second_task_or_later():
            freeze_body = self._assemble_with_freeze_policy(self.task)
            self.print_to_log_file("[SANITY] Reassembled model after validation and restored freeze policy")

            if self.reinit_optimizer_after_validation:
                self.initialize_optimizer_and_scheduler()
                self.print_to_log_file("[SANITY] Reinitialized optimizer after validation reassembly")
            else:
                self.print_to_log_file(
                    "[WARNING] Optimizer NOT reinitialized after validation reassembly. "
                    "This is only safe if assemble_model keeps identical Parameter objects."
                )

            # brief sanity
            self._log_freeze_state(self.task, freeze_body)

        return ret
