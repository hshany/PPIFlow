#!/usr/bin/env python3
"""Unified entry point for the binder refinement pipeline."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


STEP_NAMES = ("partial_flow", "seq_design", "prep", "flowpacker", "af3score")


class PipelineError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run binder refinement pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (see configs/pipeline_template.yaml).",
    )
    parser.add_argument(
        "--repo-root",
        help="Override repo root if different from config. Defaults to repo checkout.",
    )
    parser.add_argument(
        "--run-root",
        help="Override run root if different from config.",
    )
    parser.add_argument(
        "--steps",
        help="Comma-separated subset of steps to run "
        "(partial_flow,seq_design,prep,flowpacker,af3score).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if completion markers exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands/actions without executing them.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_strings(obj, mapping: Dict[str, str]):
    if isinstance(obj, dict):
        return {k: resolve_strings(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_strings(v, mapping) for v in obj]
    if isinstance(obj, str):
        result = obj
        for key, value in mapping.items():
            result = result.replace(f"${{{key}}}", value)
        return result
    return obj


@dataclass
class InputConfig:
    pdb_path: Path
    fixed_positions_csv: Path
    receptor_chain: str
    binder_chain: str


@dataclass
class DerivedPaths:
    run_root: Path
    sample_name: str
    receptor_chain: str

    def __post_init__(self):
        self.run_root.mkdir(parents=True, exist_ok=True)

    @property
    def partial_output_root(self) -> Path:
        return self.run_root / "output"

    @property
    def partial_flow_dir(self) -> Path:
        return self.partial_output_root / f"sample_{self.sample_name}_{self.receptor_chain}"

    @property
    def seq_dir(self) -> Path:
        return self.run_root / "seq"

    @property
    def link_dir(self) -> Path:
        return self.run_root / "pf" / "link_samples"

    @property
    def fa_csv(self) -> Path:
        return self.run_root / "fa.csv"

    @property
    def pf_fa_sum(self) -> Path:
        return self.run_root / "pf_fa_sum.csv"

    @property
    def af_root(self) -> Path:
        return self.run_root / "af3score"

    @property
    def flowpacker_root(self) -> Path:
        return self.af_root / "flowpacker"

    @property
    def chain_swap_dir(self) -> Path:
        return self.af_root / "pf_link_samples_chainA"

    @property
    def logs_dir(self) -> Path:
        return self.run_root / "logs"

    @property
    def af3_base_out(self) -> Path:
        return self.af_root / "af3score_base_outputs"


def to_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def log(message: str):
    print(f"[pipeline] {message}")


def run_command(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
):
    quoted = " ".join(shlex.quote(part) for part in cmd)
    if cwd:
        log(f"(cwd: {cwd}) $ {quoted}")
    else:
        log(f"$ {quoted}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


def conda_cmd(env_name: str, command: Iterable[str]) -> List[str]:
    return ["conda", "run", "-n", env_name, *command]


def ensure_file(path: Path, description: str):
    if not path.exists():
        raise PipelineError(f"Missing {description}: {path}")


def ensure_dir(path: Path, description: str):
    if not path.is_dir():
        raise PipelineError(f"Missing directory for {description}: {path}")


def create_marker(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("done\n", encoding="utf-8")


def marker_path(paths: DerivedPaths, step_name: str) -> Path:
    return paths.logs_dir / f".{step_name}.done"


def run_partial_flow(
    repo_root: Path,
    cfg: Dict,
    inputs: InputConfig,
    paths: DerivedPaths,
    *,
    dry_run: bool,
):
    script = repo_root / "sample_binder_partial.py"
    checkpoint = repo_root / "checkpoints" / "binder.ckpt"
    config_path_value = cfg.get("config")
    if not config_path_value:
        raise PipelineError("partial_flow.config must be set in the config file")
    config_path = Path(config_path_value)
    ensure_file(script, "sample_binder_partial.py")
    ensure_file(checkpoint, "partial flow checkpoint")
    ensure_file(config_path, "partial flow config")
    out_root = paths.partial_output_root
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(script),
            "--input_pdb",
            str(inputs.pdb_path),
            "--target_chain",
            inputs.receptor_chain,
            "--binder_chain",
            inputs.binder_chain,
            "--config",
            str(config_path),
            "--model_weights",
            str(checkpoint),
            "--output_dir",
            str(out_root),
            "--name",
            paths.sample_name,
            "--start_t",
            str(cfg.get("start_t", 0.6)),
            "--samples_per_target",
            str(cfg.get("samples_per_target", 5)),
        ],
    )
    run_command(cmd, cwd=repo_root, dry_run=dry_run)


def run_seq_design(
    repo_root: Path,
    cfg: Dict,
    inputs: InputConfig,
    paths: DerivedPaths,
    *,
    dry_run: bool,
):
    script = repo_root / "ProteinMPNN" / "protein_mpnn_run.py"
    weights = repo_root / "ProteinMPNN" / "model_weights"
    ensure_file(script, "ProteinMPNN runner")
    ensure_dir(weights, "ProteinMPNN weights")
    ensure_dir(paths.partial_flow_dir, "partial flow output")
    ensure_file(inputs.fixed_positions_csv, "fixed positions CSV")
    paths.seq_dir.mkdir(parents=True, exist_ok=True)
    cmd = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(script),
            "--path_to_model_weights",
            str(weights),
            "--model_name",
            "abmpnn",
            "--folder_with_pdbs_path",
            str(paths.partial_flow_dir),
            "--chain_list",
            cfg.get("chain_list", "B"),
            "--position_list",
            str(inputs.fixed_positions_csv),
            "--num_seq_per_target",
            str(cfg.get("num_seq_per_target", 8)),
            "--sampling_temp",
            str(cfg.get("sampling_temp", 0.1)),
            "--batch_size",
            str(cfg.get("batch_size", 8)),
            "--out_folder",
            str(paths.seq_dir),
        ],
    )
    run_command(cmd, cwd=repo_root, dry_run=dry_run)


def parse_fasta(fasta_path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seq_buffer: List[str] = []
    seq_idx = -1
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_buffer:
                    records.append(
                        {
                            "fasta": str(fasta_path),
                            "seq": "".join(seq_buffer),
                            "seq_idx": str(seq_idx),
                        }
                    )
                    seq_buffer = []
                seq_idx += 1
            else:
                seq_buffer.append(line)
        if seq_buffer:
            records.append(
                {
                    "fasta": str(fasta_path),
                    "seq": "".join(seq_buffer),
                    "seq_idx": str(seq_idx),
                }
            )
    return records


def run_prep_step(
    cfg: Dict,
    paths: DerivedPaths,
    *,
    dry_run: bool,
):
    ensure_dir(paths.partial_flow_dir, "partial flow output dir")
    ensure_dir(paths.seq_dir, "sequence output dir")
    if dry_run:
        log(f"Would link PDBs from {paths.partial_flow_dir} -> {paths.link_dir}")
    else:
        paths.link_dir.mkdir(parents=True, exist_ok=True)
        for pdb in sorted(paths.partial_flow_dir.glob("*.pdb")):
            dst = paths.link_dir / pdb.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(pdb.resolve(), dst)
    fasta_records: List[Dict[str, str]] = []
    for fasta in sorted(paths.seq_dir.rglob("*.fa")):
        fasta_records.extend(parse_fasta(fasta))
    log(f"Found {len(fasta_records)} FASTA records")
    if dry_run:
        return
    paths.fa_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.fa_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fasta", "seq", "seq_idx"])
        writer.writeheader()
        for record in fasta_records:
            writer.writerow(record)
    drop_first = cfg.get("drop_first_seq_idx", True)
    summary_rows: List[Dict[str, str]] = []
    for record in fasta_records:
        try:
            idx_int = int(record["seq_idx"])
        except ValueError:
            idx_int = None
        if drop_first and idx_int == 0:
            continue
        fasta_name = Path(record["fasta"]).stem
        link_name = f"{fasta_name}.pdb"
        summary_rows.append(
            {"link_name": link_name, "seq": record["seq"], "seq_idx": record["seq_idx"]}
        )
    with paths.pf_fa_sum.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["link_name", "seq", "seq_idx"])
        writer.writeheader()
        writer.writerows(summary_rows)
    log(
        f"Wrote {len(summary_rows)} FlowPacker mapping rows to {paths.pf_fa_sum}"
    )


def swap_chains(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for pdb in sorted(input_dir.glob("*.pdb")):
        dst = output_dir / pdb.name
        lines = pdb.read_text().splitlines()
        new_lines = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 22:
                chain = line[21]
                if chain == "A":
                    chain = "B"
                elif chain == "B":
                    chain = "A"
                line = f"{line[:21]}{chain}{line[22:]}"
            new_lines.append(line)
        dst.write_text("\n".join(new_lines) + "\n")


def run_flowpacker(
    repo_root: Path,
    cfg: Dict,
    paths: DerivedPaths,
    *,
    dry_run: bool,
):
    scripts_dir = repo_root / "demo_scripts" / "flowpacker_af3score"
    flowpacker_repo = repo_root / "flowpacker-main"
    base_yaml_value = cfg.get("base_yaml")
    if not base_yaml_value:
        raise PipelineError("flowpacker.base_yaml must be set in the config file")
    base_yaml = Path(base_yaml_value)
    checkpoint = flowpacker_repo / "checkpoints" / "cluster.pth"
    ensure_dir(paths.link_dir, "linked partial-flow PDBs")
    ensure_file(paths.pf_fa_sum, "sequence mapping CSV")
    ensure_dir(scripts_dir, "FlowPacker scripts directory")
    ensure_dir(flowpacker_repo, "FlowPacker repo")
    ensure_file(base_yaml, "FlowPacker base YAML")
    ensure_file(checkpoint, "FlowPacker checkpoint")
    chain_swap_dir = paths.chain_swap_dir
    if chain_swap_dir.exists() and not dry_run:
        shutil.rmtree(chain_swap_dir)
    if dry_run:
        log(f"Would swap chains into {chain_swap_dir}")
    else:
        swap_chains(paths.link_dir, chain_swap_dir)
    flowpacker_root = paths.flowpacker_root
    batch_pdb_dir = flowpacker_root / "input_pdb_batch"
    yaml_dir = flowpacker_root / "batch_yml"
    outputs_dir = flowpacker_root / "flowpacker_outputs"
    for directory in (flowpacker_root, batch_pdb_dir, yaml_dir, outputs_dir):
        if directory.exists() and not dry_run:
            shutil.rmtree(directory)
        if not dry_run:
            directory.mkdir(parents=True, exist_ok=True)
    split_script = scripts_dir / "1-split_batch.py"
    run_yaml_script = scripts_dir / "2-run_flowpacker.py"
    sampler_script = scripts_dir / "sampler_pdb_pipe.py"
    ensure_file(split_script, "FlowPacker split script")
    ensure_file(run_yaml_script, "FlowPacker YAML generator")
    ensure_file(sampler_script, "FlowPacker sampler script")
    cmd_split = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(split_script),
            "--base_path",
            str(chain_swap_dir),
            "--output_folder_base",
            str(batch_pdb_dir),
            "--num_of_jobs",
            str(cfg.get("num_jobs", 1)),
        ],
    )
    run_command(cmd_split, cwd=repo_root, dry_run=dry_run)
    cmd_yaml = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(run_yaml_script),
            "--base_yaml_path",
            str(base_yaml),
            "--base_pdb_path",
            str(batch_pdb_dir),
            "--output_yaml_dir",
            str(yaml_dir),
        ],
    )
    run_command(cmd_yaml, cwd=repo_root, dry_run=dry_run)
    if dry_run:
        log("Would patch FlowPacker YAMLs and run sampler")
        return
    yaml_files = sorted(yaml_dir.glob("*.yml"))
    if not yaml_files:
        raise PipelineError(f"No FlowPacker YAML files generated in {yaml_dir}")
    for yaml_file in yaml_files:
        with yaml_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        data["ckpt"] = str(checkpoint)
        with yaml_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False)
        cmd = conda_cmd(
            cfg["conda_env"],
            [
                "python",
                str(sampler_script),
                str(yaml_file),
                "--save_dir",
                str(outputs_dir),
                "--use_gt_masks",
                "True",
                "--csv_file",
                str(paths.pf_fa_sum),
            ],
        )
        run_command(cmd, cwd=repo_root, dry_run=False)
    log(f"FlowPacker outputs stored in {outputs_dir}")


def run_af3score(
    repo_root: Path,
    cfg: Dict,
    paths: DerivedPaths,
    *,
    dry_run: bool,
):
    scripts_dir = repo_root / "demo_scripts" / "flowpacker_af3score"
    flowpacker_outputs = paths.flowpacker_root / "flowpacker_outputs"
    run_dir = flowpacker_outputs / "run_1"
    ensure_dir(scripts_dir, "FlowPacker/AF3Score scripts directory")
    ensure_dir(run_dir, "FlowPacker output (run_1)")
    db_dir = Path(cfg["db_dir"])
    model_dir = Path(cfg["model_dir"])
    ensure_dir(db_dir, "AF3Score database directory")
    ensure_dir(model_dir, "AF3Score model directory")
    prepare_script = scripts_dir / "4.1-prepare_get_json.py"
    pdb2jax_script = scripts_dir / "4.2_prepare_pdb2jax.py"
    af3_script = scripts_dir / "run_af3score.py"
    metrics_script = scripts_dir / "easy_get_metrics.py"
    for script in (prepare_script, pdb2jax_script, af3_script, metrics_script):
        ensure_file(script, f"{script.name} script")
    base_out = paths.af3_base_out
    if base_out.exists() and not dry_run:
        shutil.rmtree(base_out)
    if not dry_run:
        (base_out / "af3_input_batch").mkdir(parents=True, exist_ok=True)
        (base_out / "single_chain_cif").mkdir(parents=True, exist_ok=True)
        (base_out / "json").mkdir(parents=True, exist_ok=True)
        (base_out / "af3score_outputs").mkdir(parents=True, exist_ok=True)
    single_seq_csv = base_out / "single_seq.csv"
    af3_input_batch = base_out / "af3_input_batch"
    af3_cif_dir = base_out / "single_chain_cif"
    af3_json_dir = base_out / "json"
    af3_jax_dir = af3_input_batch / "jax"
    af3_out_dir = base_out / "af3score_outputs"
    af3_metrics = base_out / "af3score_metrics.csv"
    if not dry_run:
        af3_jax_dir.mkdir(parents=True, exist_ok=True)
    cmd_prepare = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(prepare_script),
            "--input_dir",
            str(run_dir),
            "--output_dir_cif",
            str(af3_cif_dir),
            "--save_csv",
            str(single_seq_csv),
            "--output_dir_json",
            str(af3_json_dir),
            "--batch_dir",
            str(af3_input_batch),
            "--num_jobs",
            str(cfg.get("num_jobs", 1)),
        ],
    )
    run_command(cmd_prepare, cwd=repo_root, dry_run=dry_run)
    if dry_run:
        log("Would build AF3Score batches and run inference/metrics")
        return
    pdb_batches = sorted((af3_input_batch / "pdb").glob("*"))
    for pdb_batch in pdb_batches:
        if not pdb_batch.is_dir():
            continue
        bucket_name = pdb_batch.name
        output_folder = af3_jax_dir / bucket_name
        output_folder.mkdir(parents=True, exist_ok=True)
        cmd = conda_cmd(
            cfg["conda_env"],
            [
                "python",
                str(pdb2jax_script),
                "--pdb_folder",
                str(pdb_batch),
                "--output_folder",
                str(output_folder),
            ],
        )
        run_command(cmd, cwd=repo_root, dry_run=False)
    json_batches = sorted((af3_input_batch / "json").glob("*"))
    for json_batch in json_batches:
        if not json_batch.is_dir():
            continue
        bucket_name = json_batch.name
        bucket_match = re.search(r"(\d+)$", bucket_name)
        buckets = bucket_match.group(1) if bucket_match else ""
        cmd = conda_cmd(
            cfg["conda_env"],
            [
                "python",
                str(af3_script),
                f"--db_dir={db_dir}",
                f"--model_dir={model_dir}",
                f"--batch_json_dir={json_batch}",
                f"--batch_h5_dir={af3_jax_dir / bucket_name}",
                f"--output_dir={af3_out_dir}",
                "--run_data_pipeline=False",
                "--run_inference=true",
                "--init_guess=true",
                f"--num_samples={cfg.get('num_samples', 1)}",
                f"--buckets={buckets}",
                "--write_cif_model=False",
                "--write_summary_confidences=true",
                "--write_full_confidences=true",
                "--write_best_model_root=false",
                "--write_ranking_scores_csv=false",
                "--write_terms_of_use_file=false",
                "--write_fold_input_json_file=false",
            ],
        )
        run_command(cmd, cwd=repo_root, dry_run=False)
    cmd_metrics = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(metrics_script),
            str(af3_out_dir),
            str(af3_metrics),
        ],
    )
    run_command(cmd_metrics, cwd=repo_root, dry_run=False)
    log(f"AF3Score metrics written to {af3_metrics}")


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise PipelineError(f"Config not found: {config_path}")
    raw_cfg = load_yaml(config_path)
    default_repo = Path(__file__).resolve().parents[2]
    repo_root_str = args.repo_root or raw_cfg.get("repo_root") or str(default_repo)
    repo_root = Path(repo_root_str).expanduser().resolve()
    run_root_str = args.run_root or raw_cfg.get("run_root")
    if not run_root_str:
        raise PipelineError("run_root must be provided via config or --run-root")
    run_root = Path(run_root_str).expanduser().resolve()
    mapping = {"repo_root": str(repo_root), "run_root": str(run_root)}
    cfg = resolve_strings(raw_cfg, mapping)
    io_cfg = cfg.get("io", {})
    input_cfg = io_cfg.get("inputs")
    if not input_cfg:
        raise PipelineError("io.inputs section missing from config")
    try:
        cwd = Path.cwd().resolve()
        inputs = InputConfig(
            pdb_path=to_path(input_cfg["pdb"], cwd),
            fixed_positions_csv=to_path(input_cfg["fixed_positions_csv"], cwd),
            receptor_chain=str(input_cfg["receptor_chain"]),
            binder_chain=str(input_cfg["binder_chain"]),
        )
    except KeyError as exc:
        raise PipelineError(f"Missing input setting: {exc}") from exc
    ensure_file(inputs.pdb_path, "input PDB")
    ensure_file(inputs.fixed_positions_csv, "fixed positions CSV")
    sample_name = inputs.pdb_path.stem
    paths = DerivedPaths(run_root=run_root, sample_name=sample_name, receptor_chain=inputs.receptor_chain)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    requested_steps: Optional[List[str]] = None
    if args.steps:
        requested_steps = [step.strip() for step in args.steps.split(",") if step.strip()]
        invalid = sorted(set(requested_steps) - set(STEP_NAMES))
        if invalid:
            raise PipelineError(f"Unknown step(s) requested: {', '.join(invalid)}")
    else:
        requested_steps = list(STEP_NAMES)
    dry_run = args.dry_run
    step_settings = {name: cfg.get(name, {}) for name in STEP_NAMES}
    step_functions = {
        "partial_flow": lambda: run_partial_flow(
            repo_root, step_settings["partial_flow"], inputs, paths, dry_run=dry_run
        ),
        "seq_design": lambda: run_seq_design(
            repo_root, step_settings["seq_design"], inputs, paths, dry_run=dry_run
        ),
        "prep": lambda: run_prep_step(
            step_settings["prep"], paths, dry_run=dry_run
        ),
        "flowpacker": lambda: run_flowpacker(
            repo_root, step_settings["flowpacker"], paths, dry_run=dry_run
        ),
        "af3score": lambda: run_af3score(
            repo_root, step_settings["af3score"], paths, dry_run=dry_run
        ),
    }
    for step in STEP_NAMES:
        settings = step_settings[step]
        enabled = settings.get("enabled", True)
        env = settings.get("conda_env")
        if not enabled:
            log(f"Skipping {step}: disabled in config")
            continue
        if requested_steps and step not in requested_steps:
            log(f"Skipping {step}: not requested")
            continue
        if step != "prep" and not env:
            raise PipelineError(f"conda_env must be set for step '{step}'")
        marker = marker_path(paths, step)
        if marker.exists() and not args.force:
            log(f"Skipping {step}: marker {marker} exists (use --force to rerun)")
            continue
        log(f"=== Running step: {step} ===")
        step_functions[step]()
        if not dry_run:
            create_marker(marker)
    log("Pipeline complete")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
