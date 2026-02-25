#!/usr/bin/env python3
"""Unified entry point for the binder refinement pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple

import yaml


STEP_NAMES = (
    "partial_flow",
    "seq_design",
    "prep",
    "flowpacker",
    "af3score",
    "rosetta_relax",
    "af3_refold",
    "dockq",
)
DEFAULT_ENABLED_STEPS = {"partial_flow", "seq_design", "prep", "flowpacker", "af3score"}


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

    @property
    def af3_filtered_dir(self) -> Path:
        return self.af_root / "filtered_links"

    @property
    def af3_filtered_summary(self) -> Path:
        return self.af3_filtered_dir / "af3score_filtered.csv"

    @property
    def rosetta_root(self) -> Path:
        return self.run_root / "rosetta_relax"

    @property
    def rosetta_inputs_csv(self) -> Path:
        return self.rosetta_root / "rosetta_inputs.csv"

    @property
    def rosetta_results_dir(self) -> Path:
        return self.rosetta_root / "rst_itf_nofix"

    @property
    def rosetta_filtered_dir(self) -> Path:
        return self.rosetta_root / "filtered_links"

    @property
    def rosetta_filtered_summary(self) -> Path:
        return self.rosetta_filtered_dir / "rosetta_filtered.csv"

    @property
    def af3_refold_root(self) -> Path:
        return self.run_root / "af3_refold"

    @property
    def af3_refold_base_out(self) -> Path:
        return self.af3_refold_root / "af3_refold_base_outputs"

    @property
    def af3_refold_single_seq_csv(self) -> Path:
        return self.af3_refold_base_out / "single_seq.csv"

    @property
    def af3_refold_input_batch(self) -> Path:
        return self.af3_refold_base_out / "af3_input_batch"

    @property
    def af3_refold_cif_dir(self) -> Path:
        return self.af3_refold_base_out / "single_chain_cif"

    @property
    def af3_refold_json_dir(self) -> Path:
        return self.af3_refold_base_out / "json"

    @property
    def af3_refold_jax_dir(self) -> Path:
        return self.af3_refold_input_batch / "jax"

    @property
    def af3_refold_out_dir(self) -> Path:
        return self.af3_refold_base_out / "af3score_outputs"

    @property
    def af3_refold_metrics(self) -> Path:
        return self.af3_refold_base_out / "af3_refold_metrics.csv"

    @property
    def af3_refold_pdb_models(self) -> Path:
        return self.af3_refold_root / "pdb_models"

    @property
    def af3_refold_filtered_dir(self) -> Path:
        return self.af3_refold_root / "filtered_links"

    @property
    def af3_refold_filtered_summary(self) -> Path:
        return self.af3_refold_filtered_dir / "af3_refold_filtered.csv"

    @property
    def dockq_root(self) -> Path:
        return self.run_root / "dockq"

    @property
    def dockq_results_dir(self) -> Path:
        return self.dockq_root / "results"

    @property
    def dockq_summary_csv(self) -> Path:
        return self.dockq_results_dir / "summary_dockq_scores.csv"

    @property
    def final_hits_dir(self) -> Path:
        return self.run_root / "final_hits"

    @property
    def final_hits_csv(self) -> Path:
        return self.final_hits_dir / "dockq_filtered.csv"


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
    stdout: Optional[TextIO] = None,
):
    quoted = " ".join(shlex.quote(part) for part in cmd)
    if cwd:
        log(f"(cwd: {cwd}) $ {quoted}")
    else:
        log(f"$ {quoted}")
    if dry_run:
        return
    if stdout is not None:
        subprocess.run(cmd, check=True, cwd=cwd, stdout=stdout)
        return
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    ret = process.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


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


def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def filter_metrics_by_threshold(
    csv_path: Path, iptm_min: float, ptm_min: float
) -> Tuple[List[Dict[str, str]], List[str]]:
    ensure_file(csv_path, f"metrics CSV {csv_path.name}")
    filtered: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            fieldnames = list(reader.fieldnames)
        for row in reader:
            try:
                iptm = float(row.get("iptm", "nan"))
                ptm = float(row.get("ptm_A", "nan"))
            except (TypeError, ValueError):
                continue
            if iptm >= iptm_min and ptm >= ptm_min:
                filtered.append(row)
    return filtered, fieldnames


def write_summary_csv(
    path: Path, fieldnames: List[str], rows: List[Dict[str, str]], *, dry_run: bool
):
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def filter_af3score_outputs(
    paths: DerivedPaths, iptm_min: float, ptm_min: float, *, dry_run: bool
) -> List[str]:
    metrics = paths.af3_base_out / "af3score_metrics.csv"
    flowpacker_dir = paths.flowpacker_root / "flowpacker_outputs" / "run_1"
    ensure_file(metrics, "AF3Score metrics CSV")
    ensure_dir(flowpacker_dir, "FlowPacker outputs (run_1)")
    filtered, fieldnames = filter_metrics_by_threshold(metrics, iptm_min, ptm_min)
    log(
        f"{len(filtered)} designs pass AF3Score thresholds iptm>= {iptm_min} & ptm>= {ptm_min}"
    )
    if not filtered:
        return []
    out_dir = paths.af3_filtered_dir
    summary = paths.af3_filtered_summary
    write_summary_csv(summary, fieldnames, filtered, dry_run=dry_run)
    if dry_run:
        return [row.get("description", "") for row in filtered if row.get("description")]
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("*.pdb"):
        existing.unlink()
    linked: List[str] = []
    for row in filtered:
        desc = row.get("description")
        if not desc:
            continue
        src = flowpacker_dir / f"{desc}.pdb"
        if not src.exists():
            log(f"[pipeline] WARNING: Missing FlowPacker PDB for {desc}, skipping")
            continue
        dst = out_dir / f"{desc}.pdb"
        safe_symlink(src.resolve(), dst)
        linked.append(desc)
    log(f"Linked {len(linked)} AF3Score-filtered complexes into {out_dir}")
    return linked


def filter_rosetta_outputs(
    paths: DerivedPaths, interface_max: float, *, dry_run: bool
) -> List[str]:
    result_dir = paths.rosetta_results_dir
    ensure_dir(result_dir, "Rosetta results directory")
    csv_files = sorted(result_dir.glob("rosetta_complex_*.csv"))
    if not csv_files:
        raise PipelineError(f"No Rosetta score CSVs found in {result_dir}")
    filtered_rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and not fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                try:
                    score = float(row.get("interface_score", "nan"))
                except (TypeError, ValueError):
                    continue
                if score <= interface_max:
                    filtered_rows.append(row)
    log(
        f"{len(filtered_rows)} complexes pass Rosetta interface_score <= {interface_max}"
    )
    write_summary_csv(
        paths.rosetta_filtered_summary, fieldnames, filtered_rows, dry_run=dry_run
    )
    if dry_run or not filtered_rows:
        return [row.get("pdb_name", "") for row in filtered_rows if row.get("pdb_name")]
    filtered_dir = paths.rosetta_filtered_dir
    filtered_dir.mkdir(parents=True, exist_ok=True)
    for existing in filtered_dir.glob("*.pdb"):
        existing.unlink()
    kept: List[str] = []
    for row in filtered_rows:
        pdb_name = row.get("pdb_name")
        if not pdb_name:
            continue
        src = result_dir / f"relax_{pdb_name}.pdb"
        if not src.exists():
            log(f"[pipeline] WARNING: Missing relaxed PDB for {pdb_name}")
            continue
        dst = filtered_dir / f"{pdb_name}.pdb"
        safe_symlink(src.resolve(), dst)
        kept.append(pdb_name)
    log(f"Linked {len(kept)} Rosetta-filtered complexes into {filtered_dir}")
    return kept


def run_rosetta_relax(
    repo_root: Path,
    cfg: Dict,
    inputs: InputConfig,
    paths: DerivedPaths,
    *,
    dry_run: bool,
):
    filter_cfg = cfg.get("af3_filter", {})
    iptm_min = float(filter_cfg.get("iptm_min", 0.6))
    ptm_min = float(filter_cfg.get("ptm_min", 0.6))
    filtered = filter_af3score_outputs(paths, iptm_min, ptm_min, dry_run=dry_run)
    if not filtered:
        raise PipelineError(
            "No AF3Score entries pass the configured thresholds; unable to run Rosetta"
        )
    ensure_dir(paths.af3_filtered_dir, "AF3Score-filtered complexes")
    pdb_files = sorted(paths.af3_filtered_dir.glob("*.pdb"))
    if not pdb_files:
        raise PipelineError(
            f"No filtered AF3Score PDBs found in {paths.af3_filtered_dir}"
        )
    rosetta_script = repo_root / "myscripts" / "relax_complex.py"
    ensure_file(rosetta_script, "relax_complex.py")
    ligand_chain = str(cfg.get("ligand_chain", inputs.binder_chain))
    receptor_chain = str(cfg.get("receptor_chain", inputs.receptor_chain))
    fixed_chain = str(cfg.get("fixed_chain", receptor_chain))
    relax_flag = str(cfg.get("relax", True))
    fix_backbone = str(cfg.get("fix_backbone", False))
    max_iter = str(cfg.get("max_iter", 170))
    if not dry_run:
        paths.rosetta_root.mkdir(parents=True, exist_ok=True)
        if paths.rosetta_results_dir.exists():
            shutil.rmtree(paths.rosetta_results_dir)
        if paths.rosetta_filtered_dir.exists():
            shutil.rmtree(paths.rosetta_filtered_dir)
        paths.rosetta_results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = paths.rosetta_inputs_csv
    rows = [
        {"pdb": str(pdb.resolve()), "ligand": ligand_chain, "receptor": receptor_chain}
        for pdb in pdb_files
    ]
    if dry_run:
        log(f"Would write Rosetta input CSV with {len(rows)} entries to {csv_path}")
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pdb", "ligand", "receptor"])
            writer.writeheader()
            writer.writerows(rows)
    interface_max = float(cfg.get("interface_score_max", -8.0))
    if dry_run:
        log("Would run PyRosetta relax/interface analysis and filter results")
        return
    num_workers = str(cfg.get("num_workers", 1))
    cmd = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(rosetta_script),
            "--csv_path",
            str(csv_path),
            "--output_dir",
            str(paths.rosetta_results_dir),
            "--dump_pdb",
            "True",
            "--batch_idx",
            "0",
            "--relax",
            relax_flag,
            "--fixbb",
            fix_backbone,
            "--fixed_chain",
            fixed_chain,
            "--max_iter",
            max_iter,
            "--num_workers",
            num_workers,
        ],
    )
    run_command(cmd, cwd=repo_root, dry_run=False)
    filter_rosetta_outputs(paths, interface_max, dry_run=False)


def convert_cif_outputs_to_pdb(src_root: Path, dst_dir: Path):
    try:
        from Bio.PDB import MMCIFParser, PDBIO
    except ImportError as exc:
        raise PipelineError(
            "Biopython is required to convert AF3 outputs to PDB "
            "(pip install biopython)."
        ) from exc
    parser = MMCIFParser(QUIET=True)  # type: ignore[name-defined]
    io = PDBIO()  # type: ignore[name-defined]
    converted = 0
    for complex_dir in sorted(src_root.iterdir()):
        if not complex_dir.is_dir():
            continue
        cif_files = sorted(complex_dir.rglob("*.cif"))
        if not cif_files:
            continue
        cif_path = cif_files[0]
        try:
            structure = parser.get_structure(complex_dir.name, str(cif_path))
            io.set_structure(structure)
            dst = dst_dir / f"{complex_dir.name}.pdb"
            io.save(str(dst))
            converted += 1
        except Exception as exc:  # pragma: no cover - best effort logging
            log(f"[pipeline] WARNING: Failed to convert {cif_path}: {exc}")
    log(f"Converted {converted} mmCIF complexes to PDB in {dst_dir}")


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path):
    try:
        from Bio.PDB import MMCIFParser, PDBIO
    except ImportError as exc:
        raise PipelineError(
            "Biopython is required to convert AF3 outputs to PDB "
            "(pip install biopython)."
        ) from exc
    parser = MMCIFParser(QUIET=True)  # type: ignore[name-defined]
    io = PDBIO()  # type: ignore[name-defined]
    structure = parser.get_structure(pdb_path.stem, str(cif_path))
    io.set_structure(structure)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(pdb_path))


def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    try:
        from Bio.PDB import PDBParser
    except ImportError as exc:
        raise PipelineError(
            "Biopython is required to extract peptide sequences "
            "(pip install biopython)."
        ) from exc
    protein_letters_3to1 = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
        "MSE": "M",
    }
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    if not structure:
        raise PipelineError(f"Failed to parse PDB: {pdb_path}")
    try:
        chain = structure[0][chain_id]
    except KeyError as exc:
        raise PipelineError(f"Chain {chain_id} not found in {pdb_path}") from exc
    sequence = ""
    for residue in chain:
        if residue.id[0] == " ":
            resname = residue.get_resname().upper()
            sequence += protein_letters_3to1.get(resname, "X")
    return sequence


def _query_only_msa(sequence: str) -> str:
    return f">query\n{sequence}\n"


def write_af3_base_json(
    output_path: Path,
    *,
    name: str,
    receptor_chain: str,
    receptor_sequence: str,
    receptor_msa_path: Path,
    ligand_chain: str,
    ligand_sequence: str,
    model_seeds: Iterable[int],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": name,
        "sequences": [
            {
                "protein": {
                    "id": ligand_chain,
                    "sequence": ligand_sequence,
                    "modifications": [],
                    "unpairedMsa": _query_only_msa(ligand_sequence),
                    "pairedMsa": _query_only_msa(ligand_sequence),
                    "templates": [],
                }
            },
            {
                "protein": {
                    "id": receptor_chain,
                    "sequence": receptor_sequence,
                    "modifications": [],
                    "unpairedMsaPath": str(receptor_msa_path),
                    "pairedMsa": _query_only_msa(receptor_sequence),
                    "templates": [],
                }
            },
        ],
        "modelSeeds": list(model_seeds),
        "bondedAtomPairs": None,
        "userCCD": None,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def filter_af3_refold_outputs(
    paths: DerivedPaths, iptm_min: float, ptm_min: float, *, dry_run: bool
) -> List[str]:
    metrics = paths.af3_refold_metrics
    pdb_dir = paths.af3_refold_pdb_models
    ensure_file(metrics, "AF3 refold metrics CSV")
    ensure_dir(pdb_dir, "AF3 refold PDB directory")
    filtered, fieldnames = filter_metrics_by_threshold(metrics, iptm_min, ptm_min)
    log(
        f"{len(filtered)} designs pass AF3 refold thresholds iptm>= {iptm_min} & ptm>= {ptm_min}"
    )
    write_summary_csv(
        paths.af3_refold_filtered_summary, fieldnames, filtered, dry_run=dry_run
    )
    if dry_run or not filtered:
        return [row.get("description", "") for row in filtered if row.get("description")]
    out_dir = paths.af3_refold_filtered_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("*.pdb"):
        existing.unlink()
    linked: List[str] = []
    for row in filtered:
        desc = row.get("description")
        if not desc:
            continue
        src = pdb_dir / f"{desc}.pdb"
        if not src.exists():
            log(f"[pipeline] WARNING: Missing AF3 refold PDB for {desc}")
            continue
        dst = out_dir / f"{desc}.pdb"
        safe_symlink(src.resolve(), dst)
        linked.append(desc)
    log(f"Linked {len(linked)} AF3 refold-filtered complexes into {out_dir}")
    return linked


def run_dockq_evaluation(
    repo_root: Path, cfg: Dict, paths: DerivedPaths, *, dry_run: bool
):
    dockq_cfg = cfg.get("dockq", {})
    if not dockq_cfg.get("enabled", True):
        log("DockQ evaluation disabled in config")
        return
    dockq_env = dockq_cfg.get("conda_env")
    binary = dockq_cfg.get("binary", "DockQ")
    if dockq_env:
        binary_path = str(Path(binary).resolve()) if Path(binary).exists() else binary
    else:
        binary_path = shutil.which(binary)
        if not binary_path:
            candidate = Path(binary)
            if candidate.exists():
                binary_path = str(candidate.resolve())
        if not binary_path:
            raise PipelineError(
                f"DockQ binary '{binary}' not found. Add it to PATH or configure dockq.binary."
            )
    ensure_dir(paths.af3_refold_filtered_dir, "AF3 refold-filtered structures")
    ensure_dir(paths.rosetta_filtered_dir, "Rosetta-filtered references")
    parse_script = repo_root / "demo_scripts" / "parse_dockq_scores.py"
    ensure_file(parse_script, "DockQ summary parser")
    models = sorted(paths.af3_refold_filtered_dir.glob("*.pdb"))
    if not models:
        log("No AF3 refold-filtered models available for DockQ. Skipping evaluation.")
        return
    if dry_run:
        log(
            f"Would run DockQ on {len(models)} model/reference pairs using {binary_path}"
        )
    else:
        paths.dockq_results_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for model in models:
        reference = paths.rosetta_filtered_dir / model.name
        if not reference.exists():
            log(f"[pipeline] WARNING: Missing Rosetta reference for {model.name}")
            continue
        subdir = paths.dockq_results_dir / model.stem
        if not dry_run:
            subdir.mkdir(parents=True, exist_ok=True)
        out_file = subdir / f"{model.stem}_dockq_score"
        if out_file.exists():
            log(f"DockQ output already exists for {model.stem}, skipping")
            continue
        if dry_run:
            processed += 1
            continue
        dockq_cmd = [
            binary_path,
            "--allowed_mismatches",
            "10",
            str(model),
            str(reference),
            "--short",
        ]
        if dockq_env:
            dockq_cmd = conda_cmd(dockq_env, dockq_cmd)
        with out_file.open("w", encoding="utf-8") as handle:
            run_command(dockq_cmd, dry_run=False, stdout=handle)
        processed += 1
    if dry_run:
        log(f"Would parse DockQ outputs after {processed} planned runs")
    else:
        log(f"DockQ finished for {processed} model/reference pairs")
    if dockq_env:
        cmd_parse = conda_cmd(
            dockq_env,
            [
                "python",
                str(parse_script),
                str(paths.dockq_results_dir),
            ],
        )
    else:
        cmd_parse = [
            sys.executable,
            str(parse_script),
            str(paths.dockq_results_dir),
        ]
    run_command(cmd_parse, cwd=repo_root, dry_run=dry_run)
    filter_dockq_results(paths, dockq_cfg, dry_run=dry_run)


def run_dockq_step(repo_root: Path, cfg: Dict, paths: DerivedPaths, *, dry_run: bool):
    wrapper = {"dockq": cfg}
    run_dockq_evaluation(repo_root, wrapper, paths, dry_run=dry_run)


def filter_dockq_results(paths: DerivedPaths, cfg: Dict, *, dry_run: bool):
    summary = paths.dockq_summary_csv
    if not summary.exists():
        log(f"No DockQ summary found at {summary}; skipping DockQ filtering")
        return
    dockq_min = float(cfg.get("dockq_min", 0.5))
    best_scores: Dict[str, float] = {}
    with summary.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            folder = row.get("FolderName")
            try:
                score = float(row.get("Overall_Avg_DockQ", "nan"))
            except (TypeError, ValueError):
                continue
            if not folder:
                continue
            current = best_scores.get(folder, float("-inf"))
            if score > current:
                best_scores[folder] = score
    passing = {name: score for name, score in best_scores.items() if score >= dockq_min}
    log(f"{len(passing)} models pass DockQ >= {dockq_min}")
    if dry_run:
        return
    paths.final_hits_dir.mkdir(parents=True, exist_ok=True)
    for existing in paths.final_hits_dir.glob("*.pdb"):
        existing.unlink()
    with paths.final_hits_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FolderName", "Overall_Avg_DockQ"])
        for name, score in sorted(passing.items()):
            writer.writerow([name, f"{score:.4f}"])
            src = paths.af3_refold_filtered_dir / f"{name}.pdb"
            if not src.exists():
                log(f"[pipeline] WARNING: Missing AF3 refold PDB for {name}")
                continue
            dst = paths.final_hits_dir / f"{name}.pdb"
            safe_symlink(src.resolve(), dst)
    log(f"Final DockQ-filtered models written to {paths.final_hits_dir}")


def run_af3_refold(
    repo_root: Path,
    cfg: Dict,
    inputs: InputConfig,
    paths: DerivedPaths,
    rosetta_cfg: Dict,
    *,
    dry_run: bool,
):
    ensure_dir(paths.rosetta_filtered_dir, "Rosetta-filtered PDB directory")
    rosetta_filtered = sorted(paths.rosetta_filtered_dir.glob("*.pdb"))
    if not rosetta_filtered:
        raise PipelineError(
            f"No Rosetta-filtered complexes found in {paths.rosetta_filtered_dir}"
        )
    base_json_value = cfg.get("base_json")
    if base_json_value:
        base_json = Path(base_json_value)
    else:
        base_json = paths.af3_refold_base_out / "base_json" / "base.json"
    screen_script_value = cfg.get("peptide_screen_script")
    if screen_script_value:
        screen_script = Path(screen_script_value)
    else:
        screen_script = repo_root / "myscripts" / "peptide_variant_screen.py"
    ensure_file(screen_script, "peptide_variant_screen.py")
    if not dry_run:
        if paths.af3_refold_root.exists():
            shutil.rmtree(paths.af3_refold_root)
        if paths.dockq_root.exists():
            shutil.rmtree(paths.dockq_root)
        if paths.final_hits_dir.exists():
            shutil.rmtree(paths.final_hits_dir)
        paths.af3_refold_base_out.mkdir(parents=True, exist_ok=True)
        paths.af3_refold_pdb_models.mkdir(parents=True, exist_ok=True)
        paths.af3_refold_filtered_dir.mkdir(parents=True, exist_ok=True)
        paths.dockq_results_dir.mkdir(parents=True, exist_ok=True)
        paths.final_hits_dir.mkdir(parents=True, exist_ok=True)
    receptor_chain = str(rosetta_cfg.get("receptor_chain", inputs.receptor_chain))
    if not base_json_value:
        if dry_run:
            log("Would generate base AF3 JSON and receptor MSA from Rosetta-filtered PDBs")
        else:
            base_root = paths.af3_refold_base_out / "base_json"
            if base_root.exists():
                shutil.rmtree(base_root)
            base_root.mkdir(parents=True, exist_ok=True)
            receptor_fasta = base_root / "receptor.fasta"
            receptor_msa_dir = base_root / "receptor_msa"
            sequence = extract_chain_sequence(rosetta_filtered[0], receptor_chain)
            receptor_fasta.write_text(
                f">receptor_{receptor_chain}\n{sequence}\n", encoding="utf-8"
            )
            cmd_msa = conda_cmd(
                cfg.get("colabfold_env", "colabfold"),
                [
                    "colabfold_batch",
                    str(receptor_fasta),
                    str(receptor_msa_dir),
                    "--msa-only",
                ],
            )
            run_command(cmd_msa, cwd=repo_root, dry_run=False)
            msa_files = sorted(receptor_msa_dir.glob("*.a3m"))
            if not msa_files:
                raise PipelineError(f"No MSA files found in {receptor_msa_dir}")
            ligand_chain = str(rosetta_cfg.get("ligand_chain", inputs.binder_chain))
            ligand_sequence = extract_chain_sequence(rosetta_filtered[0], ligand_chain)
            base_json = base_root / "base.json"
            num_seeds = int(cfg.get("num_seeds", 1))
            seed_start = int(cfg.get("seed_start", 10))
            write_af3_base_json(
                base_json,
                name=rosetta_filtered[0].stem,
                receptor_chain=receptor_chain,
                receptor_sequence=sequence,
                receptor_msa_path=msa_files[0].resolve(),
                ligand_chain=ligand_chain,
                ligand_sequence=ligand_sequence,
                model_seeds=cfg.get(
                    "model_seeds", list(range(seed_start, seed_start + num_seeds))
                ),
            )
    ensure_file(base_json, "base AF3 JSON")
    ligand_chain = str(rosetta_cfg.get("ligand_chain", inputs.binder_chain))
    peptide_fasta = paths.af3_refold_base_out / "peptide_variants.fa"
    if dry_run:
        log(f"Would write peptide FASTA to {peptide_fasta}")
    else:
        with peptide_fasta.open("w", encoding="utf-8") as handle:
            for pdb in rosetta_filtered:
                sequence = extract_chain_sequence(pdb, ligand_chain)
                handle.write(f">{pdb.stem}\n{sequence}\n")
    model_dir = Path(cfg["model_dir"])
    ensure_dir(model_dir, "AF3 model dir")
    cmd_screen = conda_cmd(
        cfg["conda_env"],
        [
            "python",
            str(screen_script),
            "--json_path",
            str(base_json),
            "--peptide_chain_id",
            ligand_chain,
            "--peptide_sequences",
            str(peptide_fasta),
            "--output_dir",
            str(paths.af3_refold_base_out),
            "--model_dir",
            str(model_dir),
            "--num_diffusion_samples",
            str(cfg.get("num_diffusion_samples", 1)),
            "--num_recycles",
            str(cfg.get("num_recycles", 3)),
            "--receptor_chain_id",
            receptor_chain,
        ],
    )
    run_command(cmd_screen, cwd=repo_root, dry_run=dry_run)
    if dry_run:
        log("Would filter AF3 refold outputs and evaluate DockQ")
        return
    summary_path = paths.af3_refold_base_out / "screening_best_summary.csv"
    ensure_file(summary_path, "AF3 refold summary CSV")
    filter_cfg = cfg.get("filter", {})
    iptm_min = float(filter_cfg.get("iptm_min", 0.7))
    plddt_min = float(filter_cfg.get("plddt_min", 70.0))
    filtered_rows: List[Dict[str, str]] = []
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                iptm = float(row.get("iptm_peptide_vs_receptor", "nan"))
                plddt = float(row.get("plddt_peptide_mean", "nan"))
            except (TypeError, ValueError):
                continue
            if iptm >= iptm_min and plddt >= plddt_min:
                filtered_rows.append(row)
    log(
        f"{len(filtered_rows)} designs pass AF3 refold thresholds iptm>= {iptm_min} & plddt>= {plddt_min}"
    )
    write_summary_csv(
        paths.af3_refold_filtered_summary,
        list(filtered_rows[0].keys()) if filtered_rows else [],
        filtered_rows,
        dry_run=False,
    )
    if not filtered_rows:
        log("No AF3 refold entries passed the thresholds; skipping DockQ.")
        return
    for existing in paths.af3_refold_pdb_models.glob("*.pdb"):
        existing.unlink()
    for existing in paths.af3_refold_filtered_dir.glob("*.pdb"):
        existing.unlink()
    linked: List[str] = []
    for row in filtered_rows:
        name = row.get("name") or ""
        variant_dir = row.get("variant_dir")
        if not name or not variant_dir:
            continue
        seed_dir = Path(variant_dir)
        cif_files = sorted(seed_dir.rglob("*.cif"))
        if not cif_files:
            log(f"[pipeline] WARNING: No CIF outputs found for {name} in {seed_dir}")
            continue
        dst_pdb = paths.af3_refold_pdb_models / f"{name}.pdb"
        convert_cif_to_pdb(cif_files[0], dst_pdb)
        dst_filtered = paths.af3_refold_filtered_dir / f"{name}.pdb"
        safe_symlink(dst_pdb.resolve(), dst_filtered)
        linked.append(name)
    log(f"Linked {len(linked)} AF3 refold-filtered complexes into {paths.af3_refold_filtered_dir}")
    run_dockq_evaluation(repo_root, cfg, paths, dry_run=False)
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
    if "dockq" in step_settings:
        step_settings["dockq"] = cfg.get("dockq", cfg.get("af3_refold", {}).get("dockq", {}))
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
        "rosetta_relax": lambda: run_rosetta_relax(
            repo_root, step_settings["rosetta_relax"], inputs, paths, dry_run=dry_run
        ),
        "af3_refold": lambda: run_af3_refold(
            repo_root,
            step_settings["af3_refold"],
            inputs,
            paths,
            step_settings["rosetta_relax"],
            dry_run=dry_run,
        ),
        "dockq": lambda: run_dockq_step(
            repo_root,
            step_settings["dockq"],
            paths,
            dry_run=dry_run,
        ),
    }
    for step in STEP_NAMES:
        settings = step_settings[step]
        enabled = settings.get("enabled")
        if enabled is None:
            enabled = step in DEFAULT_ENABLED_STEPS
        env = settings.get("conda_env")
        if not enabled:
            log(f"Skipping {step}: disabled in config")
            continue
        if requested_steps and step not in requested_steps:
            log(f"Skipping {step}: not requested")
            continue
        if enabled and step not in ("prep") and not env:
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
