"""Performance benchmark comparing LightDiffusion-Next across SD1.5, SDXL, and Flux2 Klein 4B.

This script measures:
- Sampling time (model inference + denoising loop)
- Total generation time (includes VAE decode, model load)
- VRAM usage peak

Usage:
    cd LightDiffusion-Next
    python tests/benchmark_performance.py
"""

import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class BenchmarkResult:
    model_type: str
    model_name: str
    resolution: tuple[int, int]
    steps: int
    sampler: str
    scheduler: str
    warmup_time_s: float = 0.0
    generation_time_s: float = 0.0
    sampling_time_s: float = 0.0
    vae_decode_time_s: float = 0.0
    peak_vram_mb: float = 0.0
    batch_size: int = 1
    cfg_scale: float = 7.0
    notes: str = ""


@dataclass
class BenchmarkSuite:
    results: list[BenchmarkResult] = field(default_factory=list)
    system_info: dict = field(default_factory=dict)
    
    def add(self, result: BenchmarkResult):
        self.results.append(result)
    
    def to_json(self, path: str):
        data = {
            "system_info": self.system_info,
            "results": [asdict(r) for r in self.results]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


def get_system_info() -> dict:
    """Collect system information for benchmark context."""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    return info


def reset_cuda():
    """Clear CUDA cache for accurate VRAM measurement."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


def get_peak_vram_mb() -> float:
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def benchmark_sd15(suite: BenchmarkSuite, warmup: bool = True):
    """Benchmark SD1.5 with DreamShaper 8."""
    from src.user.pipeline import pipeline
    
    model_path = "./include/checkpoints/DreamShaper_8_pruned.safetensors"
    if not Path(model_path).exists():
        print(f"  Skipping SD1.5 - model not found: {model_path}")
        return
    
    prompt = "a beautiful sunset over a mountain landscape, high quality photograph"
    w, h = 512, 512
    steps = 20
    sampler = "euler"
    scheduler = "normal"
    
    print(f"\n{'='*60}")
    print(f"SD1.5 Benchmark: {w}x{h}, {steps} steps, {sampler}/{scheduler}")
    print(f"{'='*60}")
    
    # Warmup run
    warmup_time = 0.0
    if warmup:
        print("  Warmup run...")
        reset_cuda()
        t0 = time.perf_counter()
        pipeline(
            prompt=prompt,
            w=w, h=h,
            steps=5,
            sampler=sampler,
            scheduler=scheduler,
            model_path=model_path,
            hires_fix=False,
            adetailer=False,
            autohdr=False,
            enable_multiscale=False,
        )
        warmup_time = time.perf_counter() - t0
        print(f"  Warmup done in {warmup_time:.2f}s")
    
    # Timed run
    print("  Benchmark run...")
    reset_cuda()
    t0 = time.perf_counter()
    pipeline(
        prompt=prompt,
        w=w, h=h,
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        model_path=model_path,
        hires_fix=False,
        adetailer=False,
        autohdr=False,
        enable_multiscale=False,
    )
    gen_time = time.perf_counter() - t0
    peak_vram = get_peak_vram_mb()
    
    result = BenchmarkResult(
        model_type="SD1.5",
        model_name="DreamShaper_8",
        resolution=(w, h),
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        warmup_time_s=warmup_time,
        generation_time_s=gen_time,
        peak_vram_mb=peak_vram,
    )
    suite.add(result)
    
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Steps/second: {steps / gen_time:.2f}")
    print(f"  Peak VRAM: {peak_vram:.0f} MB")


def benchmark_sdxl(suite: BenchmarkSuite, warmup: bool = True):
    """Benchmark SDXL with Juggernaut-XL v9."""
    from src.user.pipeline import pipeline
    
    model_path = "./include/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    if not Path(model_path).exists():
        print(f"  Skipping SDXL - model not found: {model_path}")
        return
    
    prompt = "a beautiful sunset over a mountain landscape, high quality photograph"
    w, h = 1024, 1024
    steps = 20
    sampler = "euler"
    scheduler = "ays"  # AYS is commonly used for SDXL
    
    print(f"\n{'='*60}")
    print(f"SDXL Benchmark: {w}x{h}, {steps} steps, {sampler}/{scheduler}")
    print(f"{'='*60}")
    
    # Warmup run
    warmup_time = 0.0
    if warmup:
        print("  Warmup run...")
        reset_cuda()
        t0 = time.perf_counter()
        pipeline(
            prompt=prompt,
            w=w, h=h,
            steps=5,
            sampler=sampler,
            scheduler=scheduler,
            model_path=model_path,
            hires_fix=False,
            adetailer=False,
            autohdr=False,
            enable_multiscale=False,
        )
        warmup_time = time.perf_counter() - t0
        print(f"  Warmup done in {warmup_time:.2f}s")
    
    # Timed run
    print("  Benchmark run...")
    reset_cuda()
    t0 = time.perf_counter()
    pipeline(
        prompt=prompt,
        w=w, h=h,
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        model_path=model_path,
        hires_fix=False,
        adetailer=False,
        autohdr=False,
        enable_multiscale=False,
    )
    gen_time = time.perf_counter() - t0
    peak_vram = get_peak_vram_mb()
    
    result = BenchmarkResult(
        model_type="SDXL",
        model_name="Juggernaut-XL_v9",
        resolution=(w, h),
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        warmup_time_s=warmup_time,
        generation_time_s=gen_time,
        peak_vram_mb=peak_vram,
    )
    suite.add(result)
    
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Steps/second: {steps / gen_time:.2f}")
    print(f"  Peak VRAM: {peak_vram:.0f} MB")


def benchmark_flux2_klein(suite: BenchmarkSuite, warmup: bool = True):
    """Benchmark Flux2 Klein 4B."""
    from src.user.pipeline import pipeline
    
    model_path = "__FLUX2_KLEIN__"  # Special marker for Flux2 Klein
    diffusion_path = "./include/diffusion_model/flux-2-klein-4b.safetensors"
    
    if not Path(diffusion_path).exists():
        print(f"  Skipping Flux2 Klein - model not found: {diffusion_path}")
        return
    
    prompt = "a beautiful sunset over a mountain landscape"
    w, h = 1024, 1024
    steps = 4  # Flux2 Klein is distilled, uses fewer steps
    sampler = "euler"
    scheduler = "simple"
    cfg = 1.0  # Distilled models use CFG=1
    
    print(f"\n{'='*60}")
    print(f"Flux2 Klein Benchmark: {w}x{h}, {steps} steps, {sampler}/{scheduler}")
    print(f"{'='*60}")
    
    # Warmup run
    warmup_time = 0.0
    if warmup:
        print("  Warmup run...")
        reset_cuda()
        t0 = time.perf_counter()
        pipeline(
            prompt=prompt,
            w=w, h=h,
            steps=2,
            sampler=sampler,
            scheduler=scheduler,
            model_path=model_path,
            cfg_scale=cfg,
            hires_fix=False,
            adetailer=False,
            autohdr=False,
            enable_multiscale=False,
        )
        warmup_time = time.perf_counter() - t0
        print(f"  Warmup done in {warmup_time:.2f}s")
    
    # Timed run
    print("  Benchmark run...")
    reset_cuda()
    t0 = time.perf_counter()
    pipeline(
        prompt=prompt,
        w=w, h=h,
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        model_path=model_path,
        cfg_scale=cfg,
        hires_fix=False,
        adetailer=False,
        autohdr=False,
        enable_multiscale=False,
    )
    gen_time = time.perf_counter() - t0
    peak_vram = get_peak_vram_mb()
    
    result = BenchmarkResult(
        model_type="Flux2",
        model_name="flux-2-klein-4b",
        resolution=(w, h),
        steps=steps,
        sampler=sampler,
        scheduler=scheduler,
        warmup_time_s=warmup_time,
        generation_time_s=gen_time,
        peak_vram_mb=peak_vram,
        cfg_scale=cfg,
    )
    suite.add(result)
    
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Steps/second: {steps / gen_time:.2f}")
    print(f"  Peak VRAM: {peak_vram:.0f} MB")


def main():
    print("="*60)
    print("LightDiffusion-Next Performance Benchmark")
    print("="*60)
    
    # Ensure output directory
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./tests", exist_ok=True)
    
    suite = BenchmarkSuite(system_info=get_system_info())
    
    print("\nSystem Info:")
    for k, v in suite.system_info.items():
        print(f"  {k}: {v}")
    
    # Run benchmarks
    benchmark_sd15(suite, warmup=True)
    benchmark_sdxl(suite, warmup=True)
    benchmark_flux2_klein(suite, warmup=True)
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Resolution':<12} {'Steps':<6} {'Time (s)':<10} {'Steps/s':<10} {'VRAM (MB)':<10}")
    print("-" * 80)
    for r in suite.results:
        steps_per_s = r.steps / r.generation_time_s if r.generation_time_s > 0 else 0
        print(f"{r.model_type:<20} {f'{r.resolution[0]}x{r.resolution[1]}':<12} {r.steps:<6} {r.generation_time_s:<10.2f} {steps_per_s:<10.2f} {r.peak_vram_mb:<10.0f}")
    
    # Save results
    suite.to_json("./tests/benchmark_results.json")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
