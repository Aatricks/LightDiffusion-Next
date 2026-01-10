import os
import sys
import time
import cProfile
import pstats
import io
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.user.pipeline import pipeline

def run_benchmarked_pipeline(flux=False):
    mode_name = "Flux" if flux else "Standard"
    print(f"🚀 Starting benchmarked {mode_name} generation...")
    
    # Warm-up run
    print("🔥 Warm-up run...")
    pipeline(
        prompt="a high quality photo of a futuristic city",
        w=512,
        h=512,
        number=1,
        batch=1,
        steps=10,
        flux_enabled=flux
    )
    
    print(f"📊 Profiling {mode_name} generation...")
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.perf_counter()
    pipeline(
        prompt="a high quality photo of a futuristic city",
        w=512,
        h=512,
        number=1,
        batch=1,
        steps=20,
        flux_enabled=flux
    )
    end_time = time.perf_counter()
    
    pr.disable()
    
    print(f"✅ {mode_name} Generation finished in {end_time - start_time:.2f}s")
    
    # Analyze profile
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    
    print(f"\n=== TOP 30 {mode_name.upper()} FUNCTIONS BY CUMULATIVE TIME ===")
    print(s.getvalue())
    
    # Also show by internal time
    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)
    
    print(f"\n=== TOP 30 {mode_name.upper()} FUNCTIONS BY INTERNAL TIME ===")
    print(s.getvalue())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux", action="store_true")
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs("./output", exist_ok=True)
    run_benchmarked_pipeline(flux=args.flux)
