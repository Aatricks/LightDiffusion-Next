import pytest
pytest.skip("moved to tests/e2e/test_core_functionalities.py (refactored)", allow_module_level=True)

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.user.pipeline import pipeline

def get_absolute_path(relative_path):
    return os.path.join(project_root, relative_path)

def run_test(test_function, *args, **kwargs):
    """Decorator to time and print test information."""
    print(f"\n--- Running test: {test_function.__name__} ---")
    start_time = time.perf_counter()
    try:
        test_function(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"--- Test {test_function.__name__} finished in {end_time - start_time:.2f} seconds ---")
    except Exception as e:
        end_time = time.perf_counter()
        print(f"--- Test {test_function.__name__} failed after {end_time - start_time:.2f} seconds ---")
        print(f"Error: {e}")


@pytest.mark.slow
def test_normal_pipeline():
    """Tests the default text-to-image pipeline."""
    print("Testing normal pipeline with default settings...")
    pipeline(
        prompt="a beautiful landscape",
        w=512,
        h=512,
        number=1,
        batch=1,
    )

@pytest.mark.slow
def test_samplers():
    """Tests all available samplers."""
    samplers = ["euler", "euler_ancestral", "euler_cfgpp", "euler_ancestral_cfgpp", "dpmpp_2m_cfgpp", "dpmpp_sde_cfgpp"]
    for sampler in samplers:
        print(f"Testing sampler: {sampler}...")
        pipeline(
            prompt=f"a beautiful landscape using {sampler} sampler",
            w=512,
            h=512,
            number=1,
            batch=1,
            sampler=sampler,
        )

@pytest.mark.slow
def test_schedulers():
    """Tests all available schedulers."""
    schedulers = ["normal", "karras", "simple", "beta", "ays", "ays_sd15", "ays_sdxl"]
    for scheduler in schedulers:
        print(f"Testing scheduler: {scheduler}...")
        pipeline(
            prompt=f"a beautiful landscape using {scheduler} scheduler",
            w=512,
            h=512,
            number=1,
            batch=1,
            scheduler=scheduler,
        )

@pytest.mark.slow
def test_optimizations():
    """Tests various optimizations."""
    import time
    
    def time_pipeline(description, **kwargs):
        """Helper to time a pipeline call."""
        print(f"Testing {description}...")
        start_time = time.perf_counter()
        pipeline(
            prompt="a beautiful landscape",
            w=512,
            h=512,
            number=1,
            batch=1,
            **kwargs
        )
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"{duration:.2f}")
        return duration
    
    # Baseline: no optimizations
    baseline_time = time_pipeline("baseline (no optimizations)")
    
    # Test Stable-Fast
    stable_fast_time = time_pipeline("Stable-Fast optimization", stable_fast=True)
    
    # Test multiscale diffusion
    multiscale_time = time_pipeline("multiscale diffusion", enable_multiscale=True, multiscale_preset="performance")
    
    # Test DeepCache
    deepcache_time = time_pipeline("DeepCache", deepcache_enabled=True)
    
    # Speed comparisons
    print("\n--- Speed Comparison Results ---")
    print(f"Baseline time: {baseline_time:.2f}s")
    if stable_fast_time < baseline_time:
        speedup = baseline_time / stable_fast_time
        print(f"Stable-Fast: {stable_fast_time:.2f}s ({speedup:.2f}x speedup)")
    else:
        slowdown = stable_fast_time / baseline_time
        print(f"Stable-Fast: {stable_fast_time:.2f}s ({slowdown:.2f}x slower)")
    
    if multiscale_time < baseline_time:
        speedup = baseline_time / multiscale_time
        print(f"Multiscale: {multiscale_time:.2f}s ({speedup:.2f}x speedup)")
    else:
        slowdown = multiscale_time / baseline_time
        print(f"Multiscale: {multiscale_time:.2f}s ({slowdown:.2f}x slower)")
    
    if deepcache_time < baseline_time:
        speedup = baseline_time / deepcache_time
        print(f"DeepCache: {deepcache_time:.2f}s ({speedup:.2f}x speedup)")
    else:
        slowdown = deepcache_time / baseline_time
        print(f"DeepCache: {deepcache_time:.2f}s ({slowdown:.2f}x slower)")


@pytest.mark.slow
def test_img2img():
    """Tests the img2img pipeline."""
    print("Testing img2img pipeline...")
    # Create a dummy image for testing
    dummy_image_path = get_absolute_path("tests/dummy_image.png")
    if not os.path.exists(dummy_image_path):
        from PIL import Image
        img = Image.new('RGB', (256, 256), color = 'red')
        img.save(dummy_image_path)

    pipeline(
        prompt="a beautiful landscape",
        w=512,
        h=512,
        number=1,
        batch=1,
        img2img=True,
        img2img_image=dummy_image_path,
    )

@pytest.mark.slow
def test_api_endpoints():
    """Tests the API endpoints."""
    print("Testing API endpoints...")
    server_process = None
    try:
        # Start the server
        server_command = [sys.executable, get_absolute_path("server.py")]
        server_process = subprocess.Popen(server_command)
        print("Waiting for server to start...")
        time.sleep(30) # Wait for the server to initialize

        # Test health endpoint
        print("Testing /health endpoint...")
        response = requests.get("http://localhost:7861/health")
        response.raise_for_status()
        print("Health check passed.")

        # Test generate endpoint
        print("Testing /api/generate endpoint...")
        payload = {
            "prompt": "a beautiful landscape",
            "width": 512,
            "height": 512,
            "steps": 1,
        }
        response = requests.post("http://localhost:7861/api/generate", json=payload)
        response.raise_for_status()
        print("Generate endpoint test passed.")


    finally:
        if server_process:
            print("Shutting down server...")
            server_process.terminate()
            server_process.wait()

@pytest.mark.slow
def test_hires_fix():
    """Tests the hires_fix feature."""
    print("Testing hires_fix...")
    pipeline(
        prompt="a beautiful landscape with hires_fix",
        w=512,
        h=512,
        number=1,
        batch=1,
        hires_fix=True,
    )

@pytest.mark.slow
def test_adetailer():
    """Tests the adetailer feature."""
    print("Testing adetailer...")
    pipeline(
        prompt="a beautiful landscape with adetailer",
        w=512,
        h=512,
        number=1,
        batch=1,
        adetailer=True,
    )

@pytest.mark.slow
def test_enhance_prompt():
    """Tests the enhance_prompt feature."""
    print("Testing enhance_prompt...")
    pipeline(
        prompt="a beautiful landscape with enhance_prompt",
        w=512,
        h=512,
        number=1,
        batch=1,
        enhance_prompt=True,
    )

@pytest.mark.slow
def test_reuse_seed():
    """Tests the reuse_seed feature."""
    print("Testing reuse_seed...")
    pipeline(
        prompt="a beautiful landscape with reuse_seed",
        w=512,
        h=512,
        number=1,
        batch=1,
        reuse_seed=True,
    )

@pytest.mark.slow
def test_realistic_model():
    """Tests the realistic_model feature."""
    print("Testing realistic_model...")
    pipeline(
        prompt="a beautiful landscape with realistic_model",
        w=512,
        h=512,
        number=1,
        batch=1,
        realistic_model=True,
    )

@pytest.mark.slow
def test_all_features():
    """Tests all features at once."""
    print("Testing all features at once...")
    pipeline(
        prompt="a beautiful landscape with all features",
        w=512,
        h=512,
        number=1,
        batch=1,
        hires_fix=True,
        adetailer=True,
        enhance_prompt=True,
        reuse_seed=True,
        realistic_model=True,
    )

@pytest.mark.slow
def benchmark_optimizations():
    """Benchmarks optimizations with multiple runs for statistical analysis."""
    import time
    import statistics
    
    def benchmark_pipeline(description, runs=3, **kwargs):
        """Benchmark a pipeline configuration with multiple runs."""
        times = []
        for i in range(runs):
            print(f"Benchmarking {description} - run {i+1}/{runs}...")
            start_time = time.perf_counter()
            pipeline(
                prompt="a beautiful landscape",
                w=512,
                h=512,
                number=1,
                batch=1,
                **kwargs
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            times.append(duration)
            print(f"{duration:.2f}")
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        print(f"{avg_time:.2f}")
        return avg_time, std_dev
    
    print("=== Optimization Benchmark (3 runs each) ===")
    
    # Baseline
    baseline_avg, baseline_std = benchmark_pipeline("Baseline (no optimizations)")
    
    # Optimizations
    optimizations = [
        ("Stable-Fast", {"stable_fast": True}),
        ("Multiscale Performance", {"enable_multiscale": True, "multiscale_preset": "performance"}),
        ("DeepCache", {"deepcache_enabled": True}),
    ]
    
    results = []
    for name, kwargs in optimizations:
        avg, std = benchmark_pipeline(name, **kwargs)
        results.append((name, avg, std))
    
    # Results summary
    print("\n=== Benchmark Results Summary ===")
    print(f"{'Optimization':<25} {'Avg Time (s)':<15} {'Std Dev':<10} {'Speedup':<10}")
    print("-" * 60)
    print(f"{'Baseline':<25} {baseline_avg:<15.2f} {baseline_std:<10.2f} {'1.00x':<10}")
    
    for name, avg, std in results:
        speedup = baseline_avg / avg
        print(f"{name:<25} {avg:<15.2f} {std:<10.2f} {speedup:<10.2f}x")
    
    print("\nNote: Lower time = better performance. Speedup > 1 means faster than baseline.")

def main():
    parser = argparse.ArgumentParser(description="Run LightDiffusion-Next core functionality tests.")
    parser.add_argument("--all", action="store_true", help="Run all tests.")
    parser.add_argument("--normal", action="store_true", help="Run normal pipeline test.")
    parser.add_argument("--samplers", action="store_true", help="Run samplers test.")
    parser.add_argument("--schedulers", action="store_true", help="Run schedulers test.")
    parser.add_argument("--optimizations", action="store_true", help="Run optimizations test.")
    parser.add_argument("--img2img", action="store_true", help="Run img2img test.")
    parser.add_argument("--api", action="store_true", help="Run API endpoints test.")
    parser.add_argument("--hires_fix", action="store_true", help="Run hires_fix test.")
    parser.add_argument("--adetailer", action="store_true", help="Run adetailer test.")
    parser.add_argument("--enhance_prompt", action="store_true", help="Run enhance_prompt test.")
    parser.add_argument("--reuse_seed", action="store_true", help="Run reuse_seed test.")
    parser.add_argument("--realistic_model", action="store_true", help="Run realistic_model test.")
    parser.add_argument("--all_features", action="store_true", help="Run all features test.")
    parser.add_argument("--benchmark", action="store_true", help="Run optimization benchmark.")


    args = parser.parse_args()

    if args.all or args.normal:
        run_test(test_normal_pipeline)
    if args.all or args.samplers:
        run_test(test_samplers)
    if args.all or args.schedulers:
        run_test(test_schedulers)
    if args.all or args.optimizations:
        run_test(test_optimizations)
    if args.all or args.img2img:
        run_test(test_img2img)
    if args.all or args.api:
        run_test(test_api_endpoints)
    if args.all or args.hires_fix:
        run_test(test_hires_fix)
    if args.all or args.adetailer:
        run_test(test_adetailer)
    if args.all or args.enhance_prompt:
        run_test(test_enhance_prompt)
    if args.all or args.reuse_seed:
        run_test(test_reuse_seed)
    if args.all or args.realistic_model:
        run_test(test_realistic_model)
    if args.all or args.all_features:
        run_test(test_all_features)
    if args.benchmark:
        run_test(benchmark_optimizations)

    if not any(vars(args).values()):
        print("No tests selected. Use --all to run all tests or select specific tests.")

if __name__ == "__main__":
    main()
