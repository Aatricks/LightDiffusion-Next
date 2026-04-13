# Prompt Attention Caching

### What It Does

Caches CLIP text embeddings for prompts you've already encoded. When you reuse a prompt (or parts of it), the embedding is retrieved from cache instead of being recomputed.

### When It Helps Most

- Batch generation with same prompt
- Testing different seeds
- Incremental prompt refinement
- Generation sessions with repeated themes

### Configuration

**Enable/Disable** (default: enabled):
```python
from src.Utilities import prompt_cache

# Enable (default)
prompt_cache.enable_prompt_cache(True)

# Disable
prompt_cache.enable_prompt_cache(False)

# Check status
stats = prompt_cache.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

**Cache Settings**:
- Maximum entries: 256 prompts before pruning
- Cache structure: global dict keyed by prompt hash and CLIP identity
- Memory usage: workload-dependent, estimated from cached embedding tensors
- Cache cleared on: restart, disable, or manual clear
- Automatic pruning: removes the oldest 25% of entries when the cache exceeds its limit

### Viewing Cache Stats

```python
from src.Utilities import prompt_cache

# Print statistics
prompt_cache.print_cache_stats()

# Output:
# ============================================================
# Prompt Cache Statistics
# ============================================================
#   Status: Enabled
#   Entries: 42
#   Size: ~85.3 MB
#   Requests: 150 (hits: 108, misses: 42)
#   Hit Rate: 72.0%
# ============================================================
```

### Best Practices

1. **Leave it enabled** - negligible overhead, significant gains
2. **Monitor hit rate** - should be >50% in typical workflows
3. **Clear cache** when switching models or major prompt changes
4. **Batch similar prompts** to maximize cache hits
5. **Expect global behavior** because the cache is shared across repeated prompt encodes rather than being scoped to a single generation session
