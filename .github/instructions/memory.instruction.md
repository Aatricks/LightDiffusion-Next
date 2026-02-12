---
applyTo: '**'
---

# User Memory

## User Preferences
- Programming languages: 
- Code style preferences: 
- Development environment: 
- Communication style: 

## Project Context
- Current project type: 
- Tech stack: 
- Architecture patterns: 
- Key requirements: 

## Coding Patterns
- Preferred patterns and practices
- Code organization preferences
- Testing approaches
- Documentation style

## Context7 Research History
- Libraries researched on Context7
- Best practices discovered
- Implementation patterns used
- Version-specific findings

- 2026-02-11: Searched Context7 for pytest; no libraries found. Reviewed Context7 MCP docs (all-clients, adding-libraries, troubleshooting, api-guide, developer guide) to satisfy research requirements for this task.

## Conversation History
- 2026-02-11: Requested DifferentialDiffusion class excerpt with line numbers from src/AutoDetailer/ADetailer.py.
- 2026-02-11: Fixing ADetailer SDXL mask behavior by applying denoise_mask blending in KSamplerX0Inpaint; will add tests and validate with manual image generation.
- 2026-02-11: Added denoise_mask resizing to latent resolution to avoid shape mismatch; generated SDXL baseline and ADetailer outputs for manual verification.
- 2026-02-11: Normalized ADetailer noise masks to [0,1], aligned SDXL crop conditioning to crop-local sizes, and added unit tests plus manual SDXL ADetailer generation and image stats verification.
- 2026-02-11: Began implementation of mask-aware regression test for ADetailer SDXL noise masking.
- 2026-02-11: Added deterministic unit test that stubs sampling and verifies noise is localized to resized mask region in enhance_detail.
- Important decisions made
- Recurring questions or topics
- Solutions that worked well
- Things to avoid or that didn't work

## Notes
- 2026-02-11: pytest -q tests/unit/test_adetailer_noise_mask.py passed (4 tests).
