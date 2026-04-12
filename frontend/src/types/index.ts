export interface GenerationSettings {
    prompt: string;
    negative_prompt: string;
    width: number;
    height: number;
    num_images: number;
    batch_size: number;
    steps: number;
    cfg_scale: number;
    seed?: number;
    scheduler: string;
    sampler: string;
    model_path: string;
    refiner_model_path?: string;
    refiner_switch_step?: number;
    // Img2Img
    img2img_mode: boolean;
    img2img_image?: string; // base64
    img2img_denoise: number;
    input_image_path?: string; // Legacy/Local path

    // Toggles
    hiresfix: boolean;
    adetailer: boolean;
    enhance_prompt: boolean;
    stable_fast: boolean;
    reuse_seed: boolean;
    keep_models_loaded: boolean;
    enable_preview: boolean;
    // Preview fidelity for in-progress previews (low|balanced|high)
    preview_fidelity?: 'low' | 'balanced' | 'high';
    // Persist prompt/negative_prompt to server history when user.opts in
    persist_prompt_history?: boolean;

    // ControlNet
    controlnet_enabled: boolean;
    controlnet_model?: string;
    controlnet_strength: number;
    controlnet_type: string;

    // Multi-scale
    enable_multiscale: boolean;
    multiscale_preset: string;
    multiscale_factor: number;
    multiscale_fullres_start: number;
    multiscale_fullres_end: number;
    multiscale_intermittent_fullres: boolean;

    // DeepCache
    deepcache_enabled: boolean;
    deepcache_interval: number;
    deepcache_depth: number;

    // Other optimizations
    cfg_free_enabled: boolean;
    cfg_free_start_percent: number;
    tome_enabled: boolean;
    tome_ratio: number;
    
    // Compilation & quantization
    torch_compile: boolean;
    vae_autotune: boolean;
    fp8_inference: boolean;
    weight_quantization?: 'fp8' | 'nvfp4' | 'none' | null;
}

export interface SettingsPreferences {
    torch_compile: boolean;
    vae_autotune: boolean;
}

export interface ImageMetadata {
    prompt?: string;
    negative_prompt?: string;
    seed?: number;
    steps?: number;
    cfg_scale?: number;
    sampler?: string;
    scheduler?: string;
    model_path?: string;
    width?: number;
    height?: number;
    denoise?: number;


    // ControlNet
    controlnet_enabled: boolean;
    controlnet_model?: string;
    controlnet_strength: number;
    controlnet_type: string;

    // Multi-scale
    enable_multiscale: boolean;
    multiscale_preset: string;
    multiscale_factor: number;
    multiscale_fullres_start: number;
    multiscale_fullres_end: number;
    multiscale_intermittent_fullres: boolean;

    // DeepCache
    deepcache_enabled: boolean;
    deepcache_interval: number;
    deepcache_depth: number;

    // Other optimizations
    cfg_free_enabled: boolean;
    cfg_free_start_percent: number;
    tome_enabled: boolean;
    tome_ratio: number;
    
    // Compilation & quantization
    torch_compile?: boolean;
    vae_autotune?: boolean;
    fp8_inference?: boolean;
    weight_quantization?: 'fp8' | 'nvfp4' | 'none' | null;
}

export interface GenerationResponse {
    images?: string[]; // base64
    image?: string;    // single image base64
    info?: string;
}

export interface SettingsSnapshot {
    id: string;
    ts: number; // unix timestamp
    settings: GenerationSettings;
}

export interface ModelCapabilities {
    supports_hires_fix: boolean;
    supports_img2img: boolean;
    supports_controlnet: boolean;
    supports_inpainting: boolean;
    supports_stable_fast: boolean;
    supports_deepcache: boolean;
    supports_tome: boolean;
    preferred_resolution: number;
}

export interface ModelInfo {
    path: string;
    name: string;
    type: string;
    capabilities: ModelCapabilities;
}

export interface PreviewMessage {
    type: "preview" | "progress" | "error" | "generation_start";
    step?: number;
    total_steps?: number;
    images?: string[]; // base64
    message?: string;
    generation_id?: string;
}
