import { create } from 'zustand';
import type { GenerationSettings, ModelInfo, PreviewMessage } from '../types';

interface AppState {
    settings: GenerationSettings;
    availableModels: ModelInfo[];
    availableControlNets: string[];
    status: 'idle' | 'generating' | 'error';
    currentImage: string | null; // Latest generated image
    gallery: string[]; // History of generated images
    preview: PreviewMessage | null;
    serverStatus: boolean; // Is server reachable?

    setSettings: (settings: Partial<GenerationSettings>) => void;
    setModels: (models: ModelInfo[]) => void;
    setControlNets: (models: string[]) => void;
    setStatus: (status: 'idle' | 'generating' | 'error') => void;
    setCurrentImage: (image: string | null) => void;
    addToGallery: (image: string) => void;
    setPreview: (preview: PreviewMessage | null) => void;
    setServerStatus: (status: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
    settings: {
        prompt: "An astronaut riding a horse on mars, detailed, 8k",
        negative_prompt: "blurry, low quality, distorted",
        width: 512,
        height: 512,
        num_images: 1,
        batch_size: 1,
        steps: 20,
        cfg_scale: 7.0,
        seed: -1,
        scheduler: "karras",
        sampler: "dpmpp_2m_cfgpp",
        model_path: "", // Default empty, user must select
        refiner_model_path: "",
        refiner_switch_step: 15,
        img2img_mode: false,
        img2img_denoise: 0.75,
        img2img_image: undefined,
        controlnet_enabled: false,
        controlnet_strength: 1.0,
        controlnet_type: "canny",
        hiresfix: false,
        adetailer: false,
        enhance_prompt: false,
        stable_fast: false,
        reuse_seed: false,
        keep_models_loaded: true,
        enable_preview: true,
        enable_multiscale: false,
        multiscale_preset: "disabled",
        multiscale_factor: 0.5,
        multiscale_fullres_start: 10,
        multiscale_fullres_end: 8,
        multiscale_intermittent_fullres: true,
        deepcache_enabled: false,
        deepcache_interval: 3,
        deepcache_depth: 2,
        cfg_free_enabled: false,
        cfg_free_start_percent: 70.0,
        tome_enabled: false,
        tome_ratio: 0.5,
    },
    availableModels: [],
    availableControlNets: [],
    status: 'idle',
    currentImage: null,
    gallery: [],
    preview: null,
    serverStatus: false,

    setSettings: (newSettings) => set((state) => ({ settings: { ...state.settings, ...newSettings } })),
    setModels: (models) => set({ availableModels: models }),
    setControlNets: (models) => set({ availableControlNets: models }),
    setStatus: (status) => set({ status }),
    setCurrentImage: (image) => set({ currentImage: image }),
    addToGallery: (image) => set((state) => ({ gallery: [image, ...state.gallery] })),
    setPreview: (preview) => set({ preview }),
    setServerStatus: (status) => set({ serverStatus: status }),
}));
