import type { GenerationSettings, ImageMetadata, ModelInfo } from '../types';

export function getDefaultModel(models: ModelInfo[]) {
  return models.find((model) => model.name.toLowerCase().includes('dreamshaper')) ?? models[0];
}

function disableUnsupportedCapabilities(
  model: ModelInfo,
  updates: Partial<GenerationSettings>,
): Partial<GenerationSettings> {
  const caps = model.capabilities;

  if (!caps.supports_hires_fix) updates.hiresfix = false;
  if (!caps.supports_img2img) updates.img2img_mode = false;
  if (!caps.supports_controlnet) updates.controlnet_enabled = false;
  if (!caps.supports_stable_fast) updates.stable_fast = false;
  if (!caps.supports_deepcache) updates.deepcache_enabled = false;
  if (!caps.supports_tome) updates.tome_enabled = false;

  return updates;
}

export function getModelSelectionUpdates(
  selectedModel: ModelInfo,
  availableModels: ModelInfo[],
): Partial<GenerationSettings> {
  const updates: Partial<GenerationSettings> = {
    model_path: selectedModel.path,
  };

  if (selectedModel.type === 'Flux2Klein') {
    updates.width = 1024;
    updates.height = 1024;
    updates.sampler = 'euler';
    updates.scheduler = 'simple';
    updates.steps = 4;
    updates.cfg_scale = 1.0;
    updates.refiner_model_path = '';
  } else if (selectedModel.type === 'SDXL') {
    updates.width = 1024;
    updates.height = 1024;
    updates.sampler = 'euler';
    updates.scheduler = 'simple';
    updates.steps = 25;

    const refiner = availableModels.find(
      (model) =>
        model.type === 'SDXL' &&
        (model.name.toLowerCase().includes('refiner') || model.path.toLowerCase().includes('refiner')),
    );

    if (refiner) {
      updates.refiner_model_path = refiner.path;
      updates.refiner_switch_step = 20;
    }
  } else {
    updates.width = 512;
    updates.height = 512;
    updates.sampler = 'dpmpp_2m';
    updates.scheduler = 'karras';
    updates.steps = 20;
    updates.refiner_model_path = '';
  }

  return disableUnsupportedCapabilities(selectedModel, updates);
}

export function getMetadataSettingsUpdates(meta: Partial<ImageMetadata>): Partial<GenerationSettings> {
  const updates: Partial<GenerationSettings> = {};

  if (meta.seed !== undefined) updates.seed = meta.seed;
  if (meta.steps !== undefined) updates.steps = meta.steps;
  if (meta.cfg_scale !== undefined) updates.cfg_scale = meta.cfg_scale;
  if (meta.sampler) updates.sampler = meta.sampler;
  if (meta.scheduler) updates.scheduler = meta.scheduler;
  if (meta.model_path) updates.model_path = meta.model_path;
  if (meta.width) updates.width = meta.width;
  if (meta.height) updates.height = meta.height;
  if (meta.prompt) updates.prompt = meta.prompt;
  if (meta.negative_prompt) updates.negative_prompt = meta.negative_prompt;
  if (meta.denoise !== undefined) updates.img2img_denoise = meta.denoise;
  if (meta.weight_quantization !== undefined) updates.weight_quantization = meta.weight_quantization;
  if (meta.torch_compile !== undefined) updates.torch_compile = meta.torch_compile;
  if (meta.fp8_inference !== undefined) updates.fp8_inference = meta.fp8_inference;

  if (meta.controlnet_enabled !== undefined) updates.controlnet_enabled = meta.controlnet_enabled;
  if (meta.controlnet_model !== undefined) updates.controlnet_model = meta.controlnet_model;
  if (meta.controlnet_strength !== undefined) updates.controlnet_strength = meta.controlnet_strength;
  if (meta.controlnet_type !== undefined) updates.controlnet_type = meta.controlnet_type;
  if (meta.enable_multiscale !== undefined) updates.enable_multiscale = meta.enable_multiscale;
  if (meta.multiscale_preset !== undefined) updates.multiscale_preset = meta.multiscale_preset;
  if (meta.multiscale_factor !== undefined) updates.multiscale_factor = meta.multiscale_factor;
  if (meta.multiscale_fullres_start !== undefined) updates.multiscale_fullres_start = meta.multiscale_fullres_start;
  if (meta.multiscale_fullres_end !== undefined) updates.multiscale_fullres_end = meta.multiscale_fullres_end;
  if (meta.multiscale_intermittent_fullres !== undefined) {
    updates.multiscale_intermittent_fullres = meta.multiscale_intermittent_fullres;
  }
  if (meta.deepcache_enabled !== undefined) updates.deepcache_enabled = meta.deepcache_enabled;
  if (meta.deepcache_interval !== undefined) updates.deepcache_interval = meta.deepcache_interval;
  if (meta.deepcache_depth !== undefined) updates.deepcache_depth = meta.deepcache_depth;
  if (meta.cfg_free_enabled !== undefined) updates.cfg_free_enabled = meta.cfg_free_enabled;
  if (meta.cfg_free_start_percent !== undefined) updates.cfg_free_start_percent = meta.cfg_free_start_percent;
  if (meta.tome_enabled !== undefined) updates.tome_enabled = meta.tome_enabled;
  if (meta.tome_ratio !== undefined) updates.tome_ratio = meta.tome_ratio;

  return updates;
}
