import { type ChangeEvent, useMemo, useState } from 'react';
import { FolderClock, Layers3, SlidersHorizontal, Sparkles, WandSparkles, Workflow } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { useGenerationActions } from '../hooks/use-generation-actions';
import { getModelSelectionUpdates } from '../lib/settings';
import { cn } from '../lib/utils';
import { useStore } from '../store/useStore';
import type { GenerationSettings as GenerationSettingsShape } from '../types';
import { ImageInput } from './ImageInput';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';
import { Button } from './ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Input } from './ui/input';
import { Label } from './ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { ScrollArea } from './ui/scroll-area';
import { Switch } from './ui/switch';
import { Textarea } from './ui/textarea';
import { useShallow } from 'zustand/react/shallow';

type FeedbackTone = 'success' | 'warning' | 'error';

type FeedbackState = {
  tone: FeedbackTone;
  text: string;
};

type SelectOption = {
  value: string;
  label: string;
};

const samplerOptions = [
  'dpmpp_2m',
  'dpmpp_2m_cfgpp',
  'dpmpp_sde',
  'dpmpp_sde_cfgpp',
  'euler',
  'euler_cfgpp',
  'euler_ancestral',
  'euler_ancestral_cfgpp',
];

const schedulerOptions = ['karras', 'exponential', 'sgm_uniform', 'simple', 'normal', 'ays'];
const controlTypes = ['canny', 'depth', 'pose', 'softedge'];
const multiscalePresets = ['balanced', 'detailed', 'creative', 'disabled'];

function Field({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      {description ? <p className="text-xs leading-5 text-muted">{description}</p> : null}
      {children}
    </div>
  );
}

function FeatureSwitch({
  checked,
  description,
  disabled,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  description?: string;
  disabled?: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div
      className={cn(
        'flex items-start justify-between gap-4 rounded-[1.35rem] border px-4 py-3',
        disabled ? 'border-line bg-oat/45 opacity-70' : 'border-line bg-oat/65',
      )}
    >
      <div className="space-y-1">
        <p className="text-sm font-medium text-ink">{label}</p>
        {description ? <p className="text-xs leading-5 text-muted">{description}</p> : null}
      </div>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} />
    </div>
  );
}

function StatusLine({ tone, text }: FeedbackState) {
  return (
    <p
      className={cn(
        'text-sm',
        tone === 'error' ? 'text-clay-strong' : tone === 'warning' ? 'text-muted' : 'text-clay',
      )}
    >
      {text}
    </p>
  );
}

function SupportHint({
  capability,
  label,
}: {
  capability?: boolean;
  label: string;
}) {
  if (capability !== false) return null;

  return <p className="text-xs leading-5 text-muted">{label}</p>;
}

function OptionSelect({
  disabled,
  onValueChange,
  options,
  placeholder,
  value,
}: {
  disabled?: boolean;
  onValueChange: (value: string) => void;
  options: SelectOption[];
  placeholder: string;
  value?: string;
}) {
  return (
    <Select disabled={disabled} onValueChange={onValueChange} value={value}>
      <SelectTrigger>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {options.length > 0 ? (
          options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))
        ) : (
          <div className="px-4 py-3 text-sm text-muted">No options available</div>
        )}
      </SelectContent>
    </Select>
  );
}

export function GenerationSettings() {
  const {
    availableControlNets,
    availableModels,
    serverStatus,
    settings,
    settingsHistory,
    setSettings,
    status,
  } = useStore(useShallow((state) => ({
    availableControlNets: state.availableControlNets,
    availableModels: state.availableModels,
    serverStatus: state.serverStatus,
    settings: state.settings,
    settingsHistory: state.settingsHistory,
    setSettings: state.setSettings,
    status: state.status,
  })));
  const {
    handleGenerate,
    importSettingsFromFiles,
    restoreLastSeed,
    saveSettingsSnapshot,
    updateAutotuneSettings,
  } = useGenerationActions();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [historyFeedback, setHistoryFeedback] = useState<FeedbackState | null>(null);
  const [actionFeedback, setActionFeedback] = useState<FeedbackState | null>(null);
  const [performanceFeedback, setPerformanceFeedback] = useState<FeedbackState | null>(null);

  const modelOptions = useMemo<SelectOption[]>(
    () => availableModels.map((model) => ({ value: model.path, label: model.name })),
    [availableModels],
  );
  const controlNetOptions = useMemo<SelectOption[]>(
    () => availableControlNets.map((model) => ({ value: model, label: model })),
    [availableControlNets],
  );
  const currentModel = availableModels.find((model) => model.path === settings.model_path);
  const capabilities = currentModel?.capabilities;

  const importDropzone = useDropzone({
    accept: { 'image/*': [] },
    maxFiles: 1,
    multiple: false,
    onDrop: (acceptedFiles) => {
      void (async () => {
        const result = await importSettingsFromFiles(acceptedFiles);
        setHistoryFeedback({
          tone: result.ok ? (result.warning ? 'warning' : 'success') : 'error',
          text: result.warning ? `${result.message} ${result.warning}` : result.message,
        });
      })();
    },
  });

  const handleModelChange = (value: string) => {
    const selectedModel = availableModels.find((model) => model.path === value);
    if (!selectedModel) {
      setSettings({ model_path: value });
      return;
    }

    setSettings(getModelSelectionUpdates(selectedModel, availableModels));
  };

  const updateNumber =
    (key: keyof GenerationSettingsShape, fallback = 0) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const raw = event.currentTarget.value;
      const nextValue = raw === '' ? fallback : Number(raw);
      if (!Number.isNaN(nextValue)) {
        setSettings({ [key]: nextValue } as Partial<GenerationSettingsShape>);
      }
    };

  const restoreSnapshot = (snapshot: GenerationSettingsShape) => {
    setSettings(snapshot);
    setHistoryFeedback({
      tone: 'success',
      text: 'Restored a saved local snapshot.',
    });
  };

  const handleStableFastChange = (checked: boolean) => {
    if (checked && settings.torch_compile) {
      void (async () => {
        const result = await updateAutotuneSettings({
          stable_fast: true,
          torch_compile: false,
          vae_autotune: settings.vae_autotune,
        });
        setPerformanceFeedback(result.ok ? null : { tone: 'error', text: result.message });
      })();
      return;
    }

    setSettings({ stable_fast: checked });
    setPerformanceFeedback(null);
  };

  const handleModelAutotuneChange = (checked: boolean) => {
    void (async () => {
      const result = await updateAutotuneSettings({
        stable_fast: checked ? false : settings.stable_fast,
        torch_compile: checked,
        vae_autotune: settings.vae_autotune,
      });
      setPerformanceFeedback(result.ok ? null : { tone: 'error', text: result.message });
    })();
  };

  const handleVaeAutotuneChange = (checked: boolean) => {
    void (async () => {
      const result = await updateAutotuneSettings({
        torch_compile: settings.torch_compile,
        vae_autotune: checked,
      });
      setPerformanceFeedback(result.ok ? null : { tone: 'error', text: result.message });
    })();
  };

  const capabilityTokens = [
    currentModel?.type,
    capabilities?.supports_img2img ? 'Img2Img' : null,
    capabilities?.supports_controlnet ? 'ControlNet' : null,
    capabilities?.supports_hires_fix ? 'Hires Fix' : null,
  ].filter(Boolean) as string[];

  return (
    <section className="studio-panel flex h-full min-h-0 flex-col overflow-hidden rounded-[2rem] border border-line">
      <div className="border-b border-line px-5 py-5 sm:px-6">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-muted">Session</p>
            <p className="mt-1 text-sm font-medium text-ink">
              {currentModel ? currentModel.name : 'Select a model'}
            </p>
          </div>
          <div className="rounded-full bg-sand px-3 py-1.5 text-xs text-muted">
            {serverStatus ? 'Connected' : 'Offline'}
          </div>
        </div>

        <Button
          className="mt-4 w-full"
          size="lg"
          variant={status === 'generating' ? 'destructive' : 'default'}
          onClick={() => void handleGenerate()}
        >
          {status === 'generating' ? 'Interrupt generation' : 'Generate image'}
        </Button>

        {status === 'error' ? <StatusLine tone="error" text="Generation failed. Check the backend logs." /> : null}
      </div>

      <ScrollArea className="soft-scroll min-h-0 flex-1">
        <div className="space-y-6 px-5 py-5 sm:px-6">
          <div className="space-y-4">
            <Field label="Model">
              <OptionSelect
                onValueChange={handleModelChange}
                options={modelOptions}
                placeholder="Select a model"
                value={settings.model_path || undefined}
              />
            </Field>

            <div className="flex flex-wrap gap-2">
              {capabilityTokens.length > 0 ? (
                capabilityTokens.map((token) => (
                  <span key={token} className="rounded-full bg-sand px-3 py-1.5 text-xs text-muted">
                    {token}
                  </span>
                ))
              ) : (
                <span className="rounded-full bg-sand px-3 py-1.5 text-xs text-muted">Waiting for model metadata</span>
              )}
            </div>
          </div>

          <Field label="Prompt">
            <Textarea
              placeholder="Describe the image you want to generate..."
              value={settings.prompt}
              onChange={(event) => setSettings({ prompt: event.currentTarget.value })}
            />
          </Field>

          <Field label="Negative prompt">
            <Textarea
              className="min-h-[104px]"
              placeholder="List what the image should avoid..."
              value={settings.negative_prompt}
              onChange={(event) => setSettings({ negative_prompt: event.currentTarget.value })}
            />
          </Field>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Width">
              <Input type="number" step="64" value={settings.width} onChange={updateNumber('width', 512)} />
            </Field>
            <Field label="Height">
              <Input type="number" step="64" value={settings.height} onChange={updateNumber('height', 512)} />
            </Field>
            <Field label="Steps">
              <Input type="number" min="1" value={settings.steps} onChange={updateNumber('steps', 20)} />
            </Field>
            <Field label="CFG scale">
              <Input type="number" step="0.5" value={settings.cfg_scale} onChange={updateNumber('cfg_scale', 7)} />
            </Field>
            <Field label="Batch size">
              <Input type="number" min="1" max="4" value={settings.batch_size} onChange={updateNumber('batch_size', 1)} />
            </Field>
            <Field label="Images">
              <Input type="number" min="1" value={settings.num_images} onChange={updateNumber('num_images', 1)} />
            </Field>
          </div>

          <div className="space-y-4">
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-muted">
              <SlidersHorizontal className="h-3.5 w-3.5 text-clay" />
              Sampling
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Sampler">
                <OptionSelect
                  onValueChange={(value) => setSettings({ sampler: value })}
                  options={samplerOptions.map((sampler) => ({ value: sampler, label: sampler }))}
                  placeholder="Sampler"
                  value={settings.sampler}
                />
              </Field>
              <Field label="Scheduler">
                <OptionSelect
                  onValueChange={(value) => setSettings({ scheduler: value })}
                  options={schedulerOptions.map((scheduler) => ({ value: scheduler, label: scheduler }))}
                  placeholder="Scheduler"
                  value={settings.scheduler}
                />
              </Field>
              <Field label="Preview fidelity">
                <OptionSelect
                  disabled={!settings.enable_preview}
                  onValueChange={(value) =>
                    setSettings({ preview_fidelity: value as NonNullable<GenerationSettingsShape['preview_fidelity']> })
                  }
                  options={[
                    { value: 'low', label: 'Low · faster' },
                    { value: 'balanced', label: 'Balanced · default' },
                    { value: 'high', label: 'High · slower' },
                  ]}
                  placeholder="Preview fidelity"
                  value={settings.preview_fidelity || 'balanced'}
                />
              </Field>
              <FeatureSwitch
                checked={settings.reuse_seed}
                label="Reuse seed"
                onCheckedChange={(checked) => setSettings({ reuse_seed: checked })}
              />
            </div>
          </div>

          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
            <div className="rounded-[1.75rem] border border-line bg-paper/65 p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted">Advanced</p>
                <CollapsibleTrigger asChild>
                  <Button size="sm" variant="outline">
                    {advancedOpen ? 'Hide' : 'Advanced'}
                  </Button>
                </CollapsibleTrigger>
              </div>

              <CollapsibleContent className="data-[state=closed]:overflow-hidden">
                <div className="my-4 h-px bg-line" />

                <Accordion className="space-y-3" type="multiple" defaultValue={['sampling']} >
                  <AccordionItem value="sampling">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-clay" />
                        Sampling
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4">
                        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_auto]">
                          <Field label="Seed">
                            <Input type="number" value={settings.seed ?? -1} onChange={updateNumber('seed', -1)} />
                          </Field>
                          <Button
                            className="self-end"
                            type="button"
                            variant="outline"
                            onClick={() => {
                              void (async () => {
                                const result = await restoreLastSeed();
                                setActionFeedback({
                                  tone: result.ok ? 'success' : 'error',
                                  text: result.message,
                                });
                              })();
                            }}
                          >
                            Use last seed
                          </Button>
                        </div>
                        {actionFeedback ? <StatusLine {...actionFeedback} /> : null}
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="enhancements">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <WandSparkles className="h-4 w-4 text-clay" />
                        Enhancements
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-3">
                        <FeatureSwitch
                          checked={settings.hiresfix}
                          disabled={capabilities?.supports_hires_fix === false}
                          label="High Res Fix"
                          onCheckedChange={(checked) => setSettings({ hiresfix: checked })}
                        />
                        <SupportHint
                          capability={capabilities?.supports_hires_fix}
                          label="The selected model does not support High Res Fix."
                        />
                        <FeatureSwitch
                          checked={settings.adetailer}
                          label="ADetailer"
                          onCheckedChange={(checked) => setSettings({ adetailer: checked })}
                        />
                        <FeatureSwitch
                          checked={settings.enhance_prompt}
                          label="Prompt enhancer"
                          onCheckedChange={(checked) => setSettings({ enhance_prompt: checked })}
                        />
                        <FeatureSwitch
                          checked={settings.enable_preview}
                          label="Live preview"
                          onCheckedChange={(checked) => setSettings({ enable_preview: checked })}
                        />
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="refiner">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <Layers3 className="h-4 w-4 text-clay" />
                        Refiner
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4">
                        <Field label="Refiner model">
                          <OptionSelect
                            disabled={currentModel?.type !== 'SDXL'}
                            onValueChange={(value) => setSettings({ refiner_model_path: value === '__none' ? '' : value })}
                            options={[{ value: '__none', label: 'None' }, ...modelOptions]}
                            placeholder="None"
                            value={settings.refiner_model_path || '__none'}
                          />
                        </Field>
                        <Field label="Switch step">
                          <Input
                            disabled={!settings.refiner_model_path}
                            type="number"
                            min="1"
                            value={settings.refiner_switch_step ?? 15}
                            onChange={updateNumber('refiner_switch_step', 15)}
                          />
                        </Field>
                        <SupportHint
                          capability={currentModel?.type === 'SDXL'}
                          label="Refiner selection is only relevant for SDXL base models."
                        />
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="img2img">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <Workflow className="h-4 w-4 text-clay" />
                        Image to image
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4">
                        <FeatureSwitch
                          checked={settings.img2img_mode}
                          disabled={capabilities?.supports_img2img === false}
                          label="Enable Img2Img"
                          onCheckedChange={(checked) => setSettings({ img2img_mode: checked })}
                        />
                        <SupportHint
                          capability={capabilities?.supports_img2img}
                          label="The selected model does not support Img2Img."
                        />
                        {settings.img2img_mode ? (
                          <ImageInput
                            label="Input image"
                            value={settings.img2img_image}
                            onChange={(base64) => setSettings({ img2img_image: base64 ?? undefined })}
                          />
                        ) : null}
                        <Field label="Denoising strength">
                          <Input
                            disabled={!settings.img2img_mode}
                            type="number"
                            min="0"
                            max="1"
                            step="0.05"
                            value={settings.img2img_denoise}
                            onChange={updateNumber('img2img_denoise', 0.75)}
                          />
                        </Field>
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="controlnet">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-clay" />
                        ControlNet
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4">
                        <FeatureSwitch
                          checked={settings.controlnet_enabled}
                          disabled={capabilities?.supports_controlnet === false}
                          label="Enable ControlNet"
                          onCheckedChange={(checked) => setSettings({ controlnet_enabled: checked })}
                        />
                        <SupportHint
                          capability={capabilities?.supports_controlnet}
                          label="The selected model does not support ControlNet."
                        />
                        {settings.controlnet_enabled ? (
                          <>
                            <Field label="ControlNet model">
                              <OptionSelect
                                onValueChange={(value) =>
                                  setSettings({ controlnet_model: value === '__none' ? undefined : value })
                                }
                                options={[{ value: '__none', label: 'Select a model' }, ...controlNetOptions]}
                                placeholder="Select a ControlNet model"
                                value={settings.controlnet_model || '__none'}
                              />
                            </Field>
                            <Field label="Control type">
                              <OptionSelect
                                onValueChange={(value) => setSettings({ controlnet_type: value })}
                                options={controlTypes.map((type) => ({ value: type, label: type }))}
                                placeholder="Control type"
                                value={settings.controlnet_type}
                              />
                            </Field>
                            <Field label="Strength">
                              <Input
                                type="number"
                                min="0"
                                max="2"
                                step="0.1"
                                value={settings.controlnet_strength}
                                onChange={updateNumber('controlnet_strength', 1)}
                              />
                            </Field>
                            {!settings.img2img_mode ? (
                              <ImageInput
                                compact
                                label="Control image"
                                value={settings.img2img_image}
                                onChange={(base64) => setSettings({ img2img_image: base64 ?? undefined })}
                              />
                            ) : (
                              <p className="text-xs leading-5 text-muted">
                                Uses the Img2Img source image.
                              </p>
                            )}
                          </>
                        ) : null}
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="performance">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <SlidersHorizontal className="h-4 w-4 text-clay" />
                        Performance and optimizations
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-3">
                        <FeatureSwitch
                          checked={settings.stable_fast}
                          disabled={capabilities?.supports_stable_fast === false}
                          label="Stable Fast"
                          onCheckedChange={handleStableFastChange}
                        />
                        <SupportHint
                          capability={capabilities?.supports_stable_fast}
                          label="The selected model does not support Stable Fast."
                        />
                        <FeatureSwitch
                          checked={settings.torch_compile}
                          disabled={settings.stable_fast}
                          description="Compiles the diffusion model for faster repeat runs."
                          label="Model autotune (torch.compile)"
                          onCheckedChange={handleModelAutotuneChange}
                        />
                        <FeatureSwitch
                          checked={settings.vae_autotune}
                          description="Compiles the VAE decoder when enabled for faster decode and encode steps."
                          label="VAE autotune (torch.compile)"
                          onCheckedChange={handleVaeAutotuneChange}
                        />
                        <Field label="Weight quantization">
                          <OptionSelect
                            onValueChange={(value) =>
                              setSettings({ weight_quantization: value === 'none' ? null : (value as 'fp8' | 'nvfp4') })
                            }
                            options={[
                              { value: 'none', label: 'None · FP16/BF16' },
                              { value: 'fp8', label: 'FP8 · 8-bit' },
                              { value: 'nvfp4', label: 'NVFP4 · 4-bit' },
                            ]}
                            placeholder="Weight quantization"
                            value={settings.weight_quantization || 'none'}
                          />
                        </Field>
                        <FeatureSwitch
                          checked={settings.keep_models_loaded}
                          label="Keep models loaded"
                          onCheckedChange={(checked) => setSettings({ keep_models_loaded: checked })}
                        />
                        <FeatureSwitch
                          checked={settings.deepcache_enabled}
                          disabled={capabilities?.supports_deepcache === false}
                          label="DeepCache"
                          onCheckedChange={(checked) => setSettings({ deepcache_enabled: checked })}
                        />
                        <FeatureSwitch
                          checked={settings.tome_enabled}
                          disabled={capabilities?.supports_tome === false}
                          label="ToMe"
                          onCheckedChange={(checked) => setSettings({ tome_enabled: checked })}
                        />
                        {performanceFeedback ? <StatusLine {...performanceFeedback} /> : null}
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="multiscale">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <Layers3 className="h-4 w-4 text-clay" />
                        Multiscale generation
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4">
                        <FeatureSwitch
                          checked={settings.enable_multiscale}
                          label="Enable multiscale"
                          onCheckedChange={(checked) => setSettings({ enable_multiscale: checked })}
                        />
                        {settings.enable_multiscale ? (
                          <>
                            <Field label="Preset">
                              <OptionSelect
                                onValueChange={(value) => setSettings({ multiscale_preset: value })}
                                options={multiscalePresets.map((preset) => ({ value: preset, label: preset }))}
                                placeholder="Preset"
                                value={settings.multiscale_preset}
                              />
                            </Field>
                            <Field label="Factor">
                              <Input
                                type="number"
                                min="0.1"
                                max="1"
                                step="0.1"
                                value={settings.multiscale_factor}
                                onChange={updateNumber('multiscale_factor', 0.5)}
                              />
                            </Field>
                          </>
                        ) : null}
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="history">
                    <AccordionTrigger>
                      <span className="flex items-center gap-2">
                        <FolderClock className="h-4 w-4 text-clay" />
                        History and import
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4">
                        <FeatureSwitch
                          checked={!!settings.persist_prompt_history}
                          label="Include prompts in server history"
                          onCheckedChange={(checked) => setSettings({ persist_prompt_history: checked })}
                        />

                        <div className="flex flex-wrap gap-3">
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => {
                              void (async () => {
                                const result = await saveSettingsSnapshot();
                                setHistoryFeedback({
                                  tone: result.ok ? 'success' : 'error',
                                  text: result.message,
                                });
                              })();
                            }}
                          >
                            Save settings
                          </Button>
                        </div>

                        <div
                          {...importDropzone.getRootProps()}
                          className="cursor-pointer rounded-[1.5rem] border border-dashed border-line bg-oat/55 px-4 py-5 transition hover:border-clay/35 hover:bg-oat"
                        >
                          <input {...importDropzone.getInputProps()} />
                          <div className="space-y-1 text-sm">
                            <p className="font-medium text-ink">Import from image</p>
                            <p className="text-xs leading-5 text-muted">Drop an image to restore its settings.</p>
                          </div>
                        </div>

                        {settingsHistory.length > 0 ? (
                          <div className="space-y-2">
                            <p className="text-xs uppercase tracking-[0.16em] text-muted">Local snapshots</p>
                            <div className="space-y-2">
                              {settingsHistory.slice(0, 5).map((snapshot) => (
                                <button
                                  key={snapshot.id}
                                  type="button"
                                  onClick={() => restoreSnapshot(snapshot.settings)}
                                  className="flex w-full items-center justify-between rounded-[1.2rem] border border-line bg-paper px-4 py-3 text-left transition hover:border-clay/35 hover:bg-oat/75"
                                >
                                  <div>
                                    <p className="text-sm font-medium text-ink">{snapshot.settings.model_path || 'Saved state'}</p>
                                    <p className="text-xs leading-5 text-muted">
                                      {new Intl.DateTimeFormat(undefined, {
                                        dateStyle: 'medium',
                                        timeStyle: 'short',
                                      }).format(snapshot.ts * 1000)}
                                    </p>
                                  </div>
                                  <span className="text-xs text-muted">Restore</span>
                                </button>
                              ))}
                            </div>
                          </div>
                        ) : null}

                        {historyFeedback ? <StatusLine {...historyFeedback} /> : null}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </CollapsibleContent>
            </div>
          </Collapsible>
        </div>
      </ScrollArea>
    </section>
  );
}
