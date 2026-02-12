import { Button, NumberInput, Select, Stack, Textarea, Switch, Group, Collapse, Box, Text, Accordion } from '@mantine/core';
import { useStore } from '../store/useStore';
import { generateImage, interruptGeneration, listModels, listControlNets, getLastSeed, getSettingsHistory, postSettingsSnapshot } from '../api/client';
import { useEffect, useMemo } from 'react';
import { useDisclosure } from '@mantine/hooks';
import { IconCaretDown, IconCaretRight } from '@tabler/icons-react';
import { ImageInput } from './ImageInput';

export function GenerationSettings() {
    const { settings, setSettings, status, setStatus, setCurrentImage, addToGallery, availableModels, setModels, setPreview, availableControlNets, setControlNets, settingsHistory, setSettingsHistory, appendSettingsSnapshot } = useStore();
    const [openedAdvanced, { toggle: toggleAdvanced }] = useDisclosure(false);
    const [openedHistory, { toggle: toggleHistory }] = useDisclosure(false);

    useEffect(() => {
        listModels().then(models => {
            setModels(models);
            // Automatically select the first model if none is currently selected
            // We use the store's current state directly to avoid dependency loop
            const currentSettings = useStore.getState().settings;
            if (!currentSettings.model_path && models.length > 0) {
                const defaultModel = models.find(m => m.name.toLowerCase().includes("dreamshaper")) || models[0];
                const updates: any = { model_path: defaultModel.path };
                
                if (defaultModel.type === "Flux2Klein") {
                    updates.width = 1024;
                    updates.height = 1024;
                    updates.sampler = "euler";
                    updates.scheduler = "simple";
                    updates.steps = 4;
                    updates.cfg_scale = 1.0;
                } else if (defaultModel.type === "SDXL") {
                    updates.width = 1024;
                    updates.height = 1024;
                    updates.sampler = "euler";
                    updates.scheduler = "simple";
                    updates.steps = 25;
                } else {
                    updates.width = 512;
                    updates.height = 512;
                    updates.sampler = "dpmpp_2m";
                    updates.scheduler = "karras";
                    updates.steps = 20;
                }
                setSettings(updates);
            }
        }).catch(console.error);
        listControlNets().then(res => setControlNets(res.models)).catch(console.error);

        // Load settings history for the UI
        getSettingsHistory().then(res => {
            if (res && Array.isArray(res.history)) setSettingsHistory(res.history);
        }).catch(() => {
            /* ignore */
        });
    }, [setModels, setControlNets, setSettings, setSettingsHistory]);

    const handleGenerate = async () => {
        if (status === 'generating') {
            await interruptGeneration();
            setStatus('idle');
            return;
        }

        setStatus('generating');
        setPreview(null); // Clear previous preview
        try {
            // Auto-save a local client-side snapshot for quick restore/history
            const localSnap = {
                id: (Date.now().toString(36) + Math.random().toString(36).slice(2,8)),
                ts: Math.floor(Date.now() / 1000),
                settings: { ...settings }
            };
            appendSettingsSnapshot(localSnap);

            const res = await generateImage(settings);
            if (res.images && res.images.length > 0) {
                setCurrentImage(res.images[0]);
                addToGallery(res.images[0]);
            } else if (res.image) {
                setCurrentImage(res.image);
                addToGallery(res.image);
            }
        } catch (error) {
            console.error("Generation failed:", error);
        } finally {
            setStatus('idle');
        }
    };

    const modelOptions = useMemo(() => availableModels.map(m => ({ value: m.path, label: m.name })), [availableModels]);
    const controlNetOptions = useMemo(() => availableControlNets.map(m => ({ value: m, label: m })), [availableControlNets]);
    const currentModel = availableModels.find(m => m.path === settings.model_path);
    const caps = currentModel?.capabilities;

    return (
        <Stack gap="md" p="xs">
            <Select
                label="Model"
                placeholder="Select model"
                data={modelOptions}
                value={settings.model_path}
                nothingFoundMessage="No models found"
                maxDropdownHeight={400}
                onChange={(v) => {
                    if (!v) {
                        setSettings({ model_path: "" });
                        return;
                    }

                    const selectedModel = availableModels.find(m => m.path === v);
                    const updates: any = { model_path: v };

                    if (selectedModel) {
                        if (selectedModel.type === "Flux2Klein") {
                            updates.width = 1024;
                            updates.height = 1024;
                            updates.sampler = "euler";
                            updates.scheduler = "simple";
                            updates.steps = 4;
                            updates.cfg_scale = 1.0;
                        } else if (selectedModel.type === "SDXL") {
                            updates.width = 1024;
                            updates.height = 1024;
                            updates.sampler = "euler";
                            updates.scheduler = "simple";
                            updates.steps = 25;
                            // Auto-enable refiner if available
                            const refiner = availableModels.find(m => m.type === "SDXL" && (m.name.toLowerCase().includes("refiner") || m.path.toLowerCase().includes("refiner")));
                            if (refiner) {
                                updates.refiner_model_path = refiner.path;
                                updates.refiner_switch_step = 20;
                            }
                        } else {
                            // SD1.5
                            updates.width = 512;
                            updates.height = 512;
                            updates.sampler = "dpmpp_2m";
                            updates.scheduler = "karras";
                            updates.steps = 20;
                        }
                        
                        // Disable features not supported by new model
                        if (selectedModel.capabilities) {
                            if (!selectedModel.capabilities.supports_hires_fix) updates.hiresfix = false;
                            if (!selectedModel.capabilities.supports_img2img) updates.img2img_mode = false;
                            if (!selectedModel.capabilities.supports_controlnet) updates.controlnet_enabled = false;
                            if (!selectedModel.capabilities.supports_stable_fast) updates.stable_fast = false;
                            if (!selectedModel.capabilities.supports_deepcache) updates.deepcache_enabled = false;
                            if (!selectedModel.capabilities.supports_tome) updates.tome_enabled = false;
                        }
                    }
                    setSettings(updates);
                }}
            />

            <Textarea
                label="Prompt"
                placeholder="Describe your image..."
                minRows={3}
                autosize
                value={settings.prompt}
                onChange={(e) => setSettings({ prompt: e.currentTarget.value })}
            />

            <Textarea
                label="Negative Prompt"
                placeholder="What to avoid..."
                minRows={2}
                autosize
                value={settings.negative_prompt}
                onChange={(e) => setSettings({ negative_prompt: e.currentTarget.value })}
            />

            <Group grow>
                <NumberInput
                    label="Width"
                    value={settings.width}
                    onChange={(v) => setSettings({ width: Number(v) })}
                    step={64}
                />
                <NumberInput
                    label="Height"
                    value={settings.height}
                    onChange={(v) => setSettings({ height: Number(v) })}
                    step={64}
                />
            </Group>

            <Group grow>
                <NumberInput
                    label="Steps"
                    value={settings.steps}
                    onChange={(v) => setSettings({ steps: Number(v) })}
                />
                <NumberInput
                    label="CFG Scale"
                    value={settings.cfg_scale}
                    onChange={(v) => setSettings({ cfg_scale: Number(v) })}
                    step={0.5}
                    decimalScale={1}
                    fixedDecimalScale
                />
            </Group>

            <Group grow>
                <NumberInput
                    label="Batch Size"
                    value={settings.batch_size}
                    onChange={(v) => setSettings({ batch_size: Number(v) })}
                    min={1}
                    max={4}
                />
                <NumberInput
                    label="Images"
                    value={settings.num_images}
                    onChange={(v) => setSettings({ num_images: Number(v) })}
                    min={1}
                />
            </Group>

            <Button
                onClick={handleGenerate}
                loading={status === 'generating' && false} // We want interrupt button if generating
                color={status === 'generating' ? 'red' : 'blue'}
                size="lg"
            >
                {status === 'generating' ? 'Interrupt' : 'Generate'}
            </Button>

            <Box>
                <Group onClick={toggleAdvanced} style={{ cursor: 'pointer' }} mb={5}>
                    {openedAdvanced ? <IconCaretDown size={16} /> : <IconCaretRight size={16} />}
                    <Text size="sm" fw={500}>Advanced Settings</Text>
                </Group>
                <Collapse in={openedAdvanced}>
                    <Accordion variant="separated" radius="md" mt="xs" multiple defaultValue={['sampling']}>
                        <Accordion.Item value="sampling">
                            <Accordion.Control>Sampling & Guidance</Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    <Group>
                                        <NumberInput
                                            label="Seed"
                                            description="-1 for random"
                                            value={settings.seed}
                                            onChange={(v) => setSettings({ seed: Number(v) })}
                                        />

                                        <Button
                                            variant="outline"
                                            onClick={async () => {
                                                try {
                                                    const res = await getLastSeed();
                                                    const s = res?.seed ?? -1;
                                                    setSettings({ seed: typeof s === 'number' ? s : -1 });
                                                } catch (err) {
                                                    console.error('Failed to fetch last seed', err);
                                                }
                                            }}
                                        >
                                            Use last seed
                                        </Button>
                                    </Group>
                                    <Group grow>
                                        <Select
                                            label="Sampler"
                                            data={[
                                                "dpmpp_2m", "dpmpp_2m_cfgpp", 
                                                "dpmpp_sde", "dpmpp_sde_cfgpp", 
                                                "euler", "euler_cfgpp",
                                                "euler_ancestral", "euler_ancestral_cfgpp"
                                            ]}
                                            value={settings.sampler}
                                            onChange={(v) => setSettings({ sampler: v || "dpmpp_sde_cfgpp" })}
                                        />
                                        <Select
                                            label="Scheduler"
                                            data={["karras", "exponential", "sgm_uniform", "simple", "normal", "ays"]}
                                            value={settings.scheduler}
                                            onChange={(v) => setSettings({ scheduler: v || "ays" })}
                                        />
                                    </Group>
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>

                        <Accordion.Item value="enhancements">
                            <Accordion.Control>Enhancements</Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    <Switch
                                        label="High Res Fix"
                                        checked={settings.hiresfix}
                                        onChange={(e) => setSettings({ hiresfix: e.currentTarget.checked })}
                                        disabled={caps && !caps.supports_hires_fix}
                                    />
                                    <Switch
                                        label="ADetailer"
                                        checked={settings.adetailer}
                                        onChange={(e) => setSettings({ adetailer: e.currentTarget.checked })}
                                    />
                                    <Switch
                                        label="Prompt Enhancer"
                                        checked={settings.enhance_prompt}
                                        onChange={(e) => setSettings({ enhance_prompt: e.currentTarget.checked })}
                                    />
                                    <Switch
                                        label="Live Preview"
                                        checked={settings.enable_preview}
                                        onChange={(e) => setSettings({ enable_preview: e.currentTarget.checked })}
                                    />
                                    <Select
                                        label="Preview Fidelity"
                                        data={[
                                            { value: 'low', label: 'Low (fast)' },
                                            { value: 'balanced', label: 'Balanced (default)' },
                                            { value: 'high', label: 'High (quality)' },
                                        ]}
                                        value={settings.preview_fidelity || 'balanced'}
                                        onChange={(v) => setSettings({ preview_fidelity: v || 'balanced' })}
                                        disabled={!settings.enable_preview}
                                    />
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>

                        <Accordion.Item value="refiner">
                            <Accordion.Control>Refiner</Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    <Select
                                        label="Refiner Model"
                                        placeholder="None"
                                        clearable
                                        data={modelOptions}
                                        value={settings.refiner_model_path}
                                        nothingFoundMessage="No models found"
                                        onChange={(v) => setSettings({ refiner_model_path: v || "" })}
                                        disabled={availableModels.find(m => m.path === settings.model_path)?.type !== "SDXL"}
                                    />
                                    <NumberInput
                                        label="Switch Step"
                                        value={settings.refiner_switch_step}
                                        onChange={(v) => setSettings({ refiner_switch_step: Number(v) })}
                                        min={1}
                                        disabled={!settings.refiner_model_path}
                                    />
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>

                        <Accordion.Item value="img2img">
                            <Accordion.Control>Image to Image</Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    <Switch
                                        label="Enable Img2Img"
                                        checked={settings.img2img_mode}
                                        onChange={(e) => setSettings({ img2img_mode: e.currentTarget.checked })}
                                        disabled={caps && !caps.supports_img2img}
                                    />
                                    {settings.img2img_mode && (
                                        <ImageInput
                                            label="Input Image"
                                            value={settings.img2img_image}
                                            onChange={(b64) => setSettings({ img2img_image: b64 || undefined })}
                                        />
                                    )}
                                    <NumberInput
                                        label="Denoising Strength"
                                        value={settings.img2img_denoise}
                                        onChange={(v) => setSettings({ img2img_denoise: Number(v) })}
                                        min={0} max={1} step={0.05}
                                        disabled={!settings.img2img_mode}
                                    />
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>

                        <Accordion.Item value="controlnet">
                            <Accordion.Control>ControlNet</Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    <Switch
                                        label="Enable ControlNet"
                                        checked={settings.controlnet_enabled}
                                        onChange={(e) => setSettings({ controlnet_enabled: e.currentTarget.checked })}
                                        disabled={caps && !caps.supports_controlnet}
                                    />
                                    {settings.controlnet_enabled && (
                                        <>
                                            <Select
                                                label="ControlNet Model"
                                                placeholder="Select model"
                                                data={controlNetOptions}
                                                value={settings.controlnet_model}
                                                nothingFoundMessage="No ControlNet models found"
                                                onChange={(v) => setSettings({ controlnet_model: v || undefined })}
                                            />
                                            <Text size="sm">Control Image (uses Img2Img input)</Text>
                                            {!settings.img2img_mode && (
                                                <ImageInput
                                                    label="Control Image"
                                                    value={settings.img2img_image}
                                                    onChange={(b64) => setSettings({ img2img_image: b64 || undefined })}
                                                />
                                            )}
                                            <Group>
                                                <Select
                                                    label="Type"
                                                    data={["canny", "depth", "pose", "softedge"]}
                                                    value={settings.controlnet_type}
                                                    onChange={(v) => setSettings({ controlnet_type: v || "canny" })}
                                                />
                                                <NumberInput
                                                    label="Strength"
                                                    value={settings.controlnet_strength}
                                                    onChange={(v) => setSettings({ controlnet_strength: Number(v) })}
                                                    min={0} max={2} step={0.1}
                                                />
                                            </Group>
                                        </>
                                    )}
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>

                        <Accordion.Item value="performance">
                            <Accordion.Control>Performance & Optimizations</Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    <Switch
                                        label="Stable Fast"
                                        checked={settings.stable_fast}
                                        onChange={(e) => setSettings({ stable_fast: e.currentTarget.checked })}
                                        disabled={caps && !caps.supports_stable_fast}
                                    />
                                    <Switch
                                        label="torch.compile"
                                        description="Compile diffusion model for faster inference (mutually exclusive with Stable Fast)"
                                        checked={settings.stable_fast ? false : (settings as any).torch_compile || false}
                                        onChange={(e) => setSettings({ torch_compile: e.currentTarget.checked, stable_fast: e.currentTarget.checked ? false : settings.stable_fast } as any)}
                                        disabled={settings.stable_fast}
                                    />
                                    <Switch
                                        label="FP8 Inference"
                                        description="Quantize weights to FP8 for lower VRAM usage (Ada Lovelace+ GPUs)"
                                        checked={(settings as any).fp8_inference || false}
                                        onChange={(e) => setSettings({ fp8_inference: e.currentTarget.checked } as any)}
                                    />
                                    <Switch label="Keep Models Loaded" checked={settings.keep_models_loaded} onChange={(e) => setSettings({ keep_models_loaded: e.currentTarget.checked })} />
                                    <Switch label="Reuse Seed" checked={settings.reuse_seed} onChange={(e) => setSettings({ reuse_seed: e.currentTarget.checked })} />

                                    <Group mt="xs">
                                        <Switch
                                            label="DeepCache"
                                            checked={settings.deepcache_enabled}
                                            onChange={(e) => setSettings({ deepcache_enabled: e.currentTarget.checked })}
                                            disabled={caps && !caps.supports_deepcache}
                                        />
                                        <Switch
                                            label="ToMe"
                                            checked={settings.tome_enabled}
                                            onChange={(e) => setSettings({ tome_enabled: e.currentTarget.checked })}
                                            disabled={caps && !caps.supports_tome}
                                        />
                                    </Group>
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>

                        <Accordion.Item value="multiscale">
                            <Accordion.Control>Multiscale Generation</Accordion.Control>
                            <Accordion.Panel>
                                <Stack gap="xs">
                                    <Switch
                                        label="Enable Multiscale"
                                        checked={settings.enable_multiscale}
                                        onChange={(e) => setSettings({ enable_multiscale: e.currentTarget.checked })}
                                    />
                                    {settings.enable_multiscale && (
                                        <>
                                            <Select
                                                label="Preset"
                                                data={["balanced", "detailed", "creative", "disabled"]}
                                                value={settings.multiscale_preset}
                                                onChange={(v) => setSettings({ multiscale_preset: v || "balanced" })}
                                            />
                                            <NumberInput label="Factor" value={settings.multiscale_factor} onChange={(v) => setSettings({ multiscale_factor: Number(v) })} step={0.1} min={0.1} max={1.0} />
                                        </>
                                    )}
                                </Stack>
                            </Accordion.Panel>
                        </Accordion.Item>

                    </Accordion>
                </Collapse>
            </Box>

            <Group onClick={toggleHistory} style={{ cursor: 'pointer' }} mb={5}>
                {openedHistory ? <IconCaretDown size={16} /> : <IconCaretRight size={16} />}
                <Text size="sm" fw={500}>Settings History</Text>
            </Group>
            <Collapse in={openedHistory}>
                <Stack gap="xs" p="xs">
                    <Group>
                        <Button
                            variant="light"
                            onClick={async () => {
                                try {
                                    const res = await postSettingsSnapshot(settings, !!settings.persist_prompt_history);
                                    if (res && res.snapshot) {
                                        appendSettingsSnapshot(res.snapshot);
                                    }
                                } catch (err) {
                                    console.error('Failed to save settings to history', err);
                                }
                            }}
                        >
                            Save to history
                        </Button>

                        <Switch
                            label="Include prompt in server history (opt-in)"
                            checked={!!settings.persist_prompt_history}
                            onChange={(e) => setSettings({ persist_prompt_history: e.currentTarget.checked })}
                        />
                    </Group>

                    <Select
                        label="Load from history"
                        placeholder="Select saved settings"
                        data={(settingsHistory || []).map(h => ({ value: h.id, label: new Date(h.ts * 1000).toLocaleString() }))}
                        onChange={(v) => {
                            const snap = (settingsHistory || []).find(s => s.id === v);
                            if (snap) setSettings(snap.settings as any);
                        }}
                    />
                </Stack>
            </Collapse>

        </Stack>
    );
}
