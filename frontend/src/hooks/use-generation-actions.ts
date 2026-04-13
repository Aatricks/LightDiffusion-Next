import { startTransition, useCallback } from 'react';
import {
  generateImage,
  getImageMetadata,
  getLastSeed,
  interruptGeneration,
  postSettingsPreferences,
  postSettingsSnapshot,
} from '../api/client';
import type { GenerationSettings } from '../types';
import { getMetadataSettingsUpdates } from '../lib/settings';
import { useStore } from '../store/useStore';
import { useShallow } from 'zustand/react/shallow';

type ActionResult = {
  ok: boolean;
  message: string;
  warning?: string;
};

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      const result = event.target?.result;
      if (typeof result === 'string') {
        resolve(result);
      } else {
        reject(new Error('Unable to read image file.'));
      }
    };

    reader.onerror = () => reject(reader.error ?? new Error('Unable to read image file.'));
    reader.readAsDataURL(file);
  });
}

export function useGenerationActions() {
  const {
    settings,
    availableModels,
    status,
    setSettings,
    setStatus,
    setCurrentImage,
    addToGallery,
    addManyToGallery,
    appendSettingsSnapshot,
    setPreview,
  } = useStore(useShallow((state) => ({
    settings: state.settings,
    availableModels: state.availableModels,
    status: state.status,
    setSettings: state.setSettings,
    setStatus: state.setStatus,
    setCurrentImage: state.setCurrentImage,
    addToGallery: state.addToGallery,
    addManyToGallery: state.addManyToGallery,
    appendSettingsSnapshot: state.appendSettingsSnapshot,
    setPreview: state.setPreview,
  })));

  const importSettingsFromBase64 = useCallback(
    async (imageB64: string): Promise<ActionResult> => {
      try {
        const response = await getImageMetadata(imageB64);
        const updates = getMetadataSettingsUpdates(response?.metadata ?? {});

        setSettings(updates);

        const warning =
          updates.model_path && !availableModels.find((model) => model.path === updates.model_path)
            ? 'The imported model is not available locally.'
            : undefined;

        return {
          ok: true,
          message: 'Settings imported from image metadata.',
          warning,
        };
      } catch (error) {
        console.error('Failed to import settings from image', error);
        return {
          ok: false,
          message: 'Could not read settings from that image.',
        };
      }
    },
    [availableModels, setSettings],
  );

  const importSettingsFromFiles = useCallback(
    async (files: File[]): Promise<ActionResult> => {
      if (files.length === 0) {
        return {
          ok: false,
          message: 'Select an image to import settings.',
        };
      }

      try {
        const imageB64 = await readFileAsDataUrl(files[0]);
        return await importSettingsFromBase64(imageB64);
      } catch (error) {
        console.error('Failed to read image file', error);
        return {
          ok: false,
          message: 'Could not read that file.',
        };
      }
    },
    [importSettingsFromBase64],
  );

  const saveSettingsSnapshot = useCallback(async (): Promise<ActionResult> => {
    try {
      const response = await postSettingsSnapshot(settings, !!settings.persist_prompt_history);
      if (response?.snapshot) {
        appendSettingsSnapshot(response.snapshot);
      }

      return {
        ok: true,
        message: 'Settings saved to history.',
      };
    } catch (error) {
      console.error('Failed to save settings history', error);
      return {
        ok: false,
        message: 'Could not save settings history.',
      };
    }
  }, [appendSettingsSnapshot, settings]);

  const restoreLastSeed = useCallback(async (): Promise<ActionResult> => {
    try {
      const response = await getLastSeed();
      const seed = response?.seed ?? -1;
      setSettings({ seed: typeof seed === 'number' ? seed : -1 });

      return {
        ok: true,
        message: 'Seed restored from the last run.',
      };
    } catch (error) {
      console.error('Failed to fetch last seed', error);
      return {
        ok: false,
        message: 'Could not fetch the last seed.',
      };
    }
  }, [setSettings]);

  const updateAutotuneSettings = useCallback(
    async (
      updates: Pick<GenerationSettings, 'torch_compile' | 'vae_autotune'> & Partial<Pick<GenerationSettings, 'stable_fast'>>,
    ): Promise<ActionResult> => {
      const previous = useStore.getState().settings;
      setSettings(updates);

      try {
        await postSettingsPreferences({
          torch_compile: updates.torch_compile,
          vae_autotune: updates.vae_autotune,
        });
        return {
          ok: true,
          message: 'Autotune preferences saved.',
        };
      } catch (error) {
        console.error('Failed to save autotune preferences', error);
        setSettings({
          torch_compile: previous.torch_compile,
          vae_autotune: previous.vae_autotune,
          stable_fast: previous.stable_fast,
        });
        return {
          ok: false,
          message: 'Could not save autotune preferences.',
        };
      }
    },
    [setSettings],
  );

  const handleGenerate = useCallback(async () => {
    if (status === 'generating') {
      try {
        await interruptGeneration();
      } finally {
        setStatus('idle');
      }

      return;
    }

    setStatus('generating');
    setPreview(null);

    const localSnapshot = {
      id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`,
      ts: Math.floor(Date.now() / 1000),
      settings: { ...settings },
    };
    appendSettingsSnapshot(localSnapshot);

    try {
      const response = await generateImage(settings);
      const images = response.images ?? (response.image ? [response.image] : []);
      const image = images[0] ?? null;

      if (image) {
        startTransition(() => {
          setCurrentImage(image);
          if (images.length === 1) {
            addToGallery(image);
          } else {
            addManyToGallery(images);
          }
        });
      }
    } catch (error) {
      console.error('Generation failed', error);
      setStatus('error');
      return;
    }

    setStatus('idle');
  }, [addManyToGallery, addToGallery, appendSettingsSnapshot, setCurrentImage, setPreview, setStatus, settings, status]);

  return {
    handleGenerate,
    importSettingsFromBase64,
    importSettingsFromFiles,
    saveSettingsSnapshot,
    restoreLastSeed,
    updateAutotuneSettings,
  };
}
