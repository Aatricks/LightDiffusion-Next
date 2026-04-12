import { useEffect } from 'react';
import { getSettingsPreferences, listControlNets, listModels } from '../api/client';
import { useStore } from '../store/useStore';
import { getDefaultModel, getModelSelectionUpdates } from '../lib/settings';
import { useShallow } from 'zustand/react/shallow';

export function useGenerationBootstrap() {
  const { availableModels, setModels, setControlNets, setSettings } = useStore(useShallow((state) => ({
    availableModels: state.availableModels,
    setModels: state.setModels,
    setControlNets: state.setControlNets,
    setSettings: state.setSettings,
  })));

  useEffect(() => {
    let cancelled = false;

    const loadModels = async () => {
      try {
        const models = await listModels();
        if (cancelled) return;

        setModels(models);

        const currentSettings = useStore.getState().settings;
        if (!currentSettings.model_path && models.length > 0) {
          const defaultModel = getDefaultModel(models);
          if (defaultModel) {
            setSettings(getModelSelectionUpdates(defaultModel, models));
          }
        }
      } catch (error) {
        console.error('Failed to load models', error);
      }
    };

    const loadControlNets = async () => {
      try {
        const response = await listControlNets();
        if (!cancelled) {
          setControlNets(response.models);
        }
      } catch (error) {
        console.error('Failed to load ControlNet models', error);
      }
    };

    const loadPreferences = async () => {
      try {
        const preferences = await getSettingsPreferences();
        if (!cancelled) {
          setSettings({
            torch_compile: !!preferences.torch_compile,
            vae_autotune: !!preferences.vae_autotune,
          });
        }
      } catch (error) {
        console.error('Failed to load settings preferences', error);
      }
    };

    if (availableModels.length === 0) {
      void loadModels();
    }

    void loadControlNets();
    void loadPreferences();

    return () => {
      cancelled = true;
    };
  }, [availableModels.length, setControlNets, setModels, setSettings]);
}
