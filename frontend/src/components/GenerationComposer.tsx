import { useMemo, useState } from 'react';
import { PanelRightOpen, Sparkles } from 'lucide-react';
import { useGenerationActions } from '../hooks/use-generation-actions';
import { getModelSelectionUpdates } from '../lib/settings';
import { useStore } from '../store/useStore';
import { Button } from './ui/button';
import { Label } from './ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { Textarea } from './ui/textarea';
import { useShallow } from 'zustand/react/shallow';

interface GenerationComposerProps {
  onOpenAdvanced: () => void;
}

export function GenerationComposer({ onOpenAdvanced }: GenerationComposerProps) {
  const { handleGenerate } = useGenerationActions();
  const { availableModels, serverStatus, setSettings, settings, status } = useStore(
    useShallow((state) => ({
      availableModels: state.availableModels,
      serverStatus: state.serverStatus,
      setSettings: state.setSettings,
      settings: state.settings,
      status: state.status,
    })),
  );
  const [negativePromptOpen, setNegativePromptOpen] = useState(false);

  const modelOptions = useMemo(
    () => availableModels.map((model) => ({ label: model.name, value: model.path })),
    [availableModels],
  );
  const activeModel = availableModels.find((model) => model.path === settings.model_path);

  const handleModelChange = (value: string) => {
    const selectedModel = availableModels.find((model) => model.path === value);
    if (!selectedModel) {
      setSettings({ model_path: value });
      return;
    }

    setSettings(getModelSelectionUpdates(selectedModel, availableModels));
  };

  return (
    <section className="relative overflow-hidden rounded-[2.1rem] border border-line bg-paper/97 px-5 py-5 shadow-[0_20px_48px_-36px_color-mix(in_oklab,var(--color-ink)_16%,transparent)] sm:px-7 sm:py-7">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-[radial-gradient(circle_at_top_left,color-mix(in_oklab,var(--color-oat)_86%,transparent),transparent_66%)]" />

      <div className="relative flex flex-col gap-6">
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] lg:items-start">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-muted">
              <Sparkles className="h-3.5 w-3.5 text-clay" />
              Generate
            </div>
            <div className="space-y-1.5">
              <h1 className="max-w-4xl font-serif text-[clamp(2.75rem,5.2vw,5rem)] leading-[0.92] tracking-[-0.055em] text-ink">
                Generate the next frame.
              </h1>
              <p className="max-w-3xl text-[15px] leading-6 text-muted">
                Choose a model, write the prompt, and run. Advanced controls stay out of the way until you need them.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2 pt-2">
              <Button variant="outline" size="sm" onClick={onOpenAdvanced}>
                <PanelRightOpen className="h-4 w-4" />
                Advanced
              </Button>
              {!serverStatus ? (
                <span className="rounded-full bg-sand px-3 py-1.5 text-xs text-clay-strong">Engine offline</span>
              ) : null}
            </div>
          </div>

          <div className="space-y-3 rounded-[1.75rem] border border-line/75 bg-canvas/34 p-4 lg:p-5">
            <Label>Model</Label>
            <Select onValueChange={handleModelChange} value={settings.model_path || undefined}>
              <SelectTrigger className="h-12 rounded-[1.2rem] border-line/80 bg-paper">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {modelOptions.length > 0 ? (
                  modelOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))
                ) : (
                  <div className="px-4 py-3 text-sm text-muted">No models available</div>
                )}
              </SelectContent>
            </Select>
            <p className="text-xs leading-5 text-muted">
              {activeModel ? `${activeModel.type} ready` : 'Models load automatically when the backend is available.'}
            </p>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_15rem] lg:items-stretch">
          <div className="space-y-3 rounded-[1.9rem] border border-line/65 bg-canvas/48 p-4 lg:p-5">
            <div className="flex items-center justify-between gap-3">
              <Label>Prompt</Label>
              <button
                type="button"
                onClick={() => setNegativePromptOpen((open) => !open)}
                className="text-xs font-medium text-muted transition hover:text-ink"
              >
                {negativePromptOpen ? 'Hide negative prompt' : 'Add negative prompt'}
              </button>
            </div>
            <Textarea
              className="min-h-[172px] rounded-[1.5rem] border-line/70 bg-paper px-5 py-4 text-[16px] leading-7 shadow-none"
              placeholder="Portrait lit by late afternoon sun, soft grain, calm expression, editorial composition..."
              value={settings.prompt}
              onChange={(event) => setSettings({ prompt: event.currentTarget.value })}
            />
            {negativePromptOpen ? (
              <div className="grid gap-2 border-t border-line/70 pt-3">
                <Label>Negative prompt</Label>
                <Textarea
                  className="min-h-[108px] rounded-[1.4rem] border-line/70 bg-paper shadow-none"
                  placeholder="Things to avoid: low detail, extra limbs, blown highlights..."
                  value={settings.negative_prompt}
                  onChange={(event) => setSettings({ negative_prompt: event.currentTarget.value })}
                />
              </div>
            ) : null}
          </div>

          <div className="flex flex-col gap-3 rounded-[1.75rem] border border-line/75 bg-ink/[0.04] p-4 lg:justify-between">
            <Label>Prompt</Label>
            <div className="space-y-2">
              <p className="font-serif text-[1.35rem] leading-tight tracking-[-0.03em] text-ink">
                Run the current prompt
              </p>
              <p className="text-xs leading-5 text-muted">
                {serverStatus
                  ? 'The next run uses the current prompt and advanced settings.'
                  : 'Reconnect the engine before generating.'}
              </p>
            </div>
            <Button
              className="h-14 w-full text-[15px]"
              size="lg"
              variant={status === 'generating' ? 'destructive' : 'default'}
              onClick={() => void handleGenerate()}
            >
              {status === 'generating' ? 'Interrupt' : 'Generate'}
            </Button>
          </div>
        </div>
        {status === 'error' ? (
          <p className="text-sm text-clay-strong">Generation failed. Check the backend logs, then try again.</p>
        ) : null}
      </div>
    </section>
  );
}
