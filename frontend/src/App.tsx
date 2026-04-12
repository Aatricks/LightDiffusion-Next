import { useState } from 'react';
import { PanelRightOpen, Sparkles, Wifi, WifiOff } from 'lucide-react';
import { GenerationSettings } from './components/GenerationSettings';
import { Gallery } from './components/Gallery';
import { ImagePreview } from './components/ImagePreview';
import { Button } from './components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from './components/ui/sheet';
import { useGenerationActions } from './hooks/use-generation-actions';
import { useGenerationBootstrap } from './hooks/use-generation-bootstrap';
import { useMediaQuery } from './hooks/use-media-query';
import { useStore } from './store/useStore';
import { useShallow } from 'zustand/react/shallow';

function ConnectionBadge() {
  const serverStatus = useStore((state) => state.serverStatus);

  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-line bg-paper/82 px-3 py-2 text-xs font-medium text-muted">
      {serverStatus ? <Wifi className="h-3.5 w-3.5 text-clay" /> : <WifiOff className="h-3.5 w-3.5 text-muted" />}
      <span>{serverStatus ? 'Connected' : 'Engine offline'}</span>
    </div>
  );
}

export default function App() {
  useGenerationBootstrap();

  const [controlsOpen, setControlsOpen] = useState(false);
  const isDesktop = useMediaQuery('(min-width: 1024px)');
  const { handleGenerate } = useGenerationActions();
  const { availableModels, settings, status } = useStore(useShallow((state) => ({
    availableModels: state.availableModels,
    settings: state.settings,
    status: state.status,
  })));

  const activeModel = availableModels.find((model) => model.path === settings.model_path);
  const controlSide = isDesktop ? 'right' : 'bottom';

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <div className="page-halo pointer-events-none absolute inset-x-0 top-0 h-72" />

      {isDesktop ? (
        <div className="fixed right-5 top-5 z-40 flex items-center gap-2 rounded-[1.4rem] border border-line bg-paper p-2 shadow-[0_14px_32px_-24px_color-mix(in_oklab,var(--color-ink)_18%,transparent)]">
          <ConnectionBadge />
          <Button variant="outline" size="sm" onClick={() => setControlsOpen(true)}>
            <PanelRightOpen className="h-4 w-4" />
            Controls
          </Button>
          <Button
            size="sm"
            variant={status === 'generating' ? 'destructive' : 'default'}
            onClick={() => void handleGenerate()}
          >
            {status === 'generating' ? 'Interrupt' : 'Generate'}
          </Button>
        </div>
      ) : null}

      <main className="page-fade relative mx-auto flex min-h-screen w-full max-w-[1080px] flex-col px-4 pb-28 pt-6 sm:px-6 lg:pb-10">
        <div className="mb-4 flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-muted">
          <Sparkles className="h-3.5 w-3.5 text-clay" />
          Workspace
        </div>

        <section className="mx-auto min-h-0 w-full max-w-[980px] space-y-6">
          <ImagePreview />
          <Gallery />
        </section>
      </main>

      {!isDesktop ? (
        <div className="fixed inset-x-3 bottom-3 z-40 rounded-[1.75rem] border border-line bg-paper p-3 shadow-[0_12px_28px_-22px_color-mix(in_oklab,var(--color-ink)_18%,transparent)]">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-ink">
                {activeModel ? activeModel.name : 'Choose a model in controls'}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => setControlsOpen(true)}>
                <PanelRightOpen className="h-4 w-4" />
                Controls
              </Button>
              <Button
                variant={status === 'generating' ? 'destructive' : 'default'}
                size="sm"
                onClick={() => void handleGenerate()}
              >
                {status === 'generating' ? 'Interrupt' : 'Generate'}
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      <Sheet open={controlsOpen} onOpenChange={setControlsOpen}>
        <SheetContent
          side={controlSide}
          className={
            isDesktop
              ? 'h-[calc(100vh-2rem)] w-[26rem] overflow-hidden sm:max-w-none'
              : 'h-[min(88vh,860px)] overflow-hidden'
          }
        >
          <SheetHeader>
            <SheetTitle>{isDesktop ? 'Controls' : 'Generation controls'}</SheetTitle>
            <SheetDescription>
              {isDesktop ? 'Model, prompt, and advanced controls.' : 'Full controls for small screens.'}
            </SheetDescription>
          </SheetHeader>
          <div className="mt-4 h-[calc(100%-4rem)] min-h-0">
            <GenerationSettings />
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}
