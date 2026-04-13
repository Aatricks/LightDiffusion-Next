import { useState } from 'react';
import { GenerationComposer } from './components/GenerationComposer';
import { GenerationSettings } from './components/GenerationSettings';
import { Gallery } from './components/Gallery';
import { ImagePreview } from './components/ImagePreview';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from './components/ui/sheet';
import { useGenerationBootstrap } from './hooks/use-generation-bootstrap';
import { useMediaQuery } from './hooks/use-media-query';

export default function App() {
  useGenerationBootstrap();

  const [controlsOpen, setControlsOpen] = useState(false);
  const isDesktop = useMediaQuery('(min-width: 1024px)');
  const controlSide = isDesktop ? 'right' : 'bottom';

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <div className="page-halo pointer-events-none absolute inset-x-0 top-0 h-96" />

      <main className="page-fade relative mx-auto flex min-h-screen w-full max-w-[1320px] flex-col px-4 pb-10 pt-4 sm:px-6">
        <section className="mx-auto min-h-0 w-full max-w-[1200px] space-y-4 sm:space-y-5">
          <GenerationComposer onOpenAdvanced={() => setControlsOpen(true)} />
          <ImagePreview />
          <Gallery />
        </section>
      </main>

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
            <SheetTitle>Advanced controls</SheetTitle>
            <SheetDescription>
              Sampling, conditioning, optimization, and history for the next run.
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
