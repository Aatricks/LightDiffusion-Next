import { useCallback, useEffect, useRef, useState } from 'react';
import { ImagePlus, Import, LoaderCircle } from 'lucide-react';
import useWebSocket from 'react-use-websocket';
import { useGenerationActions } from '../hooks/use-generation-actions';
import { useStore } from '../store/useStore';
import type { PreviewMessage } from '../types';
import { Button } from './ui/button';
import { cn } from '../lib/utils';
import { useShallow } from 'zustand/react/shallow';

type FeedbackState = {
  tone: 'success' | 'warning' | 'error';
  text: string;
};

export function ImagePreview() {
  const { importSettingsFromBase64 } = useGenerationActions();
  const { currentImage, preview, setPreview, setServerStatus, status } = useStore(useShallow((state) => ({
    currentImage: state.currentImage,
    preview: state.preview,
    setPreview: state.setPreview,
    setServerStatus: state.setServerStatus,
    status: state.status,
  })));
  const [activePreviewImage, setActivePreviewImage] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<FeedbackState | null>(null);
  const currentGenerationIdRef = useRef<string | null>(null);
  const lastStepRef = useRef(-1);

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${protocol}://${window.location.host}/ws/preview`;

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data) as PreviewMessage;

        if (message.type === 'generation_start' && message.generation_id) {
          currentGenerationIdRef.current = message.generation_id;
          lastStepRef.current = -1;
          setActivePreviewImage(null);
          setPreview(null);
          return;
        }

        if (
          message.generation_id &&
          currentGenerationIdRef.current &&
          message.generation_id !== currentGenerationIdRef.current
        ) {
          return;
        }

        if (message.step !== undefined) {
          if (message.step < lastStepRef.current && message.step !== 0) {
            return;
          }
          lastStepRef.current = message.step;
        }

        if (message.images && message.images.length > 0) {
          setActivePreviewImage(message.images[0]);
        }

        setPreview(message);
      } catch (error) {
        console.error('Failed to parse websocket message', error);
      }
    },
    [setPreview],
  );

  useWebSocket(wsUrl, {
    shouldReconnect: () => true,
    reconnectInterval: 3000,
    onOpen: () => setServerStatus(true),
    onClose: () => setServerStatus(false),
    onError: () => setServerStatus(false),
    onMessage: handleMessage,
  });

  useEffect(() => {
    lastStepRef.current = -1;
  }, [status]);

  useEffect(() => {
    if (status === 'idle') {
      currentGenerationIdRef.current = null;
    }
  }, [status]);

  const isGenerating = status === 'generating';
  const displayImage = isGenerating ? (preview ? activePreviewImage : null) : currentImage;
  const progressValue =
    isGenerating && preview?.step !== undefined && preview.total_steps
      ? (preview.step / preview.total_steps) * 100
      : 0;
  const stepText =
    isGenerating && preview?.step !== undefined && preview.total_steps
      ? `Step ${preview.step} / ${preview.total_steps}`
      : isGenerating
        ? 'Generating...'
        : 'Idle';

  const handleImportFromPreview = async () => {
    if (!displayImage) return;

    const result = await importSettingsFromBase64(displayImage);
    setFeedback({
      tone: result.ok ? (result.warning ? 'warning' : 'success') : 'error',
      text: result.warning ? `${result.message} ${result.warning}` : result.message,
    });
  };

  return (
    <section className="studio-panel overflow-hidden rounded-[2.25rem] border border-line p-4 sm:p-6">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-muted">Main canvas</p>
            <h2 className="mt-1 font-serif text-[2rem] tracking-[-0.03em] text-ink">Preview</h2>
          </div>

          {displayImage ? (
            <Button variant="outline" size="sm" onClick={() => void handleImportFromPreview()}>
              <Import className="h-4 w-4" />
              Import settings from image
            </Button>
          ) : null}
        </div>

        <div className="studio-grid relative min-h-[420px] overflow-hidden rounded-[1.9rem] border border-line p-4 sm:min-h-[560px] sm:p-6">
          <div className="absolute inset-x-0 top-0 flex items-center justify-between px-4 py-4 sm:px-6">
            <div className="rounded-full bg-paper px-3 py-1.5 text-[11px] uppercase tracking-[0.16em] text-muted">
              {isGenerating ? 'Live' : 'Frame'}
            </div>
            <div className="rounded-full bg-paper px-3 py-1.5 text-xs text-muted">{stepText}</div>
          </div>

          <div className="flex h-full items-center justify-center">
            {displayImage ? (
              <img
                src={displayImage}
                alt="Generated preview"
                className="max-h-[calc(100vh-16rem)] w-auto max-w-full rounded-[1.5rem] object-contain shadow-[0_14px_32px_-24px_color-mix(in_oklab,var(--color-ink)_18%,transparent)]"
              />
            ) : (
              <div className="flex max-w-md flex-col items-center justify-center gap-4 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-paper text-clay shadow-[0_10px_24px_-18px_color-mix(in_oklab,var(--color-clay)_28%,transparent)]">
                  {isGenerating ? <LoaderCircle className="h-6 w-6 animate-spin" /> : <ImagePlus className="h-6 w-6" />}
                </div>
                <div className="space-y-2">
                  <p className="font-medium text-ink">{isGenerating ? 'Preparing preview' : 'No image yet'}</p>
                  <p className="text-sm text-muted">Generate to fill the canvas.</p>
                </div>
              </div>
            )}
          </div>

          <div className="pointer-events-none absolute inset-x-4 bottom-4 sm:inset-x-6">
            <div className="rounded-[1.4rem] border border-line bg-paper p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-muted">
                <span>{isGenerating ? 'Progress' : 'State'}</span>
                <span>{isGenerating ? `${Math.round(progressValue)}%` : displayImage ? 'Ready' : 'Waiting'}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-sand">
                <div
                  className={cn(
                    'h-full rounded-full bg-clay transition-[width] duration-300',
                    isGenerating ? 'opacity-100' : 'opacity-50',
                  )}
                  style={{ width: `${isGenerating ? progressValue : displayImage ? 100 : 12}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {feedback ? (
          <p
            className={cn(
              'text-sm',
              feedback.tone === 'error'
                ? 'text-clay-strong'
                : feedback.tone === 'warning'
                  ? 'text-muted'
                  : 'text-clay',
            )}
          >
            {feedback.text}
          </p>
        ) : null}
      </div>
    </section>
  );
}
