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
    <section className="overflow-hidden rounded-t-[2.25rem] border border-line border-b-0 bg-paper/90 p-2 shadow-[0_18px_42px_-36px_color-mix(in_oklab,var(--color-ink)_18%,transparent)] sm:p-3">
      <div className="studio-grid relative min-h-[460px] overflow-hidden rounded-[1.7rem] p-2 sm:min-h-[680px] sm:p-4">
        {isGenerating ? (
          <div className="absolute inset-x-4 top-4 z-10 flex items-center justify-between sm:inset-x-6">
            <div className="rounded-full border border-line bg-paper/92 px-3 py-1.5 text-[11px] uppercase tracking-[0.16em] text-muted">
              Generating
            </div>
            <div className="rounded-full border border-line bg-paper/92 px-3 py-1.5 text-xs text-muted">
              {stepText}
            </div>
          </div>
        ) : null}

        <div className="flex h-full items-center justify-center">
          {displayImage ? (
            <img
              src={displayImage}
              alt="Generated preview"
              className="max-h-[calc(100vh-10rem)] w-auto max-w-full rounded-[1.15rem] object-contain shadow-[0_16px_30px_-24px_color-mix(in_oklab,var(--color-ink)_18%,transparent)]"
            />
          ) : (
            <div className="flex max-w-lg flex-col items-center justify-center gap-5 px-4 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full border border-line bg-paper text-clay">
                {isGenerating ? <LoaderCircle className="h-6 w-6 animate-spin" /> : <ImagePlus className="h-6 w-6" />}
              </div>
              <div className="space-y-2">
                <p className="font-serif text-[clamp(1.8rem,3vw,2.5rem)] tracking-[-0.035em] text-ink">
                  {isGenerating ? 'Preparing the next frame' : 'Ready to generate'}
                </p>
                <p className="text-sm leading-6 text-muted">
                  {isGenerating
                    ? 'Live previews appear here as the run progresses.'
                    : 'Choose a model, write a prompt, then generate your first frame.'}
                </p>
              </div>
            </div>
          )}
        </div>

        {isGenerating ? (
          <div className="pointer-events-none absolute inset-x-4 bottom-4 sm:inset-x-6">
            <div className="rounded-[1.35rem] border border-line bg-paper/94 p-3">
              <div className="mb-2 flex items-center justify-between text-xs text-muted">
                <span>Progress</span>
                <span>{Math.round(progressValue)}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-sand">
                <div
                  className="h-full rounded-full bg-clay transition-[width] duration-300"
                  style={{ width: `${progressValue}%` }}
                />
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {displayImage ? (
        <div className="mt-1 flex justify-end pr-1">
          <Button variant="ghost" size="sm" className="text-muted hover:text-ink" onClick={() => void handleImportFromPreview()}>
            <Import className="h-4 w-4" />
            Import settings from image
          </Button>
        </div>
      ) : null}

      {feedback ? (
        <p
          className={cn(
            'mt-3 text-sm',
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
    </section>
  );
}
