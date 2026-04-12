import { Images } from 'lucide-react';
import { ScrollArea, ScrollBar } from './ui/scroll-area';
import { useStore } from '../store/useStore';
import { cn } from '../lib/utils';
import { useShallow } from 'zustand/react/shallow';

export function Gallery() {
  const { currentImage, gallery, setCurrentImage } = useStore(useShallow((state) => ({
    currentImage: state.currentImage,
    gallery: state.gallery,
    setCurrentImage: state.setCurrentImage,
  })));

  return (
    <section className="studio-panel overflow-hidden rounded-[2rem] border border-line px-5 py-5 sm:px-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-muted">Recent frames</p>
          <h2 className="mt-1 font-serif text-2xl tracking-[-0.02em] text-ink">Recent</h2>
        </div>
        <div className="inline-flex items-center gap-2 rounded-full bg-sand px-3 py-1.5 text-xs text-muted">
          <Images className="h-3.5 w-3.5 text-clay" />
          {gallery.length === 0 ? 'Empty' : `${gallery.length} image${gallery.length === 1 ? '' : 's'}`}
        </div>
      </div>

      {gallery.length === 0 ? (
        <div className="mt-5 rounded-[1.5rem] border border-dashed border-line bg-oat/55 px-4 py-8 text-sm text-muted">
          Recent images appear here.
        </div>
      ) : (
        <ScrollArea className="mt-5 w-full whitespace-nowrap">
          <div className="flex gap-3 pb-3">
            {gallery.map((image, index) => {
              const isSelected = image === currentImage;

              return (
                <button
                  key={`${index}-${image.slice(0, 28)}`}
                  type="button"
                  onClick={() => setCurrentImage(image)}
                  className={cn(
                    'group relative w-28 shrink-0 overflow-hidden rounded-[1.4rem] border bg-paper text-left transition sm:w-32',
                    isSelected
                      ? 'border-clay shadow-[0_10px_24px_-18px_color-mix(in_oklab,var(--color-clay)_28%,transparent)]'
                      : 'border-line hover:-translate-y-0.5 hover:border-clay/35',
                  )}
                  aria-label={`Open image ${index + 1}`}
                >
                  <img
                    src={image}
                    alt={`Generated frame ${index + 1}`}
                    loading="lazy"
                    decoding="async"
                    className="h-28 w-full object-cover sm:h-32"
                  />
                  {isSelected ? <div className="absolute right-3 top-3 h-2.5 w-2.5 rounded-full bg-clay" /> : null}
                </button>
              );
            })}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      )}
    </section>
  );
}
