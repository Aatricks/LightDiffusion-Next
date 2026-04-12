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
    <section className="-mt-2 overflow-hidden rounded-b-[2rem] border border-line border-t-0 bg-paper/62 px-4 pb-3 pt-2 sm:px-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-serif text-[1.05rem] tracking-[-0.025em] text-ink">Recent</h2>
        <p className="text-xs text-muted">
          {gallery.length === 0 ? 'No saved frames yet' : `${gallery.length} saved`}
        </p>
      </div>

      {gallery.length === 0 ? (
        <div className="mt-4 rounded-[1.4rem] border border-dashed border-line bg-oat/45 px-4 py-6 text-sm text-muted">
          Generated images will collect here for quick comparison.
        </div>
      ) : (
        <ScrollArea className="mt-2.5 w-full whitespace-nowrap">
          <div className="flex gap-2 pb-2">
            {gallery.map((image, index) => {
              const isSelected = image === currentImage;

              return (
                <button
                  key={`${index}-${image.slice(0, 28)}`}
                  type="button"
                  onClick={() => setCurrentImage(image)}
                  className={cn(
                    'group relative w-[4.25rem] shrink-0 overflow-hidden rounded-[1rem] border bg-paper text-left transition sm:w-[4.9rem]',
                    isSelected
                      ? 'border-clay shadow-[0_10px_20px_-18px_color-mix(in_oklab,var(--color-clay)_28%,transparent)]'
                      : 'border-line hover:-translate-y-0.5 hover:border-clay/35',
                  )}
                  aria-label={`Open image ${index + 1}`}
                >
                  <img
                    src={image}
                    alt={`Generated frame ${index + 1}`}
                    loading="lazy"
                    decoding="async"
                    className="h-[4.25rem] w-full object-cover sm:h-[4.9rem]"
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
