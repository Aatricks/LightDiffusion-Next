import { useCallback } from 'react';
import { ImageUp, Upload, X } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { Button } from './ui/button';
import { cn } from '../lib/utils';

interface ImageInputProps {
  value?: string | null;
  onChange: (base64: string | null) => void;
  label?: string;
  description?: string;
  className?: string;
  compact?: boolean;
}

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

export function ImageInput({ value, onChange, label, description, className, compact = false }: ImageInputProps) {
  const handleDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      try {
        const base64 = await readFileAsDataUrl(file);
        onChange(base64);
      } catch (error) {
        console.error('Failed to read dropped image', error);
      }
    },
    [onChange],
  );

  const { getRootProps, getInputProps, isDragAccept, isDragReject } = useDropzone({
    accept: { 'image/*': [] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
    multiple: false,
    onDrop: (acceptedFiles) => {
      void handleDrop(acceptedFiles);
    },
  });

  return (
    <div className={cn('space-y-3', className)}>
      {label ? (
        <div className="space-y-1">
          <p className="text-sm font-medium text-ink">{label}</p>
          {description ? <p className="text-xs leading-5 text-muted">{description}</p> : null}
        </div>
      ) : null}

      {value ? (
        <div className="overflow-hidden rounded-[1.5rem] border border-line bg-paper">
          <div className={cn('relative overflow-hidden bg-sand/45', compact ? 'h-40' : 'h-52')}>
            <img src={value} alt="Selected input" className="h-full w-full object-contain p-4" />
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="absolute right-3 top-3 h-9 w-9 bg-paper/92"
              onClick={() => onChange(null)}
            >
              <X className="h-4 w-4" />
              <span className="sr-only">Remove image</span>
            </Button>
          </div>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={cn(
            'rounded-[1.5rem] border border-dashed bg-oat/60 px-4 py-6 transition',
            compact ? 'min-h-36' : 'min-h-40',
            isDragAccept && 'border-clay bg-clay/8',
            isDragReject && 'border-clay-strong bg-clay/10',
            'cursor-pointer border-line hover:border-clay/45 hover:bg-oat',
          )}
        >
          <input {...getInputProps()} />
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-paper text-clay shadow-[0_12px_30px_-22px_color-mix(in_oklab,var(--color-clay)_50%,transparent)]">
              {isDragAccept ? <Upload className="h-5 w-5" /> : <ImageUp className="h-5 w-5" />}
            </div>
            <div className="space-y-1">
              <p className="text-sm font-medium text-ink">Drop an image or click to browse</p>
              <p className="text-xs leading-5 text-muted">PNG, JPG, or WEBP up to 10 MB.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
