import { Group, Text, Stack, rem } from '@mantine/core';
import { IconUpload, IconPhoto, IconX } from '@tabler/icons-react';
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone';

interface ImageInputProps {
    value?: string | null;
    onChange: (base64: string | null) => void;
    label?: string;
    [key: string]: any; // Allow other props
}

export function ImageInput({ value, onChange, label, ...props }: ImageInputProps) {
    const handleDrop = (files: File[]) => {
        const file = files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            onChange(e.target?.result as string);
        };
        reader.readAsDataURL(file);
    };

    return (
        <Stack gap="xs">
            {label && <Text size="sm" fw={500}>{label}</Text>}
            <Dropzone
                onDrop={handleDrop}
                onReject={(files) => console.log('rejected files', files)}
                maxSize={5 * 1024 ** 2}
                accept={IMAGE_MIME_TYPE}
                {...props}
                style={{
                    border: value ? 'none' : undefined,
                    padding: value ? 0 : undefined,
                    overflow: 'hidden',
                }}
            >
                {value ? (
                    <div style={{ position: 'relative', width: '100%', height: 200 }}>
                        <img
                            src={value}
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                            alt="Input"
                        />
                        <div
                            style={{
                                position: 'absolute',
                                top: 5,
                                right: 5,
                                background: 'rgba(0,0,0,0.5)',
                                borderRadius: '50%',
                                padding: 5,
                                cursor: 'pointer'
                            }}
                            onClick={(e) => {
                                e.stopPropagation();
                                onChange(null);
                            }}
                        >
                            <IconX size={16} color="white" />
                        </div>
                    </div>
                ) : (
                    <Group justify="center" gap="xl" style={{ minHeight: rem(120), pointerEvents: 'none' }}>
                        <Dropzone.Accept>
                            <IconUpload
                                style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-blue-6)' }}
                                stroke={1.5}
                            />
                        </Dropzone.Accept>
                        <Dropzone.Reject>
                            <IconX
                                style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-red-6)' }}
                                stroke={1.5}
                            />
                        </Dropzone.Reject>
                        <Dropzone.Idle>
                            <IconPhoto
                                style={{ width: rem(52), height: rem(52), color: 'var(--mantine-color-dimmed)' }}
                                stroke={1.5}
                            />
                        </Dropzone.Idle>

                        <div>
                            <Text size="xl" inline>
                                Drag images here or click to select
                            </Text>
                            <Text size="sm" c="dimmed" inline mt={7}>
                                Attach as many files as you like, each file should not exceed 5mb
                            </Text>
                        </div>
                    </Group>
                )}
            </Dropzone>
        </Stack>
    );
}
