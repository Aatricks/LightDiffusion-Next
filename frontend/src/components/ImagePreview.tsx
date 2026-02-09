import { useEffect, useState, useRef, useCallback } from 'react';
import useWebSocket from 'react-use-websocket';
import { useStore } from '../store/useStore';
import { Center, Image, Stack, Text, Progress, Paper, Group } from '@mantine/core';
import { IconPhoto } from '@tabler/icons-react';
import type { PreviewMessage } from '../types';

export function ImagePreview() {
    const { preview, setPreview, status, currentImage, setServerStatus } = useStore();
    const [activePreviewImage, setActivePreviewImage] = useState<string | null>(null);
    const lastStepRef = useRef(-1);

    // Connect to WebSocket via Vite proxy or direct relative URL
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.host}/ws/preview`;

    const handleMessage = useCallback((event: MessageEvent) => {
        try {
            const msg = JSON.parse(event.data) as PreviewMessage;

            // Enforce monotonic progress
            if (msg.step !== undefined) {
                // Heuristic: Ignore high step counts at start (ghosts)
                if (lastStepRef.current === -1 && msg.step > 2) {
                    return;
                }

                if (msg.step < lastStepRef.current) {
                    // Allow step 0 to reset if we missed a status change or it's a restart
                    if (msg.step === 0) {
                        lastStepRef.current = 0;
                    } else {
                        // Ignore out-of-order message
                        return;
                    }
                } else {
                    lastStepRef.current = msg.step;
                }
            }

            // Persist latest preview image locally immediately
            if (msg.images && msg.images.length > 0) {
                setActivePreviewImage(msg.images[0]);
            }

            setPreview(msg);
        } catch (e) {
            console.error("Failed to parse websocket message", e);
        }
    }, [setPreview]);

    useWebSocket(wsUrl, {
        shouldReconnect: () => true,
        reconnectInterval: 3000,
        onOpen: () => setServerStatus(true),
        onClose: () => setServerStatus(false),
        onError: () => setServerStatus(false),
        onMessage: handleMessage
    });

    useEffect(() => {
        if (status === 'generating') {
            // Reset step counter and clear old preview on new generation
            lastStepRef.current = -1;
            setActivePreviewImage(null);
        } else {
            lastStepRef.current = -1;
        }
    }, [status]);

    // Reset active preview when idle
    useEffect(() => {
        if (status === 'idle') {
            setActivePreviewImage(null);
        }
    }, [status]);



    // Display logic
    const isGenerating = status === 'generating';

    // Use preview image if generating and available, otherwise currentImage
    let displayImage = currentImage;
    if (isGenerating && activePreviewImage) {
        displayImage = activePreviewImage;
    }

    let progressValue = 0;
    let stepText = '';

    if (isGenerating && preview) {
        if (preview.step !== undefined && preview.total_steps !== undefined && preview.total_steps > 0) {
            progressValue = (preview.step / preview.total_steps) * 100;
            stepText = `Step ${preview.step} / ${preview.total_steps}`;
        }
    }

    return (
        <Paper shadow="sm" p="md" radius="md" h="100%" withBorder>
            <Stack h="100%" justify="center">
                {displayImage ? (
                    <Center style={{ width: '100%', height: '100%', minHeight: undefined }}>
                        <Image
                            src={displayImage}
                            alt="Preview"
                            fit="contain"
                            radius="md"
                            style={{
                                maxHeight: 'calc(100vh - 200px)',
                                maxWidth: '100%',
                                objectFit: 'contain'
                            }}
                        />
                    </Center>
                ) : (
                    <Center h={300}>
                        <Stack align="center" gap="xs" c="dimmed">
                            <IconPhoto size={48} />
                            <Text>No image generated yet</Text>
                        </Stack>
                    </Center>
                )}

                {isGenerating && (
                    <Stack gap="xs" mt="md">
                        <Group justify="space-between">
                            <Text size="sm">{stepText || "Generating..."}</Text>
                            <Text size="sm">{Math.round(progressValue)}%</Text>
                        </Group>
                        <Progress value={progressValue} animated striped size="lg" radius="xl" />
                    </Stack>
                )}
            </Stack>
        </Paper >
    );
}

