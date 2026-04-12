import axios from 'axios';
import type { GenerationSettings, GenerationResponse, ModelInfo, SettingsSnapshot, ImageMetadata } from '../types';

const api = axios.create({
    baseURL: '/api', // Proxy handles redirection to localhost:7861
});

export const listModels = async (): Promise<ModelInfo[]> => {
    const res = await api.get<ModelInfo[]>('/models');
    return res.data;
};

export const listControlNets = async (): Promise<{ models: string[] }> => {
    const res = await api.get<{ models: string[] }>('/controlnets');
    return res.data;
};

export const generateImage = async (settings: GenerationSettings): Promise<GenerationResponse> => {
    const res = await api.post<GenerationResponse>('/generate', settings);
    console.log("Generation response:", res.data);
    return res.data;
};

export const interruptGeneration = async (): Promise<void> => {
    await api.post('/interrupt');
};

export const getLastSeed = async (): Promise<{ seed: number | null }> => {
    const res = await api.get('/settings/last');
    return res.data;
};

export const getSettingsHistory = async (): Promise<{ history: SettingsSnapshot[] }> => {
    const res = await api.get('/settings/history');
    return res.data;
};

export const postSettingsSnapshot = async (settings: GenerationSettings, include_prompt: boolean = false): Promise<{ snapshot: SettingsSnapshot }> => {
    const res = await api.post('/settings/history', { settings, include_prompt });
    return res.data;
};

export const getImageMetadata = async (imageB64: string): Promise<{ metadata: ImageMetadata }> => {
    const res = await api.post('/images/metadata', { image: imageB64 });
    return res.data;
};

export const getTelemetry = async (): Promise<Record<string, unknown>> => {
    const res = await api.get('/telemetry');
    return res.data;
}

export default api;
