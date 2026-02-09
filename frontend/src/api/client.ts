import axios from 'axios';
import type { GenerationSettings, GenerationResponse, ModelInfo } from '../types';

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

export const getTelemetry = async (): Promise<any> => {
    const res = await api.get('/telemetry');
    return res.data;
}

export default api;
