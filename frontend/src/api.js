// frontend/src/api.js
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const getBotConfig = async () => {
    try {
        const res = await axios.get(`${API_URL}/config`);
        return res.data;
    } catch (error) {
        console.error("Failed to fetch config", error);
        return null;
    }
};

export const updateBotConfig = async (updates) => {
    try {
        const res = await axios.post(`${API_URL}/config`, updates);
        return res.data;
    } catch (error) {
        console.error("Failed to update config", error);
    }
};

// Add this to your existing api.js
export const getTradeHistory = async () => {
    try {
        const res = await axios.get(`${API_URL}/trades`);
        return res.data;
    } catch (error) {
        console.error("Failed to fetch history", error);
        return [];
    }
};