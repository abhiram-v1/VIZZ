const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  async getAvailableAlgorithms() {
    return this.request('/algorithms');
  }

  async getDatasetPreview(rows = 10, dataset = "stroke") {
    return this.request(`/dataset/preview?rows=${rows}&dataset=${dataset}`);
  }

  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);

    return fetch(`${API_BASE}/dataset/upload`, {
      method: 'POST',
      body: formData,
    }).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    });
  }

  async predictStroke(data) {
    return this.request('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

export default new ApiService();
