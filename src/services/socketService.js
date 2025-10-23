import { io } from 'socket.io-client';

class SocketService {
  constructor() {
    this.socket = null;
    this.listeners = new Map();
  }

  connect() {
    if (!this.socket) {
      this.socket = io('http://localhost:8000', {
        transports: ['websocket', 'polling'],
        timeout: 20000,
        forceNew: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
      });

      this.socket.on('connect', () => {
        console.log('Connected to server');
        this.emit('connected');
      });

      this.socket.on('disconnect', (reason) => {
        console.log('Disconnected from server:', reason);
        this.emit('disconnected');
      });

      this.socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        this.emit('error', { message: 'Connection failed' });
      });
    }
    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  emit(event, data) {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);

    if (this.socket) {
      this.socket.on(event, callback);
    } else {
      // If socket not connected yet, store the listener for when it connects
      this.connect();
    }
  }

  off(event, callback) {
    const listeners = this.listeners.get(event);
    if (listeners) {
      const index = listeners.indexOf(callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }

    if (this.socket) {
      this.socket.off(event, callback);
    }
  }

  startTraining(algorithm, params = {}) {
    this.connect();
    if (this.socket) {
      this.socket.emit('start_training', { algorithm, params });
    } else {
      console.warn('Socket not connected, but attempting to start training...');
      // Try again after a short delay
      setTimeout(() => {
        if (this.socket) {
          this.socket.emit('start_training', { algorithm, params });
        }
      }, 1000);
    }
  }
}

export default new SocketService();
