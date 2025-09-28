// frontend/src/hooks/useWebSocket.ts

import { useState, useEffect, useRef, useCallback } from 'react';

type ReadyState = 'connecting' | 'open' | 'closing' | 'closed';

interface WebSocketMessage {
  type: string;
  data: unknown;
  timestamp: number;
}

/**
 * Manages a WebSocket connection and provides the latest message and connection state.
 * @param {string} url - The WebSocket URL to connect to.
 * @param {object} options - Configuration options for the WebSocket connection.
 */
export function useWebSocket(
  url: string,
  options: {
    reconnect?: boolean;
    reconnectInterval?: number;
    maxReconnectAttempts?: number;
  } = {}
) {
  const {
    reconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5
  } = options;

  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [readyState, setReadyState] = useState<ReadyState>('connecting');
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!url) return;

    try {
      const socket = new WebSocket(url);
      ws.current = socket;
      setConnectionError(null);

      socket.onopen = () => {
        setReadyState('open');
        reconnectCount.current = 0;
      };

      socket.onclose = (event) => {
        setReadyState('closed');

        // Attempt to reconnect if enabled and we haven't exceeded max attempts
        if (reconnect && reconnectCount.current < maxReconnectAttempts && !event.wasClean) {
          reconnectCount.current += 1;
          setReadyState('connecting');

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      socket.onerror = () => {
        setConnectionError('WebSocket connection failed');
        setReadyState('closed');
      };

      socket.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          setLastMessage({
            type: parsedData.type || 'message',
            data: parsedData,
            timestamp: Date.now()
          });
        } catch (error) {
          // Handle non-JSON messages
          setLastMessage({
            type: 'raw',
            data: event.data,
            timestamp: Date.now()
          });
        }
      };
    } catch (error) {
      setConnectionError('Failed to create WebSocket connection');
      setReadyState('closed');
    }
  }, [url, reconnect, reconnectInterval, maxReconnectAttempts]);

  const sendMessage = useCallback((message: unknown) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      const messageString = typeof message === 'string'
        ? message
        : JSON.stringify(message);
      ws.current.send(messageString);
      return true;
    }
    return false;
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (ws.current) {
      ws.current.close();
    }
  }, []);

  useEffect(() => {
    connect();

    // Cleanup function to close the socket on component unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    lastMessage,
    readyState,
    connectionError,
    sendMessage,
    disconnect,
    reconnect: connect,
    isConnecting: readyState === 'connecting',
    isConnected: readyState === 'open',
    isDisconnected: readyState === 'closed'
  };
}