// frontend/tests/setup.ts

// Import DOM testing utilities to enhance Jest with browser-like assertions
import '@testing-library/jest-dom';

// Polyfill for ResizeObserver - Required for modern UI components
// Many components use ResizeObserver to adapt to container size changes
// JSDOM doesn't provide this API, so we mock it to prevent test failures
if (!global.ResizeObserver) {
  global.ResizeObserver = class ResizeObserver {
    observe() {
      // Mock implementation - no actual observation needed in tests
    }
    unobserve() {
      // Mock implementation - no cleanup needed in tests
    }
    disconnect() {
      // Mock implementation - no disconnection needed in tests
    }
  };
}

// Mock matchMedia for responsive design testing
// Components often use matchMedia to check screen size and adapt behavior
// JSDOM doesn't include this browser API, so we provide a mock implementation
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query: string) => ({
    matches: false, // Default to false for consistent test behavior
    media: query,
    onchange: null,
    addListener: jest.fn(), // Deprecated but still used by some libraries
    removeListener: jest.fn(), // Deprecated but still used by some libraries
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock IntersectionObserver for scroll-based animations and lazy loading
// Used by many modern UI libraries for performance optimizations
if (!global.IntersectionObserver) {
  global.IntersectionObserver = class IntersectionObserver {
    constructor() {}
    observe() {}
    unobserve() {}
    disconnect() {}
  } as any;
}

// Mock requestAnimationFrame for animation testing
// D3.js and other animation libraries use this for smooth animations
// Provide a synchronous mock for predictable test behavior
if (!global.requestAnimationFrame) {
  global.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    return setTimeout(callback, 0);
  };
}

if (!global.cancelAnimationFrame) {
  global.cancelAnimationFrame = (id: number): void => {
    clearTimeout(id);
  };
}

// Mock Canvas API for chart and graph testing
// Some visualization libraries require canvas context
// Provide basic mock to prevent crashes
if (!global.HTMLCanvasElement.prototype.getContext) {
  global.HTMLCanvasElement.prototype.getContext = jest.fn().mockImplementation(() => ({
    fillRect: jest.fn(),
    clearRect: jest.fn(),
    getImageData: jest.fn(() => ({ data: new Array(4) })),
    putImageData: jest.fn(),
    createImageData: jest.fn(() => []),
    setTransform: jest.fn(),
    drawImage: jest.fn(),
    save: jest.fn(),
    fillText: jest.fn(),
    restore: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    closePath: jest.fn(),
    stroke: jest.fn(),
    translate: jest.fn(),
    scale: jest.fn(),
    rotate: jest.fn(),
    arc: jest.fn(),
    fill: jest.fn(),
    measureText: jest.fn(() => ({ width: 0 })),
    transform: jest.fn(),
    rect: jest.fn(),
    clip: jest.fn(),
  }));
}

// Mock URL.createObjectURL for file upload testing
if (!global.URL.createObjectURL) {
  global.URL.createObjectURL = jest.fn().mockImplementation(() => 'mock-url');
}

if (!global.URL.revokeObjectURL) {
  global.URL.revokeObjectURL = jest.fn();
}

// Mock fetch for API testing
// Provide a basic mock that can be overridden in individual tests
if (!global.fetch) {
  global.fetch = jest.fn().mockImplementation(() =>
    Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve({}),
      text: () => Promise.resolve(''),
      blob: () => Promise.resolve(new Blob()),
    } as Response)
  );
}

// Console error suppression for expected warnings in tests
// Some component libraries produce expected warnings that clutter test output
const originalError = console.error;
const originalWarn = console.warn;

// Suppress specific known warnings that don't affect test validity
const suppressedWarnings = [
  'Warning: ReactDOM.render is deprecated',
  'Warning: Using UNSAFE_',
  'Warning: componentWillMount has been renamed',
  'Warning: componentWillReceiveProps has been renamed',
  'Warning: componentWillUpdate has been renamed',
];

console.error = (...args: any[]) => {
  const message = args[0];
  if (typeof message === 'string' && suppressedWarnings.some(warning => message.includes(warning))) {
    return; // Suppress this warning
  }
  originalError.apply(console, args);
};

console.warn = (...args: any[]) => {
  const message = args[0];
  if (typeof message === 'string' && suppressedWarnings.some(warning => message.includes(warning))) {
    return; // Suppress this warning
  }
  originalWarn.apply(console, args);
};

// Setup cleanup for test isolation
afterEach(() => {
  // Clear all mocks after each test to prevent state leakage
  jest.clearAllMocks();

  // Reset fetch mock to default behavior
  if (jest.isMockFunction(global.fetch)) {
    (global.fetch as jest.MockedFunction<typeof fetch>).mockClear();
  }
});

// Global test timeout configuration
// Increase timeout for tests that involve complex rendering or async operations
jest.setTimeout(10000); // 10 seconds

// Mock process.env for consistent test environment
process.env.NODE_ENV = 'test';
process.env.NEXT_PUBLIC_API_BASE_URL = 'http://localhost:3000';

// Export test utilities for common use across test files
export const createMockRouter = (overrides = {}) => ({
  push: jest.fn(),
  replace: jest.fn(),
  prefetch: jest.fn(),
  back: jest.fn(),
  forward: jest.fn(),
  refresh: jest.fn(),
  query: {},
  pathname: '/',
  route: '/',
  asPath: '/',
  ...overrides,
});

export const createMockSearchParams = (params: Record<string, string> = {}) => {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    searchParams.set(key, value);
  });
  return searchParams;
};

// Mock timers utility for testing time-dependent behavior
export const advanceTimersByTime = (ms: number) => {
  jest.advanceTimersByTime(ms);
};

export const runAllTimers = () => {
  jest.runAllTimers();
};

// Mock data generators for consistent testing
export const createMockPayoffData = () => ({
  title: 'Test Payoff Matrix',
  players: ['Player 1', 'Player 2'] as [string, string],
  strategies: [['Strategy A', 'Strategy B'], ['Strategy X', 'Strategy Y']] as [string[], string[]],
  payoffs: [
    [[2, 1], [0, 3]], // Player 1 strategies vs Player 2 strategies
    [[1, 2], [3, 0]]
  ],
  equilibrium: {
    strategies: [0, 1] as [number, number],
    payoffs: [0, 3] as [number, number],
    type: 'Nash Equilibrium' as const
  }
});

export const createMockNetworkData = () => ({
  nodes: [
    { id: '1', type: 'user' as const, status: 'susceptible' as const, influence: 1, connections: 2 },
    { id: '2', type: 'spreader' as const, status: 'infected' as const, influence: 2, connections: 3 },
    { id: '3', type: 'moderator' as const, status: 'immune' as const, influence: 1.5, connections: 2 }
  ],
  links: [
    { source: '1', target: '2', strength: 0.8, type: 'follow' as const },
    { source: '2', target: '3', strength: 0.6, type: 'trust' as const }
  ]
});

export const createMockSimulationParameters = () => ({
  networkSize: 1000,
  networkType: 'scale-free' as const,
  averageDegree: 6,
  clusteringCoefficient: 0.3,
  spreaderRatio: 0.05,
  moderatorRatio: 0.1,
  userRatio: 0.85,
  basePropagationRate: 0.1,
  decayRate: 0.05,
  recoveryRate: 0.08,
  spreaderReward: 2.0,
  moderatorReward: 1.5,
  detectionPenalty: -1.0,
  falsePositivePenalty: -0.5,
  learningRate: 0.1,
  adaptationFrequency: 5,
  timeHorizon: 100,
  enableLearning: true,
  enableNetworkEvolution: false,
  saveFrequency: 10,
  noiseLevel: 0.1,
  memoryLength: 10,
  explorationRate: 0.1,
  convergenceThreshold: 0.001
});