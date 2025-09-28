// frontend/tests/components/PayoffMatrix.test.tsx

import React from 'react';
import { render, screen } from '@testing-library/react';
import { PayoffMatrix } from '@/components/game-theory/PayoffMatrix';
import { PayoffMatrixData } from '@/types/gameTheory';

// Mock cn utility function
jest.mock('@/lib/utils', () => ({
  cn: (...classes: any[]) => classes.filter(Boolean).join(' ')
}));

describe('PayoffMatrix Component', () => {
  // Create proper mock data matching the actual component structure
  const mockPayoffData: PayoffMatrixData = {
    players: ['Spreader', 'Fact Checker'],
    strategies: {
      'Spreader': ['Aggressive', 'Moderate'],
      'Fact Checker': ['Active', 'Passive']
    },
    payoffs: [
      [{ 'Spreader': 2, 'Fact Checker': 1 }, { 'Spreader': 0, 'Fact Checker': 3 }],
      [{ 'Spreader': 1, 'Fact Checker': 2 }, { 'Spreader': 3, 'Fact Checker': 0 }]
    ],
    equilibrium: {
      strategies: [0, 1],
      payoffs: { 'Spreader': 0, 'Fact Checker': 3 },
      type: 'pure',
      stability: 0.8,
      classification: 'strict'
    }
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('renders the component title correctly', () => {
      const title = 'Test Payoff Matrix';
      render(<PayoffMatrix data={mockPayoffData} title={title} />);

      expect(screen.getByText(title)).toBeInTheDocument();
    });

    it('displays player names in the matrix header', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" />);

      expect(screen.getByText(mockPayoffData.players[0])).toBeInTheDocument();
      expect(screen.getByText(mockPayoffData.players[1])).toBeInTheDocument();
    });

    it('renders all strategy labels', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" />);

      // Check first player strategies
      mockPayoffData.strategies[mockPayoffData.players[0]].forEach(strategy => {
        expect(screen.getByText(strategy)).toBeInTheDocument();
      });

      // Check second player strategies
      mockPayoffData.strategies[mockPayoffData.players[1]].forEach(strategy => {
        expect(screen.getByText(strategy)).toBeInTheDocument();
      });
    });

    it('displays the payoff matrix table structure', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" />);

      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();

      // Check that we have the correct number of data rows (excluding headers)
      const rows = screen.getAllByRole('row');
      expect(rows).toHaveLength(mockPayoffData.strategies[mockPayoffData.players[0]].length + 2); // +2 for headers
    });
  });

  describe('Payoff Values Display', () => {
    it('displays all payoff values correctly', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" />);

      // Check each payoff value is displayed (using getAllByText for duplicate values)
      mockPayoffData.payoffs.forEach((row) => {
        row.forEach((cell) => {
          const spreaderPayoff = cell[mockPayoffData.players[0]];
          const factCheckerPayoff = cell[mockPayoffData.players[1]];

          const spreaderElements = screen.getAllByText(spreaderPayoff.toFixed(2));
          const factCheckerElements = screen.getAllByText(factCheckerPayoff.toFixed(2));

          expect(spreaderElements.length).toBeGreaterThan(0);
          expect(factCheckerElements.length).toBeGreaterThan(0);
        });
      });
    });

    it('formats payoff values with correct decimal places', () => {
      const dataWithDecimals: PayoffMatrixData = {
        ...mockPayoffData,
        payoffs: [
          [{ 'Spreader': 2.456, 'Fact Checker': 1.234 }, { 'Spreader': 0.789, 'Fact Checker': 3.567 }],
          [{ 'Spreader': 1.111, 'Fact Checker': 2.222 }, { 'Spreader': 3.333, 'Fact Checker': 0.444 }]
        ]
      };

      render(<PayoffMatrix data={dataWithDecimals} title="Test Matrix" />);

      // Check that values are displayed with 2 decimal places
      expect(screen.getByText('2.46')).toBeInTheDocument(); // 2.456 formatted
      expect(screen.getByText('1.23')).toBeInTheDocument(); // 1.234 formatted
      expect(screen.getByText('0.79')).toBeInTheDocument(); // 0.789 formatted
    });
  });

  describe('Equilibrium Highlighting', () => {
    it('highlights Nash equilibrium when highlightEquilibrium is true', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" highlightEquilibrium={true} />);

      // Find the equilibrium cell (should have yellow background)
      const cells = screen.getAllByRole('cell');
      const equilibriumCell = cells.find(cell =>
        cell.className?.includes('bg-yellow-200')
      );

      expect(equilibriumCell).toBeInTheDocument();
    });

    it('does not highlight equilibrium when highlightEquilibrium is false', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" highlightEquilibrium={false} />);

      // Check that no cells have equilibrium highlighting
      const cells = screen.getAllByRole('cell');
      const highlightedCells = cells.filter(cell =>
        cell.className?.includes('bg-yellow-200')
      );

      expect(highlightedCells).toHaveLength(0);
    });

    it('displays equilibrium summary text when highlighted', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" highlightEquilibrium={true} />);

      // Check for Nash equilibrium legend (there are multiple instances)
      const nashEquilibriumElements = screen.getAllByText('Nash Equilibrium');
      expect(nashEquilibriumElements.length).toBeGreaterThan(0);

      // Check that equilibrium information is displayed
      expect(screen.getByText(/Spreader.*Aggressive/)).toBeInTheDocument();
      expect(screen.getByText(/Fact Checker.*Passive/)).toBeInTheDocument();
    });
  });

  describe('Selected Cell Highlighting', () => {
    it('highlights selected cell when selectedCell prop is provided', () => {
      render(
        <PayoffMatrix
          data={mockPayoffData}
          title="Test Matrix"
          selectedCell={{ row: 0, col: 1 }}
        />
      );

      // Find the selected cell (should have blue background)
      const cells = screen.getAllByRole('cell');
      const selectedCell = cells.find(cell =>
        cell.className?.includes('bg-blue-100')
      );

      expect(selectedCell).toBeInTheDocument();
    });

    it('shows selected cell legend when a cell is selected', () => {
      render(
        <PayoffMatrix
          data={mockPayoffData}
          title="Test Matrix"
          selectedCell={{ row: 0, col: 1 }}
        />
      );

      expect(screen.getByText('Selected Strategy')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles missing equilibrium data gracefully', () => {
      const dataWithoutEquilibrium: PayoffMatrixData = {
        ...mockPayoffData,
        equilibrium: undefined
      };

      render(<PayoffMatrix data={dataWithoutEquilibrium} title="Test Matrix" highlightEquilibrium={true} />);

      // Component should still render without errors
      expect(screen.getByText('Test Matrix')).toBeInTheDocument();
    });

    it('handles empty strategies gracefully', () => {
      const emptyData: PayoffMatrixData = {
        ...mockPayoffData,
        strategies: {
          'Spreader': [],
          'Fact Checker': []
        },
        payoffs: []
      };

      render(<PayoffMatrix data={emptyData} title="Empty Matrix" />);

      // Component should render with basic structure
      expect(screen.getByText('Empty Matrix')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper table structure for screen readers', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" />);

      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();

      // Check for proper table headers
      const columnHeaders = screen.getAllByRole('columnheader');
      expect(columnHeaders.length).toBeGreaterThan(0);

      // Check for row headers (th elements in tbody act as row headers)
      const headers = table.querySelectorAll('th');
      expect(headers.length).toBeGreaterThan(0);
    });

    it('includes meaningful text for matrix cells', () => {
      render(<PayoffMatrix data={mockPayoffData} title="Test Matrix" />);

      // Check that payoff values are presented clearly
      const cells = screen.getAllByRole('cell');
      const cellsWithPayoffs = cells.filter(cell =>
        cell.textContent?.includes('.') && /\d+\.\d+/.test(cell.textContent)
      );

      expect(cellsWithPayoffs.length).toBeGreaterThan(0);
    });
  });
});