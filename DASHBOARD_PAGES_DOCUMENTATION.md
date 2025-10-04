# Dashboard Pages Documentation

## Game Theory & Disinformation Simulation (GTDS) Platform - Dashboard Pages

This comprehensive document explains each page within the `/frontend/src/app/(dashboard)` directory, detailing its purpose, components, functionality, and how everything works together.

---

## 📁 Dashboard Structure Overview

The dashboard is organized in a Next.js App Router layout with the following structure:

```
frontend/src/app/(dashboard)/
├── layout.tsx                 # Main dashboard layout wrapper
├── simulation/                # Game theory simulation page
│   ├── page.tsx
│   └── components/
│       ├── GameParameters.tsx
│       ├── NetworkGraph.tsx
│       └── PayoffMatrix.tsx
├── classifier/                # News classification page
│   └── page.tsx
├── analytics/                 # Analytics & insights page
│   └── page.tsx
├── equilibrium/              # Nash equilibrium analysis page
│   └── page.tsx
└── network/                  # Network analysis page
    └── page.tsx
```

---

## 🎨 Dashboard Layout (`layout.tsx`)

### Purpose
The main layout wrapper that provides consistent structure, navigation, and theming across all dashboard pages.

### Key Features
- **Responsive Sidebar Navigation**: Collapsible sidebar with navigation links
- **Breadcrumb Navigation**: Shows current location in the app hierarchy
- **Page Transitions**: Smooth animations using Framer Motion
- **Theme Support**: Dark/light mode toggle
- **Scroll to Top Button**: Appears after scrolling 300px
- **Glass-morphism Design**: Modern UI with backdrop blur effects

### Components Used
- `Header` - Top navigation bar
- `Sidebar` - Left navigation menu
- `Breadcrumbs` - Path navigation
- `ScrollToTopButton` - Floating action button

### Layout Flow
1. Mobile header (visible on small screens)
2. Collapsible sidebar (desktop: always visible, mobile: toggleable)
3. Main content area with:
   - Breadcrumbs
   - Animated page content (AnimatePresence)
   - Footer with links

### Animation Details
```typescript
pageVariants = {
  initial: { opacity: 0, y: 8, scale: 0.99 },
  animate: { opacity: 1, y: 0, scale: 1, duration: 0.3 },
  exit: { opacity: 0, y: -8, scale: 0.99, duration: 0.2 }
}
```

---

## 🎮 1. Simulation Page (`/simulation`)

### Purpose
The **core feature** of the platform - runs game theory simulations of fake news propagation on social networks with strategic agent interactions.

### Page Structure
**Location**: `/simulation/page.tsx`

### Key Components

#### 1.1 **GameParameters Component**
**File**: `simulation/components/GameParameters.tsx`

**Purpose**: Comprehensive parameter configuration form for simulation setup

**Features**:
- **5 Tabbed Sections**:
  1. **Network Tab**: Configure network topology
     - Network size (10-150 nodes)
     - Network type (Scale-free, Small-world, Random, Grid)
     - Average degree
     - Clustering coefficient

  2. **Agents Tab**: Set agent population distribution
     - Spreader ratio (fake news spreaders)
     - Moderator ratio (fact-checkers)
     - User ratio (passive consumers)
     - Bot ratio (automated agents)
     - Live population summary

  3. **Propagation Tab**: Information spread parameters
     - Base propagation rate
     - Decay rate
     - Recovery rate
     - Immunity rate

  4. **Game Theory Tab**: Payoff structure
     - Spreader rewards
     - Moderator rewards
     - Detection penalty
     - False positive penalty
     - Learning rate & adaptation frequency

  5. **Advanced Tab**: Fine-tuning options
     - Time horizon (simulation duration)
     - Random seed (reproducibility)
     - Save frequency
     - Enable learning toggle
     - Network evolution toggle
     - Noise level, memory length, exploration rate
     - Convergence threshold

**Validation**:
- Agent ratios must sum to 1.0
- Parameter constraints enforced with sliders
- Real-time validation feedback

**UI Components**:
- Sliders with value display
- Select dropdowns
- Number inputs
- Switch toggles
- Tooltips with help text

#### 1.2 **NetworkGraph Component**
**File**: `simulation/components/NetworkGraph.tsx`

**Purpose**: Smart wrapper for network visualization with simulation animation

**Features**:
- **Loading State**: Spinner while generating network
- **Empty State**: Helpful message when no data
- **Live Simulation Animation**:
  - Animates through simulation steps (500ms per step)
  - Updates node colors based on infection status
  - Updates node sizes based on influence score
  - Shows progress indicator

**Node States** (during simulation):
- `susceptible` - Slate gray (not yet exposed)
- `infected` - Red (currently spreading misinformation)
- `recovered` - Green (no longer susceptible)
- `immune` - Blue (resistant to misinformation)

**User Types** (static):
- `spreader` - Amber
- `fact_checker` - Green
- `platform` - Indigo
- `regular_user` - Slate

**Overlays**:
- **Network Statistics Panel** (top-right):
  - Total nodes
  - Total edges
  - Network density

- **Simulation Progress** (top-left, when running):
  - Current step / total steps
  - Progress bar animation

- **Node Tooltip** (on hover):
  - Node ID
  - User type
  - Influence score
  - Credibility score
  - Connection count

- **Legend** (bottom-left):
  - Color-coded status/type indicators

#### 1.3 **PayoffMatrix Component**
**File**: `simulation/components/PayoffMatrix.tsx`

**Purpose**: Interactive payoff matrix visualization with game theory analysis

**Features**:

**3 Tabbed Views**:

1. **Matrix Tab**:
   - Visual payoff table
   - Dual-value cells (player 1 / player 2 payoffs)
   - Nash equilibrium highlighting (yellow border)
   - Pareto optimal highlighting (green border)
   - Cell selection for detailed info
   - View modes: Combined, Player 1 only, Player 2 only
   - Optional heatmap visualization

2. **Analysis Tab** (4 cards):
   - **Nash Equilibria**: Lists all pure strategy Nash equilibria
   - **Dominant Strategies**: Shows strict/weak dominant strategies
   - **Pareto Optimal Outcomes**: Lists Pareto frontier
   - **Social Welfare**: Maximum vs current welfare with efficiency %

3. **Insights Tab**:
   - Strategic recommendations
   - Equilibrium interpretation
   - Efficiency considerations
   - Coordination advice

**Analysis Algorithms**:
- Finds pure strategy Nash equilibria (checks best responses)
- Identifies dominant strategies (strict & weak)
- Computes Pareto frontier
- Calculates social welfare

**Exports**:
- JSON download with full analysis

### Simulation Flow

1. **User configures parameters** via GameParameters form
2. **Form validation** ensures valid configuration
3. **Submit triggers simulation** → Backend API call
4. **Backend runs simulation**:
   - Generates network structure
   - Places agents according to ratios
   - Simulates propagation dynamics
   - Calculates Nash equilibrium
   - Tracks payoffs over time
5. **Results streamed back**:
   - Network data
   - Payoff trends
   - Final metrics
   - Convergence analysis
6. **Frontend displays**:
   - Live network animation
   - Payoff matrix with equilibrium
   - Summary metrics cards
   - Propagation timeline chart

### Metrics Displayed

**Summary Cards** (animated count-up):
- Total Reach (nodes affected)
- Platform Reputation (% score)
- Detection Rate (% detected)
- Equilibrium Stability (% stable)

**Tabs**:
1. **Summary**: Network characteristics, agent distribution
2. **Propagation Timeline**: Line chart of infected/susceptible/recovered
3. **Game Outcome**: Payoff matrix + equilibrium analysis

---

## 🤖 2. Classifier Page (`/classifier`)

### Purpose
Real-time fake news classification using trained ML models with explainable AI features.

### Features

#### Model Selection
**7 Available Models**:
1. **Ensemble** (Recommended) - 99.86% accuracy - Weighted voting
2. **Gradient Boosting** - 99.95% accuracy - Best single model
3. **Random Forest** - 99.89% accuracy - Tree ensemble
4. **Naive Bayes** - 94.83% accuracy - Probabilistic
5. **Logistic Regression** - 66.71% accuracy - Baseline
6. **DistilBERT** - Coming soon - Transformer
7. **LSTM** - Coming soon - Neural network

#### Input Interface
- **Large text area** for article/post input
- Character & word count display
- Model dropdown selector with performance info
- "Analyze Text" button

#### Results Display (3 Tabs)

**1. Prediction Tab**:
- **Verdict Card**: Large, color-coded result
  - "Likely Real News" (green) / "Likely Fake News" (red)
  - Confidence percentage with progress bar
  - Interpretation text based on confidence level
- **Metadata**:
  - Model used
  - Processing time (ms)
  - Text length

**2. Probabilities Tab**:
- **Bar Chart**: Probability distribution
- **Cards**: Real vs Fake percentages
- Visual comparison

**3. Explanation Tab**:
- **Top Influential Phrases**:
  - Phrases that support "Real" classification
  - Phrases that indicate "Fake" classification
  - Contribution percentages
- **Feature Importance**:
  - Top 5 features ranked by impact
  - Progress bars showing importance
  - Type indicators (positive/negative)
- **Model Transparency Alert**: Explains how to interpret results

#### Information Cards (Bottom)
1. **How It Works**: Explains detection methodology
2. **Model Performance**: Dataset info, accuracy stats
3. **Important Note**: Disclaimer about verification

### Workflow

1. User enters text
2. Selects model
3. Clicks "Analyze Text"
4. Backend processes:
   - Text preprocessing
   - Feature extraction
   - Model inference
   - Explanation generation (SHAP/LIME)
5. Results displayed with animations:
   - Fade-in verdict
   - Animated progress bars
   - Staggered phrase reveals

---

## 📊 3. Analytics Page (`/analytics`)

### Purpose
Comprehensive visualization of model performance, propagation dynamics, and intervention strategies.

### Key Sections

#### Filter Bar
- Dataset selector (FakeNewsNet, LIAR, Combined)
- Last updated timestamp

#### KPI Metrics (4 Animated Cards)
1. **Best Model Accuracy**: 88.4% (Ensemble)
2. **Top Feature Importance**: 28.4% (Source Credibility)
3. **Propagation Speed Ratio**: 1.7x (Fake vs Real)
4. **Best Intervention**: 79% reduction (Labeling + Penalties)

#### Visualizations

**1. Model Performance Comparison** (Bar Chart)
- X-axis: Model names (BERT, LSTM, Ensemble, Traditional ML)
- Y-axis: Performance metrics
- 3 Series:
  - Accuracy % (blue)
  - F1-Score % (purple)
  - AUC-ROC % (green)
- Key Finding callout: Ensemble superiority

**2. Feature Importance Analysis** (Horizontal Bars)
- Top 15 features ranked by importance
- Color-coded by category:
  - **Linguistic** (blue): Sentiment, Named Entities, Emotional Language
  - **Network** (purple): Source Credibility, Centrality, Propagation Velocity
  - **Stylistic** (amber): Readability, Clickbait, Sensationalism
- Percentage importance values
- Animated reveal on scroll

**3. Information Propagation Dynamics** (Line Chart)
- X-axis: Time (hours, 0-24)
- Y-axis: Reach (number of people)
- 2 Lines:
  - Fake News Reach (red)
  - Real News Reach (green)
- Summary cards:
  - Peak fake news: 91.2K
  - Peak real news: 49.3K
  - Final spread ratio: 1.85x

**4. Intervention Strategy Effectiveness** (Table)
- **Columns**:
  - Strategy name
  - Reduction % (with progress bar)
  - Cost-Effectiveness badge
  - Complexity badge
  - Implementation time
- **6 Strategies**:
  1. Content Labeling: 42% reduction, Low complexity
  2. Fact-Checking Alerts: 58% reduction, Medium complexity
  3. Source Verification: 67% reduction, High complexity
  4. Network Penalties: 54% reduction, Medium complexity
  5. **Labeling + Penalties: 79% reduction**, Medium complexity ⭐
  6. Full Multi-Layer: 86% reduction, Very High complexity
- Recommended strategy callout

### Data Sources
Based on project research findings:
- Model performance from training notebooks
- Feature importance from SHAP/LIME analysis
- Propagation data from simulation runs
- Intervention effectiveness from literature review

---

## 🎯 4. Equilibrium Page (`/equilibrium`)

### Purpose
Interactive Nash equilibrium calculator and analyzer for various player interactions.

### Interactive Controls (Left Panel - Sticky)

#### Scenario Selector
**4 Predefined Scenarios**:
1. **Baseline**: Standard social media dynamics
   - Detection penalty: 50%
   - Verification cost: 30%
   - Engagement revenue: 60%

2. **High Detection**: Strong moderation
   - Detection penalty: 90%
   - Lower engagement revenue

3. **Low Moderation**: Minimal intervention
   - Low detection penalty: 20%
   - High engagement revenue: 80%

4. **High Engagement**: Profit-driven
   - Low detection: 30%
   - Very high revenue: 95%

#### Player Interaction Selector
**4 Game Types**:
1. **Spreader vs. Moderator**
   - Strategies: Aggressive/Conservative vs Strict/Lenient

2. **Spreader vs. Platform**
   - Strategies: Post Fake/Post Mixed vs Strict/Lenient Policy

3. **Platform vs. Fact-Checker**
   - Strategies: Active/Passive Monitoring vs Comprehensive/Selective

4. **Moderator vs. User**
   - Strategies: Proactive/Reactive vs Report/Ignore

#### Parameter Sliders (5 adjustable)
1. **Detection Penalty** (0-100%)
2. **Verification Cost** (0-100%)
3. **Engagement Revenue** (0-100%)
4. **Network Density** (0-100%)
5. **Learning Rate** (0-100%)

Each slider has:
- Current value badge
- Description tooltip
- Real-time update

#### Calculate Button
- Enabled only when parameters change
- Shows "Calculating..." animation
- Generates new equilibrium

### Results Display (Right Panel)

#### Payoff Matrix Card
- Full PayoffMatrix component (see section 1.3)
- Highlights Nash equilibrium cell
- Shows equilibrium strategies

#### Equilibrium Analysis Card

**1. Pure Strategy Equilibrium Box**:
- Strategy for each player
- Type badge (Pure/Mixed)
- Stability percentage badge

**2. Expected Payoffs**:
- 2 animated cards showing payoffs at equilibrium
- Player 1 payoff (blue card)
- Player 2 payoff (red card)

**3. Strategic Insight Alert**:
Context-aware recommendations based on parameters:
- High detection (>70%): Explains spreader incentives to be truthful
- High revenue (>70%): Discusses platform conflicts of interest
- High verification cost (>60%): Suggests cost reduction strategies
- Balanced: General strategic considerations

**4. Additional Metrics**:
- **Equilibrium Type**: Strict Nash / Weak Nash / Trembling Hand Perfect
- **Social Welfare**: Average payoff

### How It Works

1. User selects scenario OR adjusts sliders manually
2. Frontend generates payoff matrix using transformation function:
   ```typescript
   payoff = basePayoff * (parameter / 50)
   ```
3. Equilibrium calculation:
   - Checks all strategy combinations
   - Identifies best responses
   - Finds cells where neither player wants to deviate
4. Displays results with animated transitions
5. Provides strategic insights based on parameter values

---

## 🕸️ 5. Network Analysis Page (`/network`)

### Purpose
Generate, visualize, and analyze social network topologies with centrality metrics and community detection.

### Network Controls (Top Card)

#### Network Selector
- **Sample Network 1** (100 nodes)
- **Sample Network 2** (150 nodes)
- **Sample Network 3** (200 nodes)
- **Generate New Network** (custom)

#### Custom Network Generation
When "Generate New" is selected, 3 additional controls appear:

1. **Network Type Dropdown**:
   - **Barabási-Albert** (scale-free, power law degree distribution)
   - **Watts-Strogatz** (small-world, high clustering)
   - **Erdős-Rényi** (random graph)

2. **Node Count Slider** (50-200)

3. **Generate Button** (with loading animation)

### Visualization Section (Left, 58% width)

#### Network Visualization Area
- Placeholder for D3.js force-directed graph
- 600px height, responsive
- Gradient background (slate → blue)

#### Visual Encoding Controls
1. **Color Nodes By**:
   - User Type (spreader/moderator/bot/user)
   - Community ID
   - Influence Score

2. **Size Nodes By**:
   - Degree (connection count)
   - Betweenness Centrality
   - Influence Score

Current selections shown as badges below visualization

### Statistical Analysis (Right Panel, 42% width)

#### 3 Tabbed Analysis Views

**1. Overview Tab**:
5 Animated metric cards:
- **Nodes**: Total count
- **Edges**: Total connections
- **Density**: Connectivity ratio (0-1)
- **Avg Clustering**: Clustering coefficient
- **Diameter**: Longest shortest path

**2. Centrality Tab**:

**Metric Selector**:
- Degree Centrality (connection count)
- Betweenness Centrality (bridge importance)
- Eigenvector Centrality (influence from connections)
- PageRank (Google algorithm)

**Visualization**:
- Bar chart showing top 20 nodes by selected metric
- Description of metric below selector

**Top 10 Table**:
- Ranked list with node IDs
- Rank badges (1, 2, 3...)
- Score values (3 decimals)

**3. Communities Tab**:

**Metrics**:
- Community count
- Modularity score (quality of community structure)

**Largest Communities Table**:
- Community ID
- Size (member count)
- Dominant Type badge (spreader/moderator/user/bot)

**Info Box**:
- Louvain algorithm explanation
- Modularity interpretation

### Network Generation Algorithm

1. **Create nodes** based on count
2. **Assign user types**:
   - First 5: spreaders
   - Next 10: moderators/fact-checkers
   - Next 5: bots
   - Remaining: regular users
3. **Generate edges** based on network type
4. **Calculate centrality metrics**:
   - Degree: edge count
   - Betweenness: shortest path participation
   - Eigenvector: recursive influence
   - PageRank: random walk probability
5. **Detect communities** (Louvain algorithm)
6. **Compute statistics**:
   - Density: edges / max possible edges
   - Clustering: triangle closure probability
   - Diameter: longest shortest path

---

## 🔧 Shared Components & Utilities

### UI Components (from `/components/ui/`)
- **Card, CardHeader, CardTitle, CardDescription, CardContent** - Container components
- **Button** - Action buttons with variants
- **Input, Textarea** - Form inputs
- **Slider** - Range input with visual feedback
- **Select** - Dropdown selector
- **Tabs** - Tabbed navigation
- **Badge** - Label/tag component
- **Alert** - Notification boxes
- **Progress** - Progress bars
- **Switch** - Toggle switches
- **Table** - Data tables
- **Separator** - Dividers
- **Tooltip** - Contextual help

### Chart Components (from `/components/charts/`)
- **LineChart** - Time series visualization
- **BarChart** - Comparative bar charts
- **NetworkVisualization** - D3.js network graphs

### Game Theory Components
- **PayoffMatrix** - Game matrix with analysis (shared)

### Hooks
- **useSimulationStore** - Zustand store for simulation state
- **useClassifier** - API hook for classification
- **useCountUp** - Animated number counter
- **useUIStore** - UI state (sidebar, theme)

### Animation Utilities
- **Framer Motion** variants for page transitions
- **AnimatePresence** for enter/exit animations
- Staggered animations for lists
- Progress bar transitions

---

## 🎨 Design System

### Color Palette
- **Primary**: Blue (#3b82f6)
- **Success**: Green (#10b981)
- **Warning**: Amber (#f59e0b)
- **Danger**: Red (#ef4444)
- **Purple**: (#8b5cf6)
- **Slate**: Gray (#64748b)

### Typography
- **Headings**: Font-bold, gradient text for titles
- **Body**: Regular weight, gray-900
- **Monospace**: For numerical data, IDs

### Spacing
- Card padding: 6-8 (1.5rem - 2rem)
- Gap between sections: 6-8
- Component spacing: 4

### Responsive Breakpoints
- **sm**: 640px
- **md**: 768px
- **lg**: 1024px
- **xl**: 1280px

### Animations
- **Duration**: 0.3-0.5s for page transitions
- **Easing**: cubic-bezier(0.4, 0, 0.2, 1)
- **Stagger delay**: 0.05-0.1s between items

---

## 🔄 Data Flow

### Simulation Page Flow
```
User Input → GameParameters
  ↓
Parameter Validation
  ↓
API Call (POST /simulations/run)
  ↓
Backend Simulation Engine
  ├── Network Generation
  ├── Agent Placement
  ├── Game Theory Calculations
  └── Propagation Simulation
  ↓
Results Object
  ├── Network data → NetworkGraph
  ├── Payoff trends → LineChart
  ├── Final metrics → MetricsCards
  └── Equilibrium → PayoffMatrix
```

### Classifier Page Flow
```
Text Input + Model Selection
  ↓
API Call (POST /classify)
  ↓
Backend ML Pipeline
  ├── Text Preprocessing
  ├── Feature Extraction
  ├── Model Inference
  └── Explanation Generation
  ↓
Classification Result
  ├── Prediction → Verdict Card
  ├── Probabilities → BarChart
  └── Explanation → Feature List
```

### Analytics Page Flow
```
Page Load
  ↓
Fetch/Load Mock Data
  ├── Model Performance
  ├── Feature Importance
  ├── Propagation Data
  └── Intervention Strategies
  ↓
Render Visualizations
  ├── Animated KPIs
  ├── Bar Charts
  ├── Line Charts
  └── Tables
```

### Equilibrium Page Flow
```
Scenario Selection / Parameter Change
  ↓
Generate Payoff Matrix (Client-side)
  ↓
Analyze Matrix
  ├── Find Nash Equilibria
  ├── Check Dominant Strategies
  ├── Compute Pareto Frontier
  └── Calculate Social Welfare
  ↓
Display Results
  ├── Payoff Matrix
  ├── Equilibrium Info
  └── Strategic Insights
```

### Network Page Flow
```
Network Selection / Generation
  ↓
Generate Network Data (Mock)
  ├── Create Nodes
  ├── Create Edges
  ├── Calculate Metrics
  └── Detect Communities
  ↓
Update Visualization & Analysis
  ├── Network Graph
  ├── Statistics Cards
  ├── Centrality Charts
  └── Community Tables
```

---

## 📱 Responsive Behavior

### Mobile (< 768px)
- Sidebar collapses to hamburger menu
- Dashboard shows mobile header
- Card grid → single column
- Tables → horizontal scroll
- Reduced padding & font sizes

### Tablet (768px - 1024px)
- 2-column grid for cards
- Sidebar visible but narrower
- Network visualization scales down

### Desktop (> 1024px)
- Full multi-column layouts
- Sidebar always visible
- Optimal chart sizes
- Maximum content width: 7xl (1280px)

---

## 🚀 Performance Optimizations

1. **Code Splitting**: Each page lazy-loaded
2. **Memoization**: useMemo for expensive calculations
3. **Virtualization**: For large network graphs
4. **Debouncing**: Parameter sliders
5. **Skeleton Loading**: While fetching data
6. **Image Optimization**: Next.js Image component
7. **Animation Performance**: GPU-accelerated transforms

---

## 🔗 API Endpoints Used

### Simulation
- `POST /api/simulations/run` - Run simulation
- `GET /api/simulations/{id}` - Get simulation results

### Classification
- `POST /api/classify` - Classify text
- `GET /api/models` - List available models

### Network
- `POST /api/networks/generate` - Generate network
- `GET /api/networks/{id}` - Get network data

### Analytics
- `GET /api/analytics/model-performance`
- `GET /api/analytics/feature-importance`
- `GET /api/analytics/propagation-stats`

---

## 🎯 Key User Journeys

### Journey 1: Run a Simulation
1. Navigate to `/simulation`
2. View default parameters
3. Adjust network size to 150
4. Change spreader ratio to 10%
5. Click "Run Simulation"
6. Watch network animation
7. View equilibrium results
8. Analyze propagation timeline

### Journey 2: Classify News
1. Navigate to `/classifier`
2. Paste article text
3. Select "Ensemble" model
4. Click "Analyze Text"
5. View verdict with confidence
6. Check probabilities chart
7. Read explanation phrases

### Journey 3: Analyze Strategies
1. Navigate to `/equilibrium`
2. Select "High Detection" scenario
3. Adjust detection penalty to 85%
4. Click "Calculate Equilibrium"
5. View payoff matrix
6. Read Nash equilibrium
7. Check strategic insights

### Journey 4: Explore Network
1. Navigate to `/network`
2. Click "Generate New Network"
3. Select "Barabási-Albert"
4. Set 200 nodes
5. Click "Generate"
6. Switch to "Centrality" tab
7. Select "Betweenness" metric
8. View top 10 central nodes

### Journey 5: Review Analytics
1. Navigate to `/analytics`
2. View KPI metrics
3. Scroll to model comparison chart
4. Check feature importance
5. Analyze propagation dynamics
6. Review intervention strategies
7. Identify best approach

---

## 🛠️ Customization Guide

### Adding a New Dashboard Page

1. **Create page directory**:
   ```
   frontend/src/app/(dashboard)/newpage/
   └── page.tsx
   ```

2. **Create page component**:
   ```typescript
   "use client";

   export default function NewPage() {
     return (
       <div className="space-y-8">
         <h1>New Page Title</h1>
         {/* Content */}
       </div>
     );
   }
   ```

3. **Add to sidebar navigation** in `/components/layout/Sidebar.tsx`

4. **Add breadcrumb** in `/components/layout/Breadcrumbs.tsx`

### Modifying a Component

1. **Locate component** in file structure
2. **Read component props** interface
3. **Update logic/UI** as needed
4. **Test with existing pages**

### Adding a New Chart

1. **Create chart component** in `/components/charts/`
2. **Use Recharts library**
3. **Follow existing chart patterns**
4. **Export from index**

---

## 📊 Component Dependency Graph

```
layout.tsx
  ├── Header
  ├── Sidebar
  └── Breadcrumbs

simulation/page.tsx
  ├── GameParameters
  │   ├── Slider
  │   ├── Select
  │   ├── Input
  │   └── Switch
  ├── NetworkGraph
  │   └── NetworkVisualization (D3.js)
  └── PayoffMatrix
      ├── Table
      ├── Tabs
      └── Badge

classifier/page.tsx
  ├── Textarea
  ├── Select
  ├── Button
  ├── Tabs
  ├── Progress
  ├── BarChart
  └── Badge

analytics/page.tsx
  ├── MetricsCard
  ├── BarChart
  ├── LineChart
  └── Table

equilibrium/page.tsx
  ├── Select
  ├── Slider
  ├── Button
  ├── PayoffMatrix
  └── Alert

network/page.tsx
  ├── Select
  ├── Slider
  ├── Button
  ├── Tabs
  ├── MetricsCard
  ├── BarChart
  └── Table
```

---

## 🔍 Testing Scenarios

### Simulation Page
- ✅ Form validation (agent ratios sum to 1)
- ✅ Network generation with different topologies
- ✅ Simulation start/stop/reset
- ✅ Results display and animation
- ✅ Responsive layout

### Classifier Page
- ✅ Text input validation
- ✅ Model selection
- ✅ Classification results
- ✅ Error handling
- ✅ Explanation display

### Analytics Page
- ✅ Data loading
- ✅ Chart rendering
- ✅ Animated counters
- ✅ Table sorting

### Equilibrium Page
- ✅ Scenario switching
- ✅ Parameter validation
- ✅ Equilibrium calculation
- ✅ Matrix highlighting
- ✅ Insight generation

### Network Page
- ✅ Network generation
- ✅ Centrality calculation
- ✅ Community detection
- ✅ Visual encoding updates

---

## 📚 Further Reading

### Related Documentation
- [API Endpoints Documentation](docs/api/endpoints.md)
- [Backend Architecture](docs/ARCHITECTURE.md)
- [Frontend State Management](docs/STATE_MANAGEMENT.md)
- [Component Library](docs/COMPONENTS.md)

### External Resources
- [Next.js App Router](https://nextjs.org/docs/app)
- [Framer Motion](https://www.framer.com/motion/)
- [Recharts](https://recharts.org/)
- [D3.js](https://d3js.org/)
- [Tailwind CSS](https://tailwindcss.com/)

---

## 🎓 Learning Path

**For New Developers**:

1. **Start with Layout** (`layout.tsx`)
   - Understand dashboard structure
   - Learn page transitions

2. **Study Classifier Page** (simplest)
   - Form submission
   - API integration
   - Results display

3. **Explore Analytics Page** (data visualization)
   - Chart components
   - Animated metrics
   - Data transformation

4. **Dive into Simulation Page** (most complex)
   - Multi-component coordination
   - Real-time updates
   - Complex state management

5. **Master Equilibrium Page** (game theory)
   - Algorithm implementation
   - Mathematical visualization
   - Analysis generation

6. **Advanced: Network Page** (graph algorithms)
   - Network science concepts
   - Graph visualization
   - Statistical analysis

---

## 🐛 Common Issues & Solutions

### Issue 1: Simulation not starting
**Cause**: Parameter validation failure
**Solution**: Check that agent ratios sum to 1.0

### Issue 2: Charts not rendering
**Cause**: Missing/invalid data
**Solution**: Verify data structure matches chart expectations

### Issue 3: Network visualization blank
**Cause**: D3.js not initialized
**Solution**: Check console for errors, ensure data has nodes/links

### Issue 4: Slow performance with large networks
**Cause**: Too many nodes (>200)
**Solution**: Implement virtualization or limit node count

### Issue 5: Animation stuttering
**Cause**: Heavy calculations during render
**Solution**: Move calculations to useMemo or web worker

---

## 🎉 Conclusion

This dashboard provides a comprehensive platform for:
- ✅ Simulating fake news propagation using game theory
- ✅ Classifying news articles with ML models
- ✅ Analyzing intervention strategies
- ✅ Computing Nash equilibria
- ✅ Visualizing social networks

Each page is carefully designed with:
- 🎨 Modern UI/UX with animations
- 📱 Responsive design
- ♿ Accessibility features
- 🚀 Performance optimizations
- 🔧 Extensible architecture

**Ready to contribute?** Check out the [Contributing Guide](docs/CONTRIBUTING.md)!

---

**Last Updated**: 2025-10-04
**Version**: 1.0.0
**Maintainer**: GTDS Platform Team
