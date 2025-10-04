# Dataset Analysis for Game Theory Fake News Detection Project

## Overview

This document provides a comprehensive analysis of the three datasets available for the Game Theory Fake News Detection research project. Each dataset offers unique characteristics and can be leveraged for different aspects of fake news detection, game theory modeling, and network analysis.

## Project Context

Based on the project documentation, this research platform combines:
- **Machine Learning**: BERT, LSTM, and ensemble models for fake news detection
- **Game Theory**: Multi-player games with spreaders, fact-checkers, and platforms
- **Network Analysis**: Social network propagation models and influence metrics
- **Strategic Analysis**: Nash equilibrium computation for optimal counter-strategies

The datasets below support various components of this comprehensive framework.

---

## 1. FakeNewsNet Dataset

### Location
`/home/ghost/fake-news-game-theory/data/raw/fakenewsnet/`

### Description
FakeNewsNet is a multi-modal fake news detection dataset that provides both news content and social media engagement data. This dataset is particularly valuable for network analysis and understanding how fake news spreads through social media platforms.

### Dataset Structure

#### Files
- `gossipcop_fake.csv` (5,323 records) - 12.5 MB
- `gossipcop_real.csv` (16,817 records) - 19.9 MB
- `politifact_fake.csv` (432 records) - 3.3 MB
- `politifact_real.csv` (624 records) - 8.3 MB

**Total Records**: 23,196 news articles

#### Data Schema
```
- id: Unique identifier for each news article
- news_url: Source URL of the news article
- title: Article headline/title
- tweet_ids: Tab-separated list of Twitter IDs that shared this article
```

### Key Characteristics

#### Strengths
- **Social Network Integration**: Each article includes Twitter engagement data, perfect for network analysis
- **Multi-source Verification**: Data from both GossipCop (entertainment) and PolitiFact (political fact-checking)
- **Balanced Dataset**: Good distribution between real and fake news across both domains
- **Network Analysis Ready**: Tweet IDs enable construction of information propagation networks

#### Data Distribution
- **GossipCop Domain**: 22,140 articles (95.4%)
  - Real: 16,817 (75.9%)
  - Fake: 5,323 (24.1%)
- **PolitiFact Domain**: 1,056 articles (4.6%)
  - Real: 624 (59.1%)
  - Fake: 432 (40.9%)

### Applications in Game Theory Framework

#### 1. **Network Analysis Component**
- Use tweet_ids to construct social media propagation networks
- Analyze information cascade patterns
- Model influence propagation through social networks
- Study network topology effects on misinformation spread

#### 2. **Game Theory Modeling**
- **Spreader Behavior**: Analyze which types of content get shared more frequently
- **Platform Strategies**: Study engagement patterns across different news types
- **Influence Metrics**: Calculate network centrality measures for key spreaders

#### 3. **Multi-Domain Analysis**
- Compare propagation patterns between entertainment (GossipCop) and political (PolitiFact) domains
- Analyze domain-specific spreading strategies
- Develop domain-adaptive game theory models

### Recommended Usage Patterns

```python
# Example: Network construction for game theory analysis
import pandas as pd
import networkx as nx

# Load FakeNewsNet data
gossipcop_fake = pd.read_csv('gossipcop_fake.csv')
politifact_real = pd.read_csv('politifact_real.csv')

# Create information propagation network
def build_propagation_network(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        tweet_ids = row['tweet_ids'].split('\t')
        # Add nodes and edges based on tweet propagation
        # This enables game theory analysis of spreader networks
    return G
```

---

## 2. Kaggle Fake News Dataset

### Location
`/home/ghost/fake-news-game-theory/data/raw/kaggle_fake_news/`

### Description
A comprehensive text classification dataset focused on news articles with full text content. This dataset is ideal for training advanced NLP models and developing robust fake news classifiers for the game theory framework.

### Dataset Structure

#### Files
- `Fake.csv` (23,489 records) - 62.8 MB
- `True.csv` (21,417 records) - 53.6 MB

**Total Records**: 44,906 articles

#### Data Schema
```
- title: Article headline
- text: Full article content (rich text data)
- subject: Article category/topic
- date: Publication date
```

### Key Characteristics

#### Strengths
- **Rich Text Content**: Complete article text enables sophisticated NLP analysis
- **Balanced Dataset**: Nearly equal distribution of fake (52.3%) and real (47.7%) news
- **Topic Diversity**: Multiple subject categories for comprehensive training
- **Temporal Information**: Date stamps enable temporal analysis of misinformation trends

#### Content Analysis
- **Average Article Length**: Substantial text content for deep learning models
- **Subject Categories**: Multiple topics including politics, world news, government news, etc.
- **Date Range**: Spans multiple years, enabling temporal trend analysis

### Applications in Game Theory Framework

#### 1. **ML Classifier Development**
- **BERT Training**: Rich text perfect for transformer-based models
- **LSTM Implementation**: Sequential text data for RNN-based classifiers
- **Ensemble Methods**: Large dataset supports complex ensemble approaches

#### 2. **Game Theory Integration**
- **Content Strategy Analysis**: Study what types of content are more likely to be fake
- **Topic-Based Games**: Model different game dynamics across subject categories
- **Temporal Strategy Evolution**: Analyze how fake news strategies change over time

#### 3. **Strategic Content Analysis**
```python
# Example: Content-based strategy analysis for game theory
def analyze_content_strategies(df):
    """Analyze linguistic patterns for game theory modeling"""
    strategies = {
        'emotional_appeal': count_emotional_words(df['text']),
        'complexity': calculate_readability_scores(df['text']),
        'topic_focus': analyze_subject_distribution(df['subject']),
        'temporal_patterns': analyze_date_patterns(df['date'])
    }
    return strategies
```

### Recommended Usage Patterns

#### Machine Learning Pipeline
```python
# Classifier training for game theory decision support
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# Load and preprocess
fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')

# Prepare for BERT training
def prepare_bert_data(fake_df, real_df):
    # Combine and label data
    fake_df['label'] = 0
    real_df['label'] = 1
    combined = pd.concat([fake_df, real_df])

    # Create training data for game theory classifier
    return combined[['text', 'label']]
```

---

## 3. LIAR Dataset

### Location
`/home/ghost/fake-news-game-theory/data/raw/liar_dataset/`

### Description
A benchmark dataset for fake news detection focusing on short political statements with detailed metadata and multi-class truthfulness labels. Developed by William Yang Wang (ACL 2017), this dataset provides fine-grained truth classifications ideal for nuanced game theory modeling.

### Dataset Structure

#### Files
- `train.tsv` (10,268 records) - 2.4 MB
- `test.tsv` (1,282 records) - 301 KB
- `valid.tsv` (1,283 records) - 301 KB
- `README` - Dataset documentation

**Total Records**: 12,833 statements

#### Data Schema (14 columns)
```
Column 1:  ID of the statement ([ID].json)
Column 2:  Label (6 classes: pants-on-fire, false, barely-true, half-true, mostly-true, true)
Column 3:  Statement text
Column 4:  Subject(s)
Column 5:  Speaker name
Column 6:  Speaker's job title
Column 7:  State information
Column 8:  Party affiliation
Column 9:  Barely true count (historical)
Column 10: False count (historical)
Column 11: Half true count (historical)
Column 12: Mostly true count (historical)
Column 13: Pants on fire count (historical)
Column 14: Context (venue/location)
```

### Key Characteristics

#### Strengths
- **Multi-class Labels**: Six levels of truthfulness (not just binary)
- **Rich Metadata**: Speaker information, political affiliations, historical patterns
- **Political Focus**: Concentrated on political statements and fact-checking
- **Historical Context**: Speaker's historical truthfulness patterns
- **Benchmark Quality**: Well-established academic benchmark

#### Label Distribution (6-class classification)
- **True**: Completely accurate statements
- **Mostly True**: Accurate with minor inaccuracies
- **Half True**: Partially accurate statements
- **Barely True**: Contains some truth but mostly misleading
- **False**: Inaccurate statements
- **Pants on Fire**: Ridiculously false statements

### Applications in Game Theory Framework

#### 1. **Multi-Player Game Modeling**
```python
# Example: Speaker reputation modeling for game theory
def model_speaker_reputation(liar_df):
    """Model speaker credibility for game theory analysis"""
    speaker_stats = liar_df.groupby('speaker').agg({
        'barely_true_count': 'sum',
        'false_count': 'sum',
        'half_true_count': 'sum',
        'mostly_true_count': 'sum',
        'pants_on_fire_count': 'sum'
    })

    # Calculate reputation score for game theory utility functions
    speaker_stats['reputation_score'] = calculate_reputation(speaker_stats)
    return speaker_stats
```

#### 2. **Fine-Grained Strategy Analysis**
- **Spreader Strategies**: Model different levels of misinformation spreading
- **Fact-Checker Strategies**: Develop graduated response strategies based on statement severity
- **Platform Strategies**: Implement nuanced content moderation based on truthfulness levels

#### 3. **Political Game Theory**
- **Party Affiliation Dynamics**: Model how political alignment affects spreading behavior
- **Speaker Authority**: Analyze how job titles and positions influence credibility games
- **Geographic Patterns**: Study state-level misinformation propagation patterns

### Advanced Analytics Opportunities

#### Historical Pattern Analysis
```python
# Game theory utility based on historical patterns
def calculate_speaker_utility(row):
    """Calculate game theory utility based on speaker history"""
    total_statements = sum([
        row['barely_true_count'], row['false_count'],
        row['half_true_count'], row['mostly_true_count'],
        row['pants_on_fire_count']
    ])

    if total_statements == 0:
        return 0.5  # Neutral for new speakers

    # Weight different truthfulness levels
    weights = {
        'pants_on_fire_count': -2,
        'false_count': -1,
        'barely_true_count': -0.5,
        'half_true_count': 0,
        'mostly_true_count': 0.5,
        'true_statements': 1  # Derived from total - others
    }

    # Calculate weighted reputation score
    score = sum(row[col] * weight for col, weight in weights.items() if col in row)
    return score / total_statements
```

---

## Cross-Dataset Integration Strategies

### 1. **Multi-Level Analysis Framework**
- **Statement Level**: Use LIAR for fine-grained truthfulness classification
- **Article Level**: Use Kaggle dataset for full-text analysis
- **Network Level**: Use FakeNewsNet for propagation analysis

### 2. **Ensemble Approach**
```python
class MultiDatasetGameTheory:
    def __init__(self):
        self.liar_model = self.load_liar_classifier()      # Fine-grained truth
        self.kaggle_model = self.load_kaggle_classifier()   # Binary classification
        self.network_model = self.load_network_analyzer()   # Social propagation

    def analyze_statement(self, text, speaker_info=None, network_data=None):
        """Comprehensive analysis using all datasets"""
        truth_level = self.liar_model.predict(text, speaker_info)
        content_score = self.kaggle_model.predict(text)
        propagation_risk = self.network_model.analyze(network_data)

        return self.combine_scores(truth_level, content_score, propagation_risk)
```

### 3. **Game Theory Integration**

#### Player Modeling
- **Spreaders**: Use all datasets to model spreading strategies
  - Content strategy (Kaggle): What types of content to share
  - Truthfulness strategy (LIAR): How much truth to include
  - Network strategy (FakeNewsNet): How to leverage social networks

- **Fact-Checkers**: Multi-level response strategies
  - Priority targeting based on propagation potential (FakeNewsNet)
  - Resource allocation based on content complexity (Kaggle)
  - Response granularity based on truthfulness level (LIAR)

- **Platforms**: Comprehensive moderation strategies
  - Content filtering using ensemble models
  - Network-based detection and intervention
  - User reputation systems based on historical patterns

## Technical Implementation Recommendations

### 1. **Data Preprocessing Pipeline**
```python
class DatasetPreprocessor:
    def process_fakenewsnet(self, df):
        """Prepare FakeNewsNet for network analysis"""
        # Extract and validate tweet IDs
        # Create network adjacency matrices
        # Calculate propagation metrics
        pass

    def process_kaggle(self, fake_df, real_df):
        """Prepare Kaggle dataset for NLP models"""
        # Text cleaning and normalization
        # Feature extraction
        # Train/test splits
        pass

    def process_liar(self, train_df, test_df, valid_df):
        """Prepare LIAR dataset for multi-class classification"""
        # Speaker encoding
        # Historical pattern features
        # Political affiliation encoding
        pass
```

### 2. **Game Theory Implementation**
```python
class FakeNewsGameTheory:
    def __init__(self, datasets):
        self.datasets = datasets
        self.players = ['spreaders', 'fact_checkers', 'platforms']

    def calculate_utilities(self, strategy_profile):
        """Calculate utilities using multi-dataset insights"""
        # Spreader utility: engagement (FakeNewsNet) + believability (Kaggle) + authority (LIAR)
        # Fact-checker utility: accuracy improvement + resource efficiency
        # Platform utility: user engagement - regulatory risk - reputation cost
        pass

    def find_nash_equilibrium(self):
        """Find equilibrium strategies using comprehensive dataset analysis"""
        pass
```

### 3. **Evaluation Metrics**

#### Classification Performance
- **Binary Classification** (Kaggle): Accuracy, Precision, Recall, F1-score
- **Multi-class Classification** (LIAR): Macro/Micro F1, per-class precision/recall
- **Network Prediction** (FakeNewsNet): Propagation accuracy, influence prediction

#### Game Theory Metrics
- **Strategy Stability**: Nash equilibrium existence and uniqueness
- **Social Welfare**: Overall system utility optimization
- **Robustness**: Performance under adversarial conditions

## Conclusion

The three datasets provide complementary strengths for the Game Theory Fake News Detection project:

1. **FakeNewsNet**: Essential for network analysis and understanding social media propagation dynamics
2. **Kaggle Dataset**: Critical for training robust NLP classifiers with comprehensive text analysis
3. **LIAR Dataset**: Valuable for fine-grained truthfulness assessment and political statement analysis

By integrating all three datasets, the project can develop a comprehensive framework that addresses fake news detection from multiple angles: content analysis, social network dynamics, and strategic behavioral modeling. This multi-faceted approach aligns perfectly with the project's game theory focus and provides the data foundation needed for advanced research in combating misinformation.

The combination enables sophisticated game theory models that can:
- Model realistic player behaviors based on actual data patterns
- Develop evidence-based counter-strategies
- Validate theoretical predictions against real-world propagation data
- Create robust systems that work across different types of misinformation and social contexts