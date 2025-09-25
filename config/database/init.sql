-- =============================
-- Fake News Game Theory Database Schema
-- =============================
-- This file initializes the PostgreSQL database schema
-- for the fake news game theory research platform.

-- Create Database (optional if handled by docker-compose)
-- CREATE DATABASE fakenews_db;

-- Connect to the DB
\c fakenews_db;

-- =============================
-- User & Authentication Tables
-- =============================

-- Users (researchers, moderators, fact-checkers, platform admins, etc.)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) CHECK (role IN ('spreader', 'fact_checker', 'platform', 'admin')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =============================
-- News & Content Tables
-- =============================

-- News Articles (true or fake)
CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    label VARCHAR(10) CHECK (label IN ('fake','true','unknown')),
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Classification Results
CREATE TABLE classifications (
    id SERIAL PRIMARY KEY,
    article_id INT REFERENCES news_articles(id) ON DELETE CASCADE,
    model_used VARCHAR(50) NOT NULL,
    prediction VARCHAR(10) CHECK (prediction IN ('fake','true')),
    confidence NUMERIC(5,4),
    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================
-- Social Network / Graph Tables
-- =============================

-- Social Network Users (nodes in graph)
CREATE TABLE social_nodes (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    influence_score NUMERIC(5,4),
    credibility_score NUMERIC(5,4),
    activity_level NUMERIC(5,4),
    verified BOOLEAN DEFAULT FALSE
);

-- Social Edges (connections in graph)
CREATE TABLE social_edges (
    id SERIAL PRIMARY KEY,
    source_node INT REFERENCES social_nodes(id) ON DELETE CASCADE,
    target_node INT REFERENCES social_nodes(id) ON DELETE CASCADE,
    trust NUMERIC(5,4),
    interaction_strength NUMERIC(5,4),
    connection_type VARCHAR(50)
);

-- =============================
-- Game Theory / Simulation Tables
-- =============================

-- Simulation Metadata
CREATE TABLE simulations (
    id UUID PRIMARY KEY,
    created_by INT REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'pending',
    parameters JSONB,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Simulation Results
CREATE TABLE simulation_results (
    id SERIAL PRIMARY KEY,
    simulation_id UUID REFERENCES simulations(id) ON DELETE CASCADE,
    round_number INT,
    total_infected INT,
    newly_infected INT,
    infection_rate NUMERIC(5,4),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Payoff Matrices
CREATE TABLE payoffs (
    id SERIAL PRIMARY KEY,
    simulation_id UUID REFERENCES simulations(id) ON DELETE CASCADE,
    player_role VARCHAR(20) CHECK (player_role IN ('spreader','fact_checker','platform')),
    payoff_value NUMERIC(10,4),
    strategy JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================
-- Indexes & Optimization
-- =============================

-- Indexes for performance
CREATE INDEX idx_news_label ON news_articles(label);
CREATE INDEX idx_classifications_article ON classifications(article_id);
CREATE INDEX idx_edges_source ON social_edges(source_node);
CREATE INDEX idx_edges_target ON social_edges(target_node);
CREATE INDEX idx_results_simulation ON simulation_results(simulation_id);

-- =============================
-- Initial Data (Optional)
-- =============================

-- Create default admin user (password should be changed in production)
INSERT INTO users (username, email, password_hash, role) VALUES
('admin', 'admin@fakenews-research.local', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj0XPrYTy.8e', 'admin');

-- Add sample news articles for testing
INSERT INTO news_articles (title, content, label, source) VALUES
('Breaking: New Research Shows Benefits of Exercise', 'A comprehensive study published today shows that regular exercise improves both physical and mental health...', 'true', 'Health Research Journal'),
('SHOCKING: Scientists Discover Chocolate Cures All Diseases', 'In a groundbreaking study that will revolutionize medicine, researchers claim chocolate can cure every known disease...', 'fake', 'FakeNews24'),
('Local Weather Update', 'Tomorrow will be partly cloudy with temperatures reaching 75°F...', 'true', 'Weather Service');

-- =============================
-- Database Functions & Triggers
-- =============================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update updated_at for users table
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();