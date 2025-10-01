"use client";

import { useState, useEffect, useRef } from "react";
import { motion, useScroll } from "framer-motion";
import { PageHeader } from "@/components/page-header";
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const MethodologyPage = () => {
  const [activeSection, setActiveSection] = useState<string>("");
  const sections = [
    { id: "game-theory", title: "Game Theory Framework" },
    { id: "network-analysis", title: "Network Analysis" },
    { id: "ml-pipeline", title: "ML Pipeline" },
  ];

  const observer = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    observer.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { threshold: 0.5 }
    );

    const elements = document.querySelectorAll("section");
    elements.forEach((el) => observer.current?.observe(el));

    return () => {
      elements.forEach((el) => observer.current?.unobserve(el));
    };
  }, []);

  const spreaderPayoff = `U_{spreader} = P_{spread} * (R_{engagement} - C_{effort}) + (1 - P_{spread}) * (-C_{effort})`;
  const factCheckerPayoff = `U_{checker} = P_{correct} * (R_{accuracy} - C_{effort}) + (1 - P_{correct}) * (-C_{effort} - P_{penalty})`;
  const platformPayoff = `U_{platform} = \sum_{i \in Users} (R_{user_i} - C_{moderation_i})`;
  const propagationFormula = `P(infection) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 * IS + \beta_2 * CQ + \beta_3 * V_u)}}`;

  const traditionalMlCode = `
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Features: Linguistic, Stylistic, User-Level
X_train, y_train = load_data("train")
X_test, y_test = load_data("test")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
`;

  const bertCode = `
import torch
from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer("Fake news is a threat to democracy.", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
`;

  const lstmCode = `
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)
`;

  return (
    <div className="container mx-auto py-12">
      <PageHeader
        title="Research Methodology"
        subtitle="The scientific backbone of our project, detailing the theoretical and technical frameworks."
      />
      <div className="flex">
        <aside className="hidden md:block w-1/4 sticky top-24 self-start">
          <nav>
            <h3 className="text-lg font-semibold mb-4">On this page</h3>
            <ul>
              {sections.map((section) => (
                <li key={section.id}>
                  <a
                    href={`#${section.id}`}
                    className={`block py-2 px-4 rounded-lg transition-colors ${
                      activeSection === section.id
                        ? "bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                        : "text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                    }`}
                  >
                    {section.title}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </aside>

        <main className="w-full md:w-3/4 md:pl-8">
          <motion.section
            id="game-theory"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-16"
          >
            <h2 className="text-3xl font-bold mb-4">Game Theory Framework</h2>
            <p className="mb-4">
              The core of our project is a game-theoretic model that captures the strategic interactions between three key players in the information ecosystem: Fake News Spreaders, Fact-Checkers, and Social Media Platforms.
            </p>
            <h3 className="text-2xl font-semibold mb-2">Payoff Functions</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold">Spreader Utility</h4>
                <BlockMath math={spreaderPayoff} />
              </div>
              <div>
                <h4 className="font-semibold">Fact-Checker Utility</h4>
                <BlockMath math={factCheckerPayoff} />
              </div>
              <div>
                <h4 className="font-semibold">Platform Utility</h4>
                <BlockMath math={platformPayoff} />
              </div>
            </div>
          </motion.section>

          <motion.section
            id="network-analysis"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mb-16"
          >
            <h2 className="text-3xl font-bold mb-4">Network Analysis and Propagation Model</h2>
            <p className="mb-4">
              We model the spread of information through a social network represented as a graph G = (V, E), where V is the set of users (nodes) and E is the set of connections (edges). The propagation of a piece of content is modeled using a probabilistic approach.
            </p>
            <h3 className="text-2xl font-semibold mb-2">Propagation Formula</h3>
            <BlockMath math={propagationFormula} />
            <p className="mt-4">
              Where <InlineMath math="IS" /> is the influence score of the user, <InlineMath math="CQ" /> is the content quality, and <InlineMath math="V_u" /> represents the user's vulnerability to misinformation.
            </p>
          </motion.section>

          <motion.section
            id="ml-pipeline"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mb-16"
          >
            <h2 className="text-3xl font-bold mb-4">The Machine Learning Pipeline</h2>
            <p className="mb-4">
              Our machine learning pipeline is designed to classify content as reliable or misinformation, leveraging a variety of features and model architectures.
            </p>
            <h3 className="text-2xl font-semibold mb-2">Data & Preprocessing</h3>
            <p className="mb-4">
              We use several publicly available datasets, including FakeNewsNet, LIAR, and Kaggle Fake News. The text data is cleaned by removing stop words, special characters, and applying stemming/lemmatization.
            </p>
            <h3 className="text-2xl font-semibold mb-2">Feature Engineering</h3>
            <ul className="list-disc list-inside mb-4">
              <li><b>Linguistic Features:</b> Sentiment, subjectivity, complexity.</li>
              <li><b>Stylistic Features:</b> Use of punctuation, capitalization.</li>
              <li><b>User-Level Features:</b> Account age, number of followers.</li>
              <li><b>Propagation Features:</b> Retweet count, network centrality.</li>
            </ul>
            <h3 className="text-2xl font-semibold mb-2">Classification Models</h3>
            <div className="space-y-8">
              <div>
                <h4 className="font-semibold">Traditional ML (Random Forest)</h4>
                <SyntaxHighlighter language="python" style={vscDarkPlus}>
                  {traditionalMlCode}
                </SyntaxHighlighter>
              </div>
              <div>
                <h4 className="font-semibold">BERT</h4>
                <SyntaxHighlighter language="python" style={vscDarkPlus}>
                  {bertCode}
                </SyntaxHighlighter>
              </div>
              <div>
                <h4 className="font-semibold">LSTM</h4>
                <SyntaxHighlighter language="python" style={vscDarkPlus}>
                  {lstmCode}
                </SyntaxHighlighter>
              </div>
            </div>
          </motion.section>
        </main>
      </div>
    </div>
  );
};

export default MethodologyPage;
