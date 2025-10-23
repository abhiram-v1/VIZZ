import React, { useState, useEffect, useRef } from 'react';
import { socketService } from '../services/socketService';

const MLLearningJourneyPage = () => {
  const [currentSection, setCurrentSection] = useState(0);
  const [showAnimations, setShowAnimations] = useState(false);
  const [experimentResults, setExperimentResults] = useState(null);
  const [isRunningExperiment, setIsRunningExperiment] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const sectionRefs = useRef([]);
  const contentRef = useRef(null);

  // Learning journey sections
  const learningSections = [
    {
      id: 'ml-intro',
      title: '🤖 What is Machine Learning?',
      content: 'Machine Learning is like teaching a computer to recognize patterns and make decisions, just like how you learned to recognize your friends\' faces or predict the weather.',
      analogy: 'Like teaching a child to recognize animals by showing them pictures',
      concepts: [
        {
          type: 'Supervised Learning',
          description: 'Learning with a teacher - you show examples with correct answers',
          example: 'Like learning to read with a teacher showing you "A is for Apple"',
          icon: (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="#3b82f6"/>
              <path d="M19 15L20.09 18.26L23 19L20.09 19.74L19 23L17.91 19.74L15 19L17.91 18.26L19 15Z" fill="#1d4ed8"/>
            </svg>
          ),
          color: '#3b82f6'
        },
        {
          type: 'Unsupervised Learning',
          description: 'Learning without a teacher - finding hidden patterns',
          example: 'Like organizing your toys by color without being told how',
          icon: (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="11" cy="11" r="8" stroke="#10b981" strokeWidth="2"/>
              <path d="M21 21L16.65 16.65" stroke="#10b981" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="11" cy="8" r="2" fill="#10b981"/>
              <circle cx="8" cy="14" r="2" fill="#059669"/>
              <circle cx="14" cy="14" r="2" fill="#047857"/>
            </svg>
          ),
          color: '#10b981'
        },
        {
          type: 'Reinforcement Learning',
          description: 'Learning through trial and error with rewards',
          example: 'Like learning to play a video game by trying different moves',
          icon: (
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="3" y="3" width="18" height="18" rx="2" fill="#f59e0b" opacity="0.1"/>
              <rect x="6" y="6" width="12" height="12" rx="1" fill="#f59e0b"/>
              <circle cx="9" cy="9" r="1" fill="#ffffff"/>
              <circle cx="15" cy="9" r="1" fill="#ffffff"/>
              <path d="M9 12H15" stroke="#ffffff" strokeWidth="1" strokeLinecap="round"/>
              <path d="M10 15H14" stroke="#ffffff" strokeWidth="1" strokeLinecap="round"/>
            </svg>
          ),
          color: '#f59e0b'
        }
      ]
    },
    {
      id: 'decision-trees',
      title: '🌳 Decision Trees: The Foundation',
      content: 'Decision trees are like a flowchart that asks yes/no questions to make decisions. They\'re the building blocks of many advanced algorithms.',
      analogy: 'Like a doctor\'s checklist: "Is the patient over 65?" → "Do they have high blood pressure?" → "Are they overweight?"',
      concepts: [
        {
          term: 'Entropy',
          description: 'How messy or uncertain our data is',
          example: 'Like a mixed bag of red and blue marbles - high entropy means it\'s hard to predict what color you\'ll pick',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="3" y="3" width="18" height="18" rx="2" fill="#ef4444" opacity="0.1"/>
              <circle cx="8" cy="8" r="2" fill="#ef4444"/>
              <circle cx="16" cy="8" r="2" fill="#dc2626"/>
              <circle cx="8" cy="16" r="2" fill="#b91c1c"/>
              <circle cx="16" cy="16" r="2" fill="#991b1b"/>
              <path d="M8 8L16 16M16 8L8 16" stroke="#ef4444" strokeWidth="1" strokeLinecap="round"/>
            </svg>
          ),
          color: '#ef4444'
        },
        {
          term: 'Information Gain',
          description: 'How much a question helps us reduce uncertainty',
          example: 'Like asking "Is it an animal?" in 20 Questions - it eliminates half the possibilities',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 3V21H21" stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round"/>
              <path d="M9 9L12 6L15 9L18 6" stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M9 15L12 12L15 15L18 12" stroke="#8b5cf6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="6" cy="6" r="2" fill="#8b5cf6"/>
              <circle cx="18" cy="18" r="2" fill="#7c3aed"/>
            </svg>
          ),
          color: '#8b5cf6'
        },
        {
          term: 'Splitting',
          description: 'Dividing data into groups based on a question',
          example: 'Like separating your toys into "big" and "small" piles',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="#06b6d4"/>
              <path d="M8 8L16 8" stroke="#0891b2" strokeWidth="2" strokeLinecap="round"/>
              <path d="M8 12L16 12" stroke="#0891b2" strokeWidth="2" strokeLinecap="round"/>
              <path d="M8 16L16 16" stroke="#0891b2" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          ),
          color: '#06b6d4'
        }
      ]
    },
    {
      id: 'ensemble-learning',
      title: '👥 Ensemble Learning: Teamwork Makes the Dream Work',
      content: 'Instead of relying on one decision tree, we use many trees working together - like having a team of experts instead of just one doctor.',
      analogy: 'Like a jury of 100 people voting on a decision - the majority is usually right',
      concepts: [
        {
          term: 'Weak Learners',
          description: 'Simple models that are slightly better than random guessing',
          example: 'Like a student who gets 60% on a test - not great alone, but helpful in a group',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="8" fill="#f97316" opacity="0.1"/>
              <circle cx="12" cy="12" r="4" fill="#f97316"/>
              <path d="M8 8L16 16M16 8L8 16" stroke="#f97316" strokeWidth="1" strokeLinecap="round"/>
            </svg>
          ),
          color: '#f97316'
        },
        {
          term: 'Strong Learner',
          description: 'A powerful model created by combining many weak learners',
          example: 'Like a team of 100 students working together to solve a problem',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="10" fill="#22c55e"/>
              <path d="M9 12L11 14L15 10" stroke="#ffffff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="12" r="6" fill="#16a34a"/>
            </svg>
          ),
          color: '#22c55e'
        },
        {
          term: 'Voting',
          description: 'Each tree votes on the final decision',
          example: 'Like asking 10 friends for movie recommendations and picking the most popular one',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="3" y="4" width="18" height="16" rx="2" fill="#6366f1" opacity="0.1"/>
              <rect x="5" y="6" width="14" height="12" rx="1" fill="#6366f1"/>
              <circle cx="9" cy="10" r="1" fill="#ffffff"/>
              <circle cx="15" cy="10" r="1" fill="#ffffff"/>
              <path d="M9 12H15" stroke="#ffffff" strokeWidth="1" strokeLinecap="round"/>
              <path d="M9 14H15" stroke="#ffffff" strokeWidth="1" strokeLinecap="round"/>
            </svg>
          ),
          color: '#6366f1'
        }
      ]
    },
    {
      id: 'boosting-algorithms',
      title: '🚀 Boosting: Learning from Mistakes',
      content: 'Boosting is like learning from your mistakes - each new tree focuses on the cases that previous trees got wrong.',
      analogy: 'Like a student who studies harder on topics they got wrong in the last test',
      concepts: [
        {
          algorithm: 'AdaBoost',
          description: 'Adaptive Boosting - adjusts weights based on performance',
          example: 'Like a teacher who gives more attention to students who are struggling',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="#ec4899"/>
              <path d="M19 15L20.09 18.26L23 19L20.09 19.74L19 23L17.91 19.74L15 19L17.91 18.26L19 15Z" fill="#db2777"/>
              <circle cx="12" cy="12" r="3" fill="#ffffff"/>
            </svg>
          ),
          color: '#ec4899'
        },
        {
          algorithm: 'Gradient Boosting',
          description: 'Uses gradient descent to minimize errors',
          example: 'Like adjusting your bike riding technique after each fall',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 3V21H21" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round"/>
              <path d="M9 9L12 6L15 9L18 6" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M9 15L12 12L15 15L18 12" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="6" cy="6" r="2" fill="#14b8a6"/>
              <circle cx="18" cy="18" r="2" fill="#0d9488"/>
            </svg>
          ),
          color: '#14b8a6'
        },
        {
          algorithm: 'XGBoost',
          description: 'Extreme Gradient Boosting - optimized for speed and performance',
          example: 'Like a professional athlete who practices the most efficient techniques',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="#f59e0b"/>
              <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="#fbbf24" opacity="0.6"/>
              <circle cx="12" cy="12" r="2" fill="#ffffff"/>
            </svg>
          ),
          color: '#f59e0b'
        }
      ]
    },
    {
      id: 'experiment',
      title: '🧪 Let\'s Experiment: Compare the Algorithms!',
      content: 'Now let\'s see how different algorithms perform on real data. We\'ll train multiple models and compare their accuracy, precision, and recall.',
      analogy: 'Like testing different study methods to see which one helps students learn best',
      experiment: true
    }
  ];

  // Animation and visibility effects
  useEffect(() => {
    setIsVisible(true);
    setShowAnimations(true);
    
    // Staggered animation for content
    const timer = setTimeout(() => {
      setAnimationStep(1);
    }, 300);
    
    return () => clearTimeout(timer);
  }, [currentSection]);

  const runExperiment = async () => {
    setIsRunningExperiment(true);
    setExperimentResults(null);
    
    // Simulate experiment results with realistic timing
    const steps = [
      { delay: 500, step: 'Training Single Tree...' },
      { delay: 1000, step: 'Training AdaBoost...' },
      { delay: 1500, step: 'Training Gradient Boosting...' },
      { delay: 2000, step: 'Training XGBoost...' },
      { delay: 2500, step: 'Analyzing Results...' }
    ];
    
    steps.forEach(({ delay, step }) => {
      setTimeout(() => {
        console.log(step);
      }, delay);
    });
    
    setTimeout(() => {
      setExperimentResults({
        singleTree: { accuracy: 0.78, precision: 0.75, recall: 0.72, f1: 0.73 },
        adaboost: { accuracy: 0.85, precision: 0.82, recall: 0.80, f1: 0.81 },
        gradientBoosting: { accuracy: 0.88, precision: 0.86, recall: 0.84, f1: 0.85 },
        xgboost: { accuracy: 0.90, precision: 0.89, recall: 0.87, f1: 0.88 }
      });
      setIsRunningExperiment(false);
    }, 3000);
  };

  const nextSection = () => {
    if (currentSection < learningSections.length - 1) {
      setCurrentSection(currentSection + 1);
    }
  };

  const prevSection = () => {
    if (currentSection > 0) {
      setCurrentSection(currentSection - 1);
    }
  };

  const currentStep = learningSections[currentSection];

  return (
    <div className={`ml-learning-journey ${isVisible ? 'visible' : ''}`}>
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="hero-title-main">Machine Learning</span>
            <span className="hero-title-sub">Learning Journey</span>
          </h1>
          <p className="hero-description">
            Master machine learning concepts from basics to advanced boosting algorithms
            through interactive visualizations and real-world examples.
          </p>
          <div className="hero-stats">
            <div className="stat-item">
              <div className="stat-number">5</div>
              <div className="stat-label">Learning Modules</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">4</div>
              <div className="stat-label">Algorithms</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">100%</div>
              <div className="stat-label">Interactive</div>
            </div>
          </div>
        </div>
        <div className="hero-visual">
          <div className="floating-elements">
            <div className="floating-element element-1">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="#667eea"/>
                <path d="M19 15L20.09 18.26L23 19L20.09 19.74L19 23L17.91 19.74L15 19L17.91 18.26L19 15Z" fill="#764ba2"/>
                <path d="M5 11L6.09 13.26L8 14L6.09 14.74L5 17L3.91 14.74L2 14L3.91 13.26L5 11Z" fill="#f59e0b"/>
              </svg>
            </div>
            <div className="floating-element element-2">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22L6.66 19.7C7.14 19.87 7.64 20 8 20C19 20 22 3 22 3C21 5 14 5.25 9 6.25C4 7.25 2 11.5 2 13.5C2 15.5 3.75 16.5 4.5 16.5C5.25 16.5 6 16 6 15.5C6 15 5.5 14.5 5 14.5C4.5 14.5 4 15 4 15.5C4 16 4.5 16.5 5 16.5C5.5 16.5 6 16 6 15.5C6 15 5.5 14.5 5 14.5C4.5 14.5 4 15 4 15.5C4 16 4.5 16.5 5 16.5" fill="#10b981"/>
                <circle cx="12" cy="12" r="3" fill="#059669"/>
                <path d="M9 1L12 4L15 1" stroke="#10b981" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="floating-element element-3">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="3" width="18" height="18" rx="2" fill="#667eea" opacity="0.1"/>
                <path d="M9 9H15V15H9V9Z" fill="#667eea"/>
                <path d="M3 9H9V15H3V9Z" fill="#764ba2"/>
                <path d="M15 3H21V9H15V3Z" fill="#f59e0b"/>
                <path d="M9 15H15V21H9V15Z" fill="#10b981"/>
              </svg>
            </div>
            <div className="floating-element element-4">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="#f59e0b"/>
                <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="#fbbf24" opacity="0.6"/>
              </svg>
            </div>
            <div className="floating-element element-5">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="10" fill="#ef4444"/>
                <circle cx="12" cy="12" r="6" fill="#ffffff"/>
                <circle cx="12" cy="12" r="2" fill="#ef4444"/>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className={`journey-progress ${isVisible ? 'animate-in' : ''}`}>
        <div className="progress-header">
          <h3>Learning Progress</h3>
          <div className="progress-stats">
            <span className="current-step">{currentSection + 1}</span>
            <span className="total-steps">/ {learningSections.length}</span>
          </div>
        </div>
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${((currentSection + 1) / learningSections.length) * 100}%` }}
          ></div>
        </div>
        <div className="progress-text">
          {currentStep.title}
        </div>
      </div>

      {/* Main Content */}
      <div className={`journey-content ${isVisible ? 'animate-in' : ''}`} ref={contentRef}>
        <div className="section-header">
          <h1 className={`section-title ${animationStep ? 'animate-in' : ''}`}>
            {currentStep.title}
          </h1>
          <p className={`section-description ${animationStep ? 'animate-in' : ''}`}>
            {currentStep.content}
          </p>
          {currentStep.analogy && (
            <div className={`analogy-box ${animationStep ? 'animate-in' : ''}`}>
              <div className="analogy-icon">💡</div>
              <div className="analogy-text">
                <strong>Think of it like this:</strong> {currentStep.analogy}
              </div>
            </div>
          )}
        </div>

        {/* Section-specific content */}
        {currentStep.id === 'ml-intro' && (
          <div className="ml-types-visualization">
            <h3>🎯 Three Types of Machine Learning</h3>
            <div className="ml-types-grid">
              {currentStep.concepts.map((concept, index) => (
                <div key={index} className="ml-type-card" style={{ borderColor: concept.color }}>
                  <div className="type-icon" style={{ color: concept.color }}>
                    {concept.icon}
                  </div>
                  <h4>{concept.type}</h4>
                  <p>{concept.description}</p>
                  <div className="type-example">
                    <strong>Example:</strong> {concept.example}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {currentStep.id === 'decision-trees' && (
          <div className="decision-tree-explanation">
            <h3>🌳 How Decision Trees Work</h3>
            <div className="tree-concepts">
              {currentStep.concepts.map((concept, index) => (
                <div key={index} className="concept-card" style={{ borderLeftColor: concept.color }}>
                  <div className="concept-header">
                    <span className="concept-icon" style={{ color: concept.color }}>
                      {concept.icon}
                    </span>
                    <h4>{concept.term}</h4>
                  </div>
                  <p>{concept.description}</p>
                  <div className="concept-example">
                    <strong>Real-world example:</strong> {concept.example}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="tree-visualization">
              <h4>🎯 Interactive Tree Building</h4>
              <div className="tree-demo">
                <div className="tree-node root">
                  <div className="node-content">Age &gt; 65?</div>
                  <div className="tree-branches">
                    <div className="branch left">Yes → High Risk</div>
                    <div className="branch right">No → Check BMI</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="cricket-example">
              <h4>🏏 Real-World Example: Cricket Decision Tree</h4>
              <p className="example-intro">
                Let's see how a decision tree works with a simple cricket scenario!
              </p>
              <div className="cricket-tree-container">
                <div className="cricket-tree">
                  <div className="cricket-node root-node">
                    <div className="node-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8 2L9 8L15 8L16 2" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                        <path d="M8 8L9 14L15 14L16 8" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                        <path d="M8 14L9 20L15 20L16 14" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                        <circle cx="12" cy="12" r="2" fill="#ffffff"/>
                      </svg>
                    </div>
                    <div className="node-question">Is it raining?</div>
                  </div>
                  
                  <div className="cricket-branches">
                    <div className="cricket-branch yes-branch">
                      <div className="branch-label">Yes</div>
                      <div className="cricket-node umbrella-node">
                        <div className="node-icon">
                          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2L13 8L11 8L12 2Z" fill="#ffffff"/>
                            <path d="M8 8C8 8 10 6 12 8C14 6 16 8 16 8" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                            <path d="M12 8L12 20" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                            <circle cx="12" cy="8" r="1" fill="#ffffff"/>
                          </svg>
                        </div>
                        <div className="node-question">Do we have an umbrella?</div>
                      </div>
                      
                      <div className="umbrella-branches">
                        <div className="umbrella-branch yes-umbrella">
                          <div className="branch-label">Yes</div>
                          <div className="cricket-leaf play-leaf">
                            <div className="leaf-icon">
                              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2L13 6L11 6L12 2Z" fill="#059669"/>
                                <path d="M8 6L16 6" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                                <path d="M10 6L10 18" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                                <path d="M14 6L14 18" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                                <path d="M10 18L14 18" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                              </svg>
                            </div>
                            <div className="leaf-decision">Go play cricket!</div>
                            <div className="leaf-reason">Protected from rain</div>
                          </div>
                        </div>
                        
                        <div className="umbrella-branch no-umbrella">
                          <div className="branch-label">No</div>
                          <div className="cricket-leaf no-play-leaf">
                            <div className="leaf-icon">
                              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M3 9L12 2L21 9V20C21 20.55 20.55 21 20 21H4C3.45 21 3 20.55 3 20V9Z" fill="#dc2626"/>
                                <path d="M9 9V21" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                                <path d="M15 9V21" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                                <path d="M9 12H15" stroke="#ffffff" strokeWidth="2" strokeLinecap="round"/>
                                <circle cx="12" cy="15" r="1" fill="#ffffff"/>
                              </svg>
                            </div>
                            <div className="leaf-decision">Stay home</div>
                            <div className="leaf-reason">Will get wet without umbrella</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="cricket-branch no-branch">
                      <div className="branch-label">No</div>
                      <div className="cricket-leaf play-leaf">
                        <div className="leaf-icon">
                          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2L13 6L11 6L12 2Z" fill="#059669"/>
                            <path d="M8 6L16 6" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                            <path d="M10 6L10 18" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                            <path d="M14 6L14 18" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                            <path d="M10 18L14 18" stroke="#059669" strokeWidth="2" strokeLinecap="round"/>
                          </svg>
                        </div>
                        <div className="leaf-decision">Go play cricket!</div>
                        <div className="leaf-reason">Perfect weather for cricket</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="cricket-explanation">
                <h5>🧠 How This Decision Tree Works:</h5>
                <div className="explanation-steps">
                  <div className="explanation-step">
                    <div className="step-number">1</div>
                    <div className="step-content">
                      <strong>First Question:</strong> "Is it raining?" - This is the most important question because rain affects everything else.
                    </div>
                  </div>
                  <div className="explanation-step">
                    <div className="step-number">2</div>
                    <div className="step-content">
                      <strong>If No Rain:</strong> Go play cricket! No need to ask more questions.
                    </div>
                  </div>
                  <div className="explanation-step">
                    <div className="step-number">3</div>
                    <div className="step-content">
                      <strong>If Rain:</strong> Ask "Do we have an umbrella?" - This helps decide if we can still play.
                    </div>
                  </div>
                  <div className="explanation-step">
                    <div className="step-number">4</div>
                    <div className="step-content">
                      <strong>Final Decision:</strong> Based on the answers, we get a clear decision: Play or Stay Home.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentStep.id === 'ensemble-learning' && (
          <div className="ensemble-explanation">
            <h3>👥 Ensemble Learning in Action</h3>
            <div className="ensemble-concepts">
              {currentStep.concepts.map((concept, index) => (
                <div key={index} className="concept-card" style={{ borderLeftColor: concept.color }}>
                  <div className="concept-header">
                    <span className="concept-icon" style={{ color: concept.color }}>
                      {concept.icon}
                    </span>
                    <h4>{concept.term}</h4>
                  </div>
                  <p>{concept.description}</p>
                  <div className="concept-example">
                    <strong>Real-world example:</strong> {concept.example}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="ensemble-visualization">
              <h4>🗳️ How Voting Works</h4>
              <div className="voting-demo">
                <div className="trees-row">
                  <div className="tree-voter">Tree 1: "High Risk"</div>
                  <div className="tree-voter">Tree 2: "Low Risk"</div>
                  <div className="tree-voter">Tree 3: "High Risk"</div>
                  <div className="tree-voter">Tree 4: "High Risk"</div>
                  <div className="tree-voter">Tree 5: "Low Risk"</div>
                </div>
                <div className="voting-result">
                  <div className="result-box">
                    <strong>Final Decision: "High Risk"</strong>
                    <div className="vote-count">3 out of 5 trees voted "High Risk"</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentStep.id === 'boosting-algorithms' && (
          <div className="boosting-explanation">
            <h3>🚀 Boosting Algorithms: Learning from Mistakes</h3>
            
            {/* Core Boosting Concept */}
            <div className="boosting-core-concept">
              <h4>🎯 The Core Idea: Team Learning</h4>
              <div className="concept-explanation">
                <p>Imagine you're working on a group project where each person builds on the previous person's work, focusing on fixing their mistakes. That's exactly how boosting works!</p>
                <div className="analogy-visualization">
                  <div className="analogy-step">
                    <div className="analogy-icon">👤</div>
                    <div className="analogy-text">
                      <strong>Student 1:</strong> Makes initial attempt, gets some right, some wrong
                    </div>
                  </div>
                  <div className="analogy-arrow">→</div>
                  <div className="analogy-step">
                    <div className="analogy-icon">👤</div>
                    <div className="analogy-text">
                      <strong>Student 2:</strong> Focuses on the problems Student 1 got wrong
                    </div>
                  </div>
                  <div className="analogy-arrow">→</div>
                  <div className="analogy-step">
                    <div className="analogy-icon">👤</div>
                    <div className="analogy-text">
                      <strong>Student 3:</strong> Fixes the remaining mistakes from Students 1 & 2
                    </div>
                  </div>
                  <div className="analogy-arrow">→</div>
                  <div className="analogy-step">
                    <div className="analogy-icon">👥</div>
                    <div className="analogy-text">
                      <strong>Final Result:</strong> All students vote on the final answer
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* AdaBoost Detailed Explanation */}
            <div className="algorithm-deep-dive">
              <h4>⚖️ AdaBoost: Adaptive Weight Learning</h4>
              <div className="algorithm-visualization">
                <div className="visual-step">
                  <h5>Step 1: Initial Setup</h5>
                  <div className="data-points-demo">
                    <div className="data-point correct">✓</div>
                    <div className="data-point correct">✓</div>
                    <div className="data-point wrong">✗</div>
                    <div className="data-point correct">✓</div>
                    <div className="data-point wrong">✗</div>
                    <div className="data-point correct">✓</div>
                  </div>
                  <p>All samples start with equal weight (1/6 each)</p>
                </div>
                
                <div className="visual-step">
                  <h5>Step 2: First Tree Makes Mistakes</h5>
                  <div className="mistake-highlight">
                    <div className="data-point correct">✓</div>
                    <div className="data-point correct">✓</div>
                    <div className="data-point wrong highlighted">✗</div>
                    <div className="data-point correct">✓</div>
                    <div className="data-point wrong highlighted">✗</div>
                    <div className="data-point correct">✓</div>
                  </div>
                  <p>Tree 1 got 2 samples wrong - these get higher weight for Tree 2</p>
                </div>
                
                <div className="visual-step">
                  <h5>Step 3: Weighted Learning</h5>
                  <div className="weighted-samples">
                    <div className="sample light">Sample 1 (weight: 0.1)</div>
                    <div className="sample light">Sample 2 (weight: 0.1)</div>
                    <div className="sample heavy">Sample 3 (weight: 0.4) ← Focus here!</div>
                    <div className="sample light">Sample 4 (weight: 0.1)</div>
                    <div className="sample heavy">Sample 5 (weight: 0.4) ← Focus here!</div>
                    <div className="sample light">Sample 6 (weight: 0.1)</div>
                  </div>
                  <p>Tree 2 focuses more on the previously misclassified samples</p>
                </div>
              </div>
            </div>

            {/* Gradient Boosting Detailed Explanation */}
            <div className="algorithm-deep-dive">
              <h4>📉 Gradient Boosting: Learning from Residuals</h4>
              <div className="gradient-explanation">
                <div className="residual-demo">
                  <h5>How Residuals Work:</h5>
                  <div className="residual-steps">
                    <div className="residual-step">
                      <div className="step-label">Actual Values:</div>
                      <div className="values">[100, 200, 150, 300, 250]</div>
                    </div>
                    <div className="residual-step">
                      <div className="step-label">Tree 1 Predictions:</div>
                      <div className="values">[120, 180, 140, 280, 230]</div>
                    </div>
                    <div className="residual-step">
                      <div className="step-label">Residuals (Actual - Predicted):</div>
                      <div className="values">[-20, +20, +10, +20, +20]</div>
                    </div>
                    <div className="residual-step">
                      <div className="step-label">Tree 2 learns to predict these residuals!</div>
                    </div>
                  </div>
                </div>
                
                <div className="gradient-process">
                  <h5>🔄 The Gradient Process:</h5>
                  <div className="process-flow">
                    <div className="flow-step">
                      <div className="flow-icon">1️⃣</div>
                      <div className="flow-text">Make initial predictions</div>
                    </div>
                    <div className="flow-arrow">→</div>
                    <div className="flow-step">
                      <div className="flow-icon">2️⃣</div>
                      <div className="flow-text">Calculate errors (residuals)</div>
                    </div>
                    <div className="flow-arrow">→</div>
                    <div className="flow-step">
                      <div className="flow-icon">3️⃣</div>
                      <div className="flow-text">Train new tree on residuals</div>
                    </div>
                    <div className="flow-arrow">→</div>
                    <div className="flow-step">
                      <div className="flow-icon">4️⃣</div>
                      <div className="flow-text">Add tree to ensemble</div>
                    </div>
                    <div className="flow-arrow">→</div>
                    <div className="flow-step">
                      <div className="flow-icon">5️⃣</div>
                      <div className="flow-text">Repeat until satisfied</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* XGBoost Explanation */}
            <div className="algorithm-deep-dive">
              <h4>⚡ XGBoost: Extreme Optimization</h4>
              <div className="xgboost-features">
                <div className="feature-grid">
                  <div className="feature-card">
                    <div className="feature-icon">🚀</div>
                    <h6>Speed Optimizations</h6>
                    <p>Parallel processing and memory optimization for faster training</p>
                  </div>
                  <div className="feature-card">
                    <div className="feature-icon">🛡️</div>
                    <h6>Regularization</h6>
                    <p>Built-in overfitting prevention with L1 and L2 regularization</p>
                  </div>
                  <div className="feature-card">
                    <div className="feature-icon">📊</div>
                    <h6>Advanced Features</h6>
                    <p>Handles missing values and categorical features automatically</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Interactive Decision Boundary Demo */}
            <div className="interactive-demo">
              <h4>🎮 Interactive Decision Boundary Evolution</h4>
              <div className="boundary-demo">
                <div className="demo-controls">
                  <button className="demo-btn">Start Animation</button>
                  <div className="demo-info">
                    <p>Watch how the decision boundary improves with each tree!</p>
                  </div>
                </div>
                <div className="boundary-visualization">
                  <div className="data-scatter">
                    <div className="point red">•</div>
                    <div className="point red">•</div>
                    <div className="point blue">•</div>
                    <div className="point blue">•</div>
                    <div className="point red">•</div>
                    <div className="point blue">•</div>
                  </div>
                  <div className="boundary-line initial">Initial Boundary</div>
                  <div className="boundary-line improved">Improved Boundary</div>
                </div>
              </div>
            </div>

            {/* Training Loss Explanation */}
            <div className="loss-explanation">
              <h4>📈 Understanding Training Loss</h4>
              <div className="loss-visualization">
                <div className="loss-chart">
                  <div className="chart-title">Training Loss Over Time</div>
                  <div className="loss-curve">
                    <div className="curve-point high">Tree 1: 0.8</div>
                    <div className="curve-point medium">Tree 2: 0.6</div>
                    <div className="curve-point low">Tree 3: 0.4</div>
                    <div className="curve-point lowest">Tree 4: 0.2</div>
                  </div>
                </div>
                <div className="loss-explanation-text">
                  <h5>What This Means:</h5>
                  <ul>
                    <li><strong>Decreasing curve = Good!</strong> Each tree is learning from previous mistakes</li>
                    <li><strong>Steep drop = Fast learning</strong> The algorithm is quickly improving</li>
                    <li><strong>Leveling off = Convergence</strong> Adding more trees won't help much</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Algorithm Comparison */}
            <div className="algorithm-comparison">
              <h4>⚖️ Boosting Algorithm Comparison</h4>
              <div className="comparison-table">
                <div className="comparison-header">
                  <div className="metric">Algorithm</div>
                  <div className="metric">Speed</div>
                  <div className="metric">Accuracy</div>
                  <div className="metric">Interpretability</div>
                  <div className="metric">Overfitting Risk</div>
                </div>
                
                <div className="comparison-row adaboost">
                  <div className="algorithm-name">AdaBoost</div>
                  <div className="metric-value">⭐⭐⭐</div>
                  <div className="metric-value">⭐⭐⭐</div>
                  <div className="metric-value">⭐⭐⭐⭐</div>
                  <div className="metric-value">⭐⭐</div>
                </div>
                
                <div className="comparison-row gradient">
                  <div className="algorithm-name">Gradient Boosting</div>
                  <div className="metric-value">⭐⭐</div>
                  <div className="metric-value">⭐⭐⭐⭐</div>
                  <div className="metric-value">⭐⭐⭐</div>
                  <div className="metric-value">⭐⭐</div>
                </div>
                
                <div className="comparison-row xgboost">
                  <div className="algorithm-name">XGBoost</div>
                  <div className="metric-value">⭐⭐⭐⭐⭐</div>
                  <div className="metric-value">⭐⭐⭐⭐⭐</div>
                  <div className="metric-value">⭐⭐</div>
                  <div className="metric-value">⭐⭐⭐⭐</div>
                </div>
              </div>
              
              <div className="comparison-insights">
                <h5>🎯 Key Insights:</h5>
                <div className="insights-grid">
                  <div className="insight-card">
                    <strong>🏃‍♂️ Speed:</strong> XGBoost is fastest, AdaBoost is moderate, Gradient Boosting is slowest
                  </div>
                  <div className="insight-card">
                    <strong>🎯 Accuracy:</strong> XGBoost usually wins, but Gradient Boosting is very close
                  </div>
                  <div className="insight-card">
                    <strong>🔍 Interpretability:</strong> AdaBoost is easiest to understand, XGBoost is most complex
                  </div>
                  <div className="insight-card">
                    <strong>🛡️ Overfitting:</strong> XGBoost has best protection, others need careful tuning
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentStep.id === 'experiment' && (
          <div className="experiment-section">
            <h3>🧪 Performance Comparison Experiment</h3>
            <p>Let's train different algorithms on the same dataset and compare their performance!</p>
            
            <div className="experiment-controls">
              <button 
                className="run-experiment-btn"
                onClick={runExperiment}
                disabled={isRunningExperiment}
              >
                {isRunningExperiment ? '🔄 Running Experiment...' : '🚀 Run Performance Comparison'}
              </button>
            </div>

            {isRunningExperiment && (
              <div className="experiment-progress">
                <div className="progress-steps">
                  <div className="progress-step active">Training Single Tree...</div>
                  <div className="progress-step">Training AdaBoost...</div>
                  <div className="progress-step">Training Gradient Boosting...</div>
                  <div className="progress-step">Training XGBoost...</div>
                  <div className="progress-step">Analyzing Results...</div>
                </div>
              </div>
            )}

            {experimentResults && (
              <div className="experiment-results">
                <h4>📊 Performance Results</h4>
                <div className="results-grid">
                  <div className="results-header">
                    <div className="metric">Algorithm</div>
                    <div className="metric">Accuracy</div>
                    <div className="metric">Precision</div>
                    <div className="metric">Recall</div>
                    <div className="metric">F1-Score</div>
                  </div>
                  
                  <div className="result-row single-tree">
                    <div className="algorithm-name">Single Decision Tree</div>
                    <div className="metric-value">{(experimentResults.singleTree.accuracy * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.singleTree.precision * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.singleTree.recall * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.singleTree.f1 * 100).toFixed(1)}%</div>
                  </div>
                  
                  <div className="result-row adaboost">
                    <div className="algorithm-name">AdaBoost</div>
                    <div className="metric-value">{(experimentResults.adaboost.accuracy * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.adaboost.precision * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.adaboost.recall * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.adaboost.f1 * 100).toFixed(1)}%</div>
                  </div>
                  
                  <div className="result-row gradient-boosting">
                    <div className="algorithm-name">Gradient Boosting</div>
                    <div className="metric-value">{(experimentResults.gradientBoosting.accuracy * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.gradientBoosting.precision * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.gradientBoosting.recall * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.gradientBoosting.f1 * 100).toFixed(1)}%</div>
                  </div>
                  
                  <div className="result-row xgboost">
                    <div className="algorithm-name">XGBoost</div>
                    <div className="metric-value">{(experimentResults.xgboost.accuracy * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.xgboost.precision * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.xgboost.recall * 100).toFixed(1)}%</div>
                    <div className="metric-value">{(experimentResults.xgboost.f1 * 100).toFixed(1)}%</div>
                  </div>
                </div>
                
                <div className="results-analysis">
                  <h4>🔍 Analysis</h4>
                  <div className="analysis-points">
                    <div className="analysis-point">
                      <strong>🏆 Best Performer:</strong> XGBoost achieved the highest accuracy (90.0%)
                    </div>
                    <div className="analysis-point">
                      <strong>📈 Improvement:</strong> Boosting algorithms significantly outperform single trees
                    </div>
                    <div className="analysis-point">
                      <strong>⚡ Speed vs Accuracy:</strong> XGBoost provides the best balance of performance and efficiency
                    </div>
                    <div className="analysis-point">
                      <strong>🎯 Key Insight:</strong> Ensemble methods work because they combine different perspectives
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="journey-navigation">
        <button 
          className="nav-btn prev-btn"
          onClick={prevSection}
          disabled={currentSection === 0}
        >
          ← Previous
        </button>
        
        <div className="section-indicators">
          {learningSections.map((_, index) => (
            <div 
              key={index}
              className={`indicator ${index === currentSection ? 'active' : ''} ${index < currentSection ? 'completed' : ''}`}
              onClick={() => setCurrentSection(index)}
            >
              {index + 1}
            </div>
          ))}
        </div>
        
        <button 
          className="nav-btn next-btn"
          onClick={nextSection}
          disabled={currentSection === learningSections.length - 1}
        >
          Next →
        </button>
      </div>
    </div>
  );
};

export default MLLearningJourneyPage;
