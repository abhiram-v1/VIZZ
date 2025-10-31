import React, { useState, useEffect, useRef } from 'react';
import { socketService } from '../services/socketService';
import { Icon } from '../components/Icons';
import * as d3 from 'd3';

const MLLearningJourneyPage = () => {
  const [currentSection, setCurrentSection] = useState(0);
  const [showAnimations, setShowAnimations] = useState(false);
  const [experimentResults, setExperimentResults] = useState(null);
  const [isRunningExperiment, setIsRunningExperiment] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [activeBoostingTab, setActiveBoostingTab] = useState('overview'); // 'overview', 'adaboost', 'gradient', 'xgboost', 'comparison'
  const [adaboostStep, setAdaboostStep] = useState(0); // For AdaBoost step navigation
  const sectionRefs = useRef([]);
  const contentRef = useRef(null);

  // Learning journey sections
  const learningSections = [
    {
      id: 'ml-intro',
      title: 'What is Machine Learning?',
      titleIcon: 'ml',
      content: 'Machine Learning helps us learn patterns from student data (test scores, homework grades, attendance, etc.) so we can predict academic performance reliably.',
      analogy: 'Like a teacher learning patterns after seeing thousands of student cases',
      concepts: [
        {
          type: 'Supervised Learning',
          description: 'Learning from labeled examples (inputs with known outcomes)',
          example: 'Predicting student final grades from test scores and homework when we know past students\' outcomes',
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
          description: 'Finding natural structure in data without labels',
          example: 'Grouping students by study patterns and behavior without knowing their grades',
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
          description: 'Learning by taking actions and receiving feedback over time',
          example: 'Adaptive learning system that adjusts difficulty based on whether students answer correctly',
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
      title: 'Decision Trees: The Foundation',
      titleIcon: 'tree',
      content: 'Decision trees split students by simple questions (thresholds) to reach a performance decision. They\'re the building blocks of many boosted models.',
      analogy: 'Like asking questions in a classroom: "Are you taller than 5 feet?" → "Do you play basketball?" → "Are you on the basketball team?"',
      concepts: [
        {
          term: 'Entropy',
          description: 'How mixed or uncertain the groups are',
          example: 'Like a classroom where half the students are tall and half are short = very mixed (high entropy). But if almost everyone is tall = not mixed at all (low entropy)',
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
          description: 'How much clearer things get after asking a question',
          example: 'Asking "Are you taller than 5 feet?" in that classroom separates tall and short students really well - that\'s big information gain! A bad question like "Do you like pizza?" wouldn\'t help sort by height at all',
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
          description: 'Dividing the group into smaller subgroups using a question',
          example: 'In our classroom, we split by asking "Are you taller than 5 feet?" → creates two groups: tall students and short students. Then we can ask more questions to split those groups further',
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
      title: 'Ensemble Learning: Many Trees, Better Decisions',
      titleIcon: 'team',
      content: 'Rather than one tree, we combine many trees and let them vote. This reduces individual biases and improves reliability on student performance predictions.',
      analogy: 'Like multiple teachers giving opinions and the consensus guiding the final grade',
      concepts: [
        {
          term: 'Weak Learners',
          description: 'Simple trees that are only slightly better than chance',
          example: 'A shallow tree that captures test score thresholds but misses other performance factors',
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
          description: 'The combined model that outperforms any single tree',
          example: 'An ensemble that uses test scores, homework, attendance, and study hours together to predict performance',
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
          description: 'Each tree votes; majority (or averaged score) is the prediction',
          example: '7/10 trees predict Pass → final decision: Pass',
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
      title: 'Boosting: Focus on What Was Missed',
      titleIcon: 'rocket',
      content: 'Boosting builds trees one after another. Each new tree focuses more on examples that were previously misclassified, steadily reducing errors.',
      analogy: 'Like reviewing test questions you got wrong last time and studying those patterns more carefully',
      concepts: [
        {
          algorithm: 'AdaBoost',
          description: 'Builds an ensemble of weak learners (decision stumps) that adaptively focus on previously misclassified examples',
          example: 'If students who scored high on test 1 were misclassified by Tree 1, Tree 2 is forced to pay more attention to them',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Scale/Balance representing adaptive weights */}
              <path d="M3 12H21" stroke="#ec4899" strokeWidth="2" strokeLinecap="round"/>
              <path d="M12 3V12" stroke="#ec4899" strokeWidth="2" strokeLinecap="round"/>
              <circle cx="5" cy="16" r="2" fill="#ec4899" opacity="0.3"/>
              <circle cx="12" cy="16" r="2" fill="#ec4899"/>
              <circle cx="19" cy="16" r="2" fill="#ec4899" opacity="0.7"/>
              {/* Arrows showing adaptation */}
              <path d="M8 14L10 12L8 10" stroke="#db2777" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
              <path d="M16 14L14 12L16 10" stroke="#db2777" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            </svg>
          ),
          color: '#ec4899'
        },
        {
          algorithm: 'Gradient Boosting',
          description: 'Builds small trees that fix what the model still gets wrong, step by step',
          example: 'Each new tree focuses on remaining mistakes and gently corrects them',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Gradient/improvement curve */}
              <path d="M3 20L7 16L11 18L15 12L19 14L21 10" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
              {/* Correction arrows pointing upward */}
              <path d="M7 18L7 14" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M11 20L11 16" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M15 14L15 10" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              <path d="M19 16L19 12" stroke="#0d9488" strokeWidth="1.5" strokeLinecap="round"/>
              {/* Step indicators */}
              <circle cx="7" cy="14" r="1.5" fill="#14b8a6"/>
              <circle cx="11" cy="16" r="1.5" fill="#14b8a6"/>
              <circle cx="15" cy="10" r="1.5" fill="#14b8a6"/>
              <circle cx="19" cy="12" r="1.5" fill="#14b8a6"/>
            </svg>
          ),
          color: '#14b8a6'
        },
        {
          algorithm: 'XGBoost',
          description: 'A faster, more careful version of Gradient Boosting that helps avoid overfitting',
          example: 'Good when you have many students and need speed with safety checks',
          icon: (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Lightning bolt for speed */}
              <path d="M13 2L8 10H12L11 18L16 10H12L13 2Z" fill="#f59e0b" stroke="#fbbf24" strokeWidth="0.5"/>
              {/* Shield outline for regularization/protection */}
              <path d="M6 8C6 6 8 4 12 4C16 4 18 6 18 8C18 10 16 14 12 18C8 14 6 10 6 8Z" stroke="#fbbf24" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.6"/>
              {/* Speed lines */}
              <path d="M19 6L21 4" stroke="#fbbf24" strokeWidth="1.5" strokeLinecap="round" opacity="0.7"/>
              <path d="M20 7L21.5 5.5" stroke="#fbbf24" strokeWidth="1.5" strokeLinecap="round" opacity="0.7"/>
            </svg>
          ),
          color: '#f59e0b'
        }
      ]
    },
    {
      id: 'experiment',
      title: 'Let\'s Experiment: Compare the Algorithms!',
      titleIcon: 'experiment',
      content: 'Now let\'s see how different algorithms perform on real data. We\'ll train multiple models and compare their accuracy, precision, and recall.',
      analogy: 'Like testing different study methods to see which one helps students learn best',
      experiment: true
    }
  ];

  // Animation and visibility effects
  useEffect(() => {
    setIsVisible(true);
    setShowAnimations(true);
    
    // Scroll to top when section changes - ensures users start reading from the beginning
    window.scrollTo({ top: 0, behavior: 'smooth' });
    // Also scroll the content container to start
    setTimeout(() => {
      if (contentRef.current) {
        contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
    
    // Staggered animation for content
    const timer = setTimeout(() => {
      setAnimationStep(1);
    }, 300);
    
    return () => clearTimeout(timer);
  }, [currentSection]);

  // D3 Tree Visualization
  useEffect(() => {
    // Only render tree when viewing decision trees section
    const decisionTreeSection = learningSections.find(section => section.id === 'decision-trees');
    if (!decisionTreeSection || currentSection !== learningSections.indexOf(decisionTreeSection)) {
      return;
    }

    // Clear previous tree
    d3.select("#decision-tree-svg").selectAll("*").remove();

    // Tree data for classroom/height decision tree (consistent with entropy/info gain examples)
    const treeData = {
      name: "Are you taller than 5 feet?",
      children: [
        {
          name: "Do you play basketball?",
          branchLabel: "Yes",
          children: [
            { name: "Tall athlete group", icon: "🏀" },
            { name: "Tall non-athlete group", icon: "📚" }
          ]
        },
        {
          name: "Short student group",
          branchLabel: "No",
          icon: "👥"
        }
      ]
    };

    const width = 1200;
    const height = 700;
    
    const svgElement = d3.select("#decision-tree-svg")
      .append("svg")
      .attr("viewBox", [0, 0, width, height]);

    // Add gradient definition
    const defs = svgElement.append("defs");
    const gradient = defs.append("linearGradient")
      .attr("id", "nodeGradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "100%");
    
    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#667eea");
    
    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#764ba2");

    const svg = svgElement
      .append("g")
      .attr("transform", "translate(150,50)");

    // Vertical tree layout - swap width and height for vertical orientation
    // Increased size significantly to add more spacing between nodes
    // Increased horizontal spacing (width - 300) and vertical spacing (height - 100)
    const treeLayout = d3.tree()
      .size([width - 300, height - 150])
      .separation((a, b) => {
        // Add extra separation between nodes
        // Siblings get more space, parents and children get more vertical space
        if (a.parent === b.parent) {
          return 1.5; // More space between siblings
        }
        return 1.2; // More space between parent and child
      });
    
    const root = d3.hierarchy(treeData);
    treeLayout(root);

    // Links - for vertical tree, swap x and y coordinates
    svg.selectAll(".link")
      .data(root.links())
      .join("path")
      .attr("class", "link")
      .attr("d", d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y))
      .attr("stroke-width", 3)
      .attr("stroke", "#667eea")
      .attr("opacity", 0.7)
      .attr("fill", "none");

    // Calculate node dimensions based on text length
    const getNodeWidth = (text) => {
      const baseWidth = 180;
      const minWidth = 200;
      const charWidth = 8;
      return Math.max(minWidth, Math.min(baseWidth + text.length * charWidth, 280));
    };

    const getNodeHeight = (hasIcon) => hasIcon ? 60 : 50;

    // Nodes - for vertical tree, swap x and y coordinates
    const node = svg.selectAll(".node")
      .data(root.descendants())
      .join("g")
      .attr("class", "node")
      .attr("transform", d => `translate(${d.x},${d.y})`);

    // Add rectangles with dynamic sizing
    node.each(function(d) {
      const nodeWidth = getNodeWidth(d.data.name);
      const nodeHeight = getNodeHeight(d.data.icon);
      const nodeGroup = d3.select(this);
      
      // Create rectangle
      const rect = nodeGroup.append("rect")
        .attr("x", -nodeWidth / 2)
        .attr("y", -nodeHeight / 2)
        .attr("width", nodeWidth)
        .attr("height", nodeHeight)
        .attr("fill", "url(#nodeGradient)")
        .attr("stroke", "rgba(255, 255, 255, 0.3)")
        .attr("stroke-width", 2)
        .attr("rx", 12)
        .attr("ry", 12)
        .style("cursor", "pointer")
        .on("mouseover", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill", "#764ba2")
            .attr("stroke", "rgba(255, 255, 255, 0.5)")
            .attr("stroke-width", 3);
        })
        .on("mouseout", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill", "url(#nodeGradient)")
            .attr("stroke", "rgba(255, 255, 255, 0.3)")
            .attr("stroke-width", 2);
        });

      // Add icon if available
      if (d.data.icon) {
        nodeGroup.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", -8)
          .attr("font-size", "24px")
          .text(d.data.icon);
      }

      // Add text
      nodeGroup.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", d.data.icon ? 8 : 5)
        .attr("fill", "#ffffff")
        .attr("font-size", "15px")
        .attr("font-weight", "600")
        .style("pointer-events", "none")
        .text(d.data.name);
    });

    // Add branch labels (Yes/No) along the links - positioned to the side
    svg.selectAll(".branch-label")
      .data(root.links())
      .join("g")
      .attr("class", "branch-label-group")
      .each(function(d) {
        const labelGroup = d3.select(this);
        const midX = (d.source.x + d.target.x) / 2;
        const midY = (d.source.y + d.target.y) / 2;
        
        // Determine label text
        let labelText = "";
        if (d.target.data.branchLabel) {
          labelText = d.target.data.branchLabel;
        } else if (d.target.parent && d.target.parent.children) {
          const index = d.target.parent.children.indexOf(d.target);
          if (d.target.parent.data.name === "Do you play basketball?") {
            labelText = index === 0 ? "Yes" : "No";
          }
        }
        
        if (!labelText) return;
        
        // Create background circle/ellipse for label
        labelGroup.append("ellipse")
          .attr("cx", midX + 35)
          .attr("cy", midY)
          .attr("rx", 22)
          .attr("ry", 14)
          .attr("fill", "#667eea")
          .attr("stroke", "rgba(255, 255, 255, 0.5)")
          .attr("stroke-width", 2)
          .attr("opacity", 0.9);
        
        // Add label text
        labelGroup.append("text")
          .attr("x", midX + 35)
          .attr("y", midY)
          .attr("text-anchor", "middle")
          .attr("dy", "0.35em")
          .attr("fill", "#ffffff")
          .attr("font-size", "13px")
          .attr("font-weight", "700")
          .text(labelText);
      });

    // Cleanup function
    return () => {
      d3.select("#decision-tree-svg").selectAll("*").remove();
    };
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
      // Scroll to top when changing sections
      window.scrollTo({ top: 0, behavior: 'smooth' });
      // Also scroll the content container if it exists
      if (contentRef.current) {
        contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  const prevSection = () => {
    if (currentSection > 0) {
      setCurrentSection(currentSection - 1);
      // Scroll to top when changing sections
      window.scrollTo({ top: 0, behavior: 'smooth' });
      // Also scroll the content container if it exists
      if (contentRef.current) {
        contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  const currentStep = learningSections[currentSection];

  return (
    <div className={`ml-learning-journey ${isVisible ? 'visible' : ''}`}>
      {/* Hero Section - Only show on first section */}
      {currentSection === 0 && (
        <div className="hero-section animated-intro">
          <div className="hero-content">
            <h1 className="hero-title animated-title">
              <span className="hero-title-main slide-in-right">Machine Learning</span>
              <span className="hero-title-sub slide-in-right-delay">Journey</span>
            </h1>
            <p className="hero-description fade-in">
              Learn core ML concepts using one consistent context: predicting student performance from academic data — from decision trees to advanced boosting.
            </p>
            <div className="hero-stats fade-in-delay">
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
            {/* Animated Graph Visualization */}
            <div className="graph-visualization">
              <svg className="graph-svg" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet">
                {/* Grid lines */}
                <defs>
                  <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#667eea" stopOpacity="0" />
                    <stop offset="100%" stopColor="#667eea" stopOpacity="1" />
                  </linearGradient>
                </defs>
                {/* Grid */}
                <g className="grid-lines">
                  {[0, 1, 2, 3, 4, 5].map((i) => (
                    <line key={`h-${i}`} x1="0" y1={50 + i * 40} x2="400" y2={50 + i * 40} stroke="rgba(255,255,255,0.1)" strokeWidth="1"/>
                  ))}
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                    <line key={`v-${i}`} x1={i * 50} y1="0" x2={i * 50} y2="300" stroke="rgba(255,255,255,0.1)" strokeWidth="1"/>
                  ))}
                </g>
                {/* Animated learning curve */}
                <path 
                  className="learning-curve" 
                  d="M 0 250 Q 100 200, 200 150 T 400 80" 
                  fill="none" 
                  stroke="url(#lineGradient)" 
                  strokeWidth="4"
                  strokeLinecap="round"
                />
                {/* Data points */}
                <circle className="data-point-graph point-1" cx="50" cy="240" r="6" fill="#667eea">
                  <animate attributeName="opacity" values="0;1;1" dur="1s" begin="0.5s" fill="freeze"/>
                </circle>
                <circle className="data-point-graph point-2" cx="150" cy="180" r="6" fill="#764ba2">
                  <animate attributeName="opacity" values="0;1;1" dur="1s" begin="1s" fill="freeze"/>
                </circle>
                <circle className="data-point-graph point-3" cx="250" cy="120" r="6" fill="#f59e0b">
                  <animate attributeName="opacity" values="0;1;1" dur="1s" begin="1.5s" fill="freeze"/>
                </circle>
                <circle className="data-point-graph point-4" cx="350" cy="90" r="6" fill="#10b981">
                  <animate attributeName="opacity" values="0;1;1" dur="1s" begin="2s" fill="freeze"/>
                </circle>
                {/* Animated labels */}
                <text x="50" y="270" fill="rgba(255,255,255,0.7)" fontSize="12" textAnchor="middle" className="graph-label">
                  <animate attributeName="opacity" values="0;1" dur="0.5s" begin="0.5s" fill="freeze"/>
                  Trees
                </text>
                <text x="350" y="120" fill="rgba(255,255,255,0.7)" fontSize="12" textAnchor="middle" className="graph-label">
                  <animate attributeName="opacity" values="0;1" dur="0.5s" begin="2s" fill="freeze"/>
                  Boosting
                </text>
              </svg>
            </div>
            {/* Floating algorithm icons */}
            <div className="floating-elements">
              <div className="floating-element element-1 slide-right">
                <Icon name="tree" size={32} />
              </div>
              <div className="floating-element element-2 slide-right-delay">
                <Icon name="target" size={32} />
              </div>
              <div className="floating-element element-3 slide-right-delay-2">
                <Icon name="rocket" size={32} />
              </div>
              <div className="floating-element element-4 slide-right-delay-3">
                <Icon name="bolt" size={32} />
              </div>
            </div>
          </div>
        </div>
      )}

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
            {currentStep.titleIcon && <Icon name={currentStep.titleIcon} size={32} style={{ marginRight: '10px', verticalAlign: 'middle' }} />}
            {currentStep.title}
          </h1>
          <p className={`section-description ${animationStep ? 'animate-in' : ''}`}>
            {currentStep.content}
          </p>
          {currentStep.analogy && (
            <div className={`analogy-box ${animationStep ? 'animate-in' : ''}`}>
              <div className="analogy-icon"><Icon name="lightbulb" size={24} /></div>
              <div className="analogy-text">
                <strong>Think of it like this:</strong> {currentStep.analogy}
              </div>
            </div>
          )}
        </div>

        {/* Section-specific content */}
        {currentStep.id === 'ml-intro' && (
          <div className="ml-types-visualization">
            <h3><Icon name="target" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Three Types of Machine Learning</h3>
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
            <h3><Icon name="tree" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> How Decision Trees Work</h3>
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
            
            <div className="supermarket-example">
              <h4><Icon name="hospital" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Real-World Example: Classroom Decision Tree</h4>
              <p className="example-intro">
                Organizing students in a classroom by asking simple questions about height and activities.
              </p>
              <div className="decision-tree-container">
                <div id="decision-tree-svg" className="decision-tree-d3"></div>
              </div>

              <div className="supermarket-explanation">
                <h5><Icon name="brain" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> How This Decision Tree Works:</h5>
                <div className="explanation-steps">
                  <div className="explanation-step">
                    <div className="step-number">1</div>
                    <div className="step-content">
                      <strong>First Question:</strong> "Are you taller than 5 feet?" — this splits the classroom into tall and short students.
                    </div>
                  </div>
                  <div className="explanation-step">
                    <div className="step-number">2</div>
                    <div className="step-content">
                      <strong>If Tall:</strong> Ask "Do you play basketball?" to further organize the tall students into athletes and non-athletes.
                    </div>
                  </div>
                  <div className="explanation-step">
                    <div className="step-number">3</div>
                    <div className="step-content">
                      <strong>Final Groups:</strong> We end up with three organized groups: tall athletes, tall non-athletes, and short students. This helps us understand and organize the classroom better!
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentStep.id === 'ensemble-learning' && (
          <div className="ensemble-explanation">
            <div className="ensemble-intro">
              <h3><Icon name="team" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Ensemble Learning: When Many Minds Work Together</h3>
              <p className="ensemble-intro-text">
                Ensemble learning combines multiple decision trees (or other models) to create a more accurate and reliable prediction system. 
                Instead of relying on a single tree that might overfit or miss important patterns, an ensemble uses the "wisdom of the crowd" principle.
              </p>
            </div>

            <div className="ensemble-comparison-visual">
              <div className="comparison-header">
                <h4><Icon name="scale" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Single Tree vs. Ensemble</h4>
              </div>
              <div className="comparison-grid">
                <div className="comparison-item single-tree">
                  <div className="comparison-icon">
                    <Icon name="tree" size={40} />
                  </div>
                  <h5>Single Decision Tree</h5>
                  <div className="comparison-stats">
                    <div className="stat-item">
                      <span className="stat-label">Accuracy</span>
                      <span className="stat-value">~75%</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Reliability</span>
                      <span className="stat-value">Lower</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Bias Risk</span>
                      <span className="stat-value">Higher</span>
                    </div>
                  </div>
                  <div className="comparison-drawbacks">
                    <p><strong>Limitations:</strong></p>
                    <ul>
                      <li>May overfit to training data</li>
                      <li>Single perspective on patterns</li>
                      <li>Vulnerable to specific data quirks</li>
                    </ul>
                  </div>
                </div>

                <div className="comparison-arrow">
                  <Icon name="right" size={32} />
                </div>

                <div className="comparison-item ensemble">
                  <div className="comparison-icon">
                    <Icon name="team" size={40} />
                  </div>
                  <h5>Ensemble of Trees</h5>
                  <div className="comparison-stats">
                    <div className="stat-item">
                      <span className="stat-label">Accuracy</span>
                      <span className="stat-value">~88-92%</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Reliability</span>
                      <span className="stat-value">Higher</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Bias Risk</span>
                      <span className="stat-value">Lower</span>
                    </div>
                  </div>
                  <div className="comparison-benefits">
                    <p><strong>Advantages:</strong></p>
                    <ul>
                      <li>Multiple perspectives cancel out errors</li>
                      <li>More robust to outliers</li>
                      <li>Better generalization to new data</li>
                      <li>Reduces variance and overfitting</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="ensemble-mechanisms">
              <h4><Icon name="gears" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Key Mechanisms of Ensemble Learning</h4>
              <div className="mechanisms-grid">
                {currentStep.concepts.map((concept, index) => (
                  <div key={index} className="mechanism-card">
                    <div className="mechanism-header">
                      <div className="mechanism-icon-wrapper">
                        <Icon name={index === 0 ? "target" : index === 1 ? "scale" : "vote"} size={32} />
                      </div>
                      <h5>{concept.term}</h5>
                    </div>
                    <p className="mechanism-description">{concept.description}</p>
                    <div className="mechanism-details">
                      <div className="detail-row">
                        <Icon name="target" size={14} style={{ marginRight: '6px' }} />
                        <span><strong>How it works:</strong> {concept.example}</span>
                      </div>
                    </div>
                    {index === 2 && (
                      <div className="mechanism-note">
                        <Icon name="lightbulb" size={14} style={{ marginRight: '6px' }} />
                        <span>In practice, we can use simple majority voting or weighted voting where better-performing trees have more influence.</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="ensemble-visualization-enhanced">
              <h4><Icon name="vote" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Interactive Voting Demonstration</h4>
              <p className="visual-description">Watch how multiple trees work together to make a final prediction. Each tree independently evaluates the student and votes.</p>
              
              <div className="voting-demo-enhanced">
                <div className="trees-panel">
                  <div className="trees-header">
                    <Icon name="tree" size={18} style={{ marginRight: '6px' }} />
                    <span>5 Decision Trees Evaluate Student</span>
                  </div>
                  <div className="trees-grid">
                    {[
                      { id: 1, prediction: "Pass", confidence: 85, reason: "Score 72, Homework 90" },
                      { id: 2, prediction: "Fail", confidence: 62, reason: "Score below 70" },
                      { id: 3, prediction: "Pass", confidence: 78, reason: "Score 68, good attendance" },
                      { id: 4, prediction: "Pass", confidence: 82, reason: "Score 75, perfect homework" },
                      { id: 5, prediction: "Fail", confidence: 55, reason: "Low score" }
                    ].map((tree, idx) => (
                      <div key={tree.id} className="tree-card-enhanced">
                        <div className="tree-card-header">
                          <Icon name="tree" size={16} />
                          <span>Tree {tree.id}</span>
                        </div>
                        <div className={`tree-prediction ${tree.prediction === "Pass" ? "high-risk" : "low-risk"}`}>
                          <span className="prediction-label">{tree.prediction}</span>
                          <span className="confidence-badge">{tree.confidence}%</span>
                        </div>
                        <div className="tree-reason">
                          <Icon name="target" size={12} style={{ marginRight: '4px' }} />
                          <span>{tree.reason}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="voting-process">
                  <div className="vote-arrow">
                    <Icon name="arrowDown" size={24} />
                  </div>
                  <div className="vote-count-visual">
                    <div className="vote-breakdown">
                      <div className="vote-category high-risk-votes">
                        <div className="vote-header">
                          <Icon name="checkCircle" size={16} />
                          <span>Pass Votes</span>
                        </div>
                        <div className="vote-number">3</div>
                        <div className="vote-trees">Trees 1, 3, 4</div>
                      </div>
                      <div className="vote-category low-risk-votes">
                        <div className="vote-header">
                          <Icon name="cross" size={16} />
                          <span>Fail Votes</span>
                        </div>
                        <div className="vote-number">2</div>
                        <div className="vote-trees">Trees 2, 5</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="final-result-enhanced">
                  <div className="result-header">
                    <Icon name="checkCircle" size={24} style={{ marginRight: '8px' }} />
                    <span>Final Ensemble Decision</span>
                  </div>
                  <div className="result-box-enhanced">
                    <div className="result-main">Pass</div>
                    <div className="result-details">
                      <div className="result-margin">Majority: 3-2</div>
                      <div className="result-confidence">Average Confidence: 72.4%</div>
                    </div>
                  </div>
                  <div className="result-explanation">
                    <Icon name="brain" size={14} style={{ marginRight: '6px' }} />
                    <span>The ensemble combines all tree predictions and makes a more reliable final decision than any single tree could.</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="ensemble-principles">
              <h4><Icon name="target" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Why Ensembles Work Better</h4>
              <div className="principles-grid">
                <div className="principle-card">
                  <div className="principle-icon">
                    <Icon name="shield" size={28} />
                  </div>
                  <h5>Error Reduction</h5>
                  <p>Individual tree errors cancel each other out. If Tree 1 makes a mistake on one student, Trees 2-5 may correct it.</p>
                </div>
                <div className="principle-card">
                  <div className="principle-icon">
                    <Icon name="scale" size={28} />
                  </div>
                  <h5>Bias Balancing</h5>
                  <p>Each tree may have different biases. The ensemble averages these biases, leading to a more balanced final model.</p>
                </div>
                <div className="principle-card">
                  <div className="principle-icon">
                    <Icon name="chart" size={28} />
                  </div>
                  <h5>Variance Reduction</h5>
                  <p>Individual trees might vary significantly. By averaging predictions, we reduce the overall variance and get more stable results.</p>
                </div>
                <div className="principle-card">
                  <div className="principle-icon">
                    <Icon name="brain" size={28} />
                  </div>
                  <h5>Complementary Strengths</h5>
                  <p>Different trees capture different patterns. Tree 1 might excel at test score-based decisions, while Tree 2 catches homework-related patterns.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {currentStep.id === 'boosting-algorithms' && (
          <div className="boosting-explanation">
            <h3><Icon name="rocket" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Boosting Algorithms: Learning from Mistakes</h3>
            
            {/* Tab Navigation */}
            <div className="boosting-tabs">
              <button 
                className={`tab-btn ${activeBoostingTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveBoostingTab('overview')}
              >
                <Icon name="target" size={16} style={{ marginRight: '6px' }} />
                Overview
              </button>
              <button 
                className={`tab-btn ${activeBoostingTab === 'adaboost' ? 'active' : ''}`}
                onClick={() => { setActiveBoostingTab('adaboost'); setAdaboostStep(0); }}
              >
                <Icon name="scale" size={16} style={{ marginRight: '6px' }} />
                AdaBoost
              </button>
              <button 
                className={`tab-btn ${activeBoostingTab === 'gradient' ? 'active' : ''}`}
                onClick={() => setActiveBoostingTab('gradient')}
              >
                <Icon name="chart" size={16} style={{ marginRight: '6px' }} />
                Gradient Boosting
              </button>
              <button 
                className={`tab-btn ${activeBoostingTab === 'xgboost' ? 'active' : ''}`}
                onClick={() => setActiveBoostingTab('xgboost')}
              >
                <Icon name="bolt" size={16} style={{ marginRight: '6px' }} />
                XGBoost
              </button>
              <button 
                className={`tab-btn ${activeBoostingTab === 'comparison' ? 'active' : ''}`}
                onClick={() => setActiveBoostingTab('comparison')}
              >
                <Icon name="scale" size={16} style={{ marginRight: '6px' }} />
                Comparison
              </button>
            </div>

            {/* Tab Content */}
            <div className="boosting-tab-content">
              {/* Overview Tab */}
              {activeBoostingTab === 'overview' && (
                <div className="tab-panel">
                  {/* Algorithm Overview Cards */}
                  <div className="boosting-algorithms-grid">
                    {currentStep.concepts.map((algo, index) => (
                      <div key={index} className="boosting-algorithm-card" style={{ borderColor: algo.color }}>
                        <div className="algorithm-icon-large" style={{ color: algo.color }}>
                          {algo.icon}
                        </div>
                        <h4 className="algorithm-name">{algo.algorithm}</h4>
                        <p className="algorithm-description">{algo.description}</p>
                        <div className="algorithm-example">
                          <strong>Example:</strong> {algo.example}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="boosting-core-concept">
                    <h4><Icon name="target" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> The Core Idea: Team Learning</h4>
                    <div className="concept-explanation">
                      <p>Imagine you're working on a group project where each person builds on the previous person's work, focusing on fixing their mistakes. That's exactly how boosting works!</p>
                      <div className="analogy-visualization">
                        <div className="analogy-step">
                          <div className="analogy-icon"><Icon name="user" size={24} /></div>
                          <div className="analogy-text">
                            <strong>Student 1:</strong> Makes initial attempt, gets some right, some wrong
                          </div>
                        </div>
                        <div className="analogy-arrow">→</div>
                        <div className="analogy-step">
                          <div className="analogy-icon"><Icon name="user" size={24} /></div>
                          <div className="analogy-text">
                            <strong>Student 2:</strong> Focuses on the problems Student 1 got wrong
                          </div>
                        </div>
                        <div className="analogy-arrow">→</div>
                        <div className="analogy-step">
                          <div className="analogy-icon"><Icon name="user" size={24} /></div>
                          <div className="analogy-text">
                            <strong>Student 3:</strong> Fixes the remaining mistakes from Students 1 & 2
                          </div>
                        </div>
                        <div className="analogy-arrow">→</div>
                        <div className="analogy-step">
                          <div className="analogy-icon"><Icon name="team" size={24} /></div>
                          <div className="analogy-text">
                            <strong>Final Result:</strong> All students vote on the final answer
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* AdaBoost Tab */}
              {activeBoostingTab === 'adaboost' && (
                <div className="tab-panel">
                  <div className="algorithm-deep-dive">
                    <h4><Icon name="scale" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> AdaBoost: Adaptive Boosting Explained</h4>
                    
                    {/* Core Concept */}
                    <div className="core-concept-box">
                      <h5><Icon name="target" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> What is AdaBoost?</h5>
                      <p><strong>AdaBoost (Adaptive Boosting)</strong> combines multiple <strong>weak learners</strong> (simple decision stumps) into one powerful model. Each stump is a tiny tree with just <strong>one decision</strong> (e.g., "Score > 70?" → Pass or Fail).</p>
                      <p>The "Adaptive" part means it <strong>adapts its focus</strong>: trees added later pay more attention to students that earlier trees got wrong.</p>
                    </div>

                    {/* Step Navigation */}
                    <div className="step-navigation">
                      <h5><Icon name="sync" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> How AdaBoost Works</h5>
                      <div className="step-tabs">
                        {[1, 2, 3, 4, 5, 6].map((stepNum) => (
                          <button
                            key={stepNum}
                            className={`step-tab ${adaboostStep === stepNum - 1 ? 'active' : ''}`}
                            onClick={() => setAdaboostStep(stepNum - 1)}
                          >
                            Step {stepNum}
                          </button>
                        ))}
                      </div>
                      <div className="step-nav-buttons">
                        <button 
                          className="step-nav-btn"
                          onClick={() => setAdaboostStep(Math.max(0, adaboostStep - 1))}
                          disabled={adaboostStep === 0}
                        >
                          <Icon name="left" size={14} style={{ marginRight: '4px' }} /> Previous
                        </button>
                        <button 
                          className="step-nav-btn"
                          onClick={() => setAdaboostStep(Math.min(5, adaboostStep + 1))}
                          disabled={adaboostStep === 5}
                        >
                          Next <Icon name="right" size={14} style={{ marginLeft: '4px' }} />
                        </button>
                      </div>
                    </div>

                    {/* Step Content */}
                    <div className="step-content-wrapper">
                      {/* Step 1 */}
                      {adaboostStep === 0 && (
                        <div className="visual-step">
                          <h6>Step 1: Initial Setup - All Students Start Equal</h6>
                          <div className="data-points-demo">
                            <div className="data-point equal">Student 1 (Score: 45, Weight: 0.17)</div>
                            <div className="data-point equal">Student 2 (Score: 52, Weight: 0.17)</div>
                            <div className="data-point equal">Student 3 (Score: 58, Weight: 0.17)</div>
                            <div className="data-point equal">Student 4 (Score: 68, Weight: 0.17)</div>
                            <div className="data-point equal">Student 5 (Score: 72, Weight: 0.17)</div>
                            <div className="data-point equal">Student 6 (Score: 48, Weight: 0.17)</div>
                          </div>
                          <p><strong>Every student starts equal:</strong> All 6 students have weight = 1/6. The first decision stump trains on this balanced dataset. At this point, we haven't made any predictions yet.</p>
                        </div>
                      )}

                      {/* Step 2 */}
                      {adaboostStep === 1 && (
                        <div className="visual-step">
                          <h6>Step 2: First Stump Makes Predictions</h6>
                          <div className="stump-visualization">
                            <div className="stump-node">
                              <div className="stump-question">Score > 70?</div>
                              <div className="stump-branches">
                                <div className="stump-branch">Yes → Pass</div>
                                <div className="stump-branch">No → Fail</div>
                              </div>
                            </div>
                          </div>
                          <div className="prediction-results">
                            <h6>First Stump Predictions:</h6>
                            <div className="prediction-grid">
                              <div className="prediction-item">
                                <span className="patient-info">Student 1 (Score: 45)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 2 (Score: 52)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 3 (Score: 58)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 4 (Score: 68)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Pass</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 5 (Score: 72)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Pass</span>
                                <span className="actual-label">Actual: Pass</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 6 (Score: 48)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                            </div>
                            <p className="iteration-summary"><strong>First iteration result:</strong> Most predictions are correct, but Student 4 was misclassified! Let's see what happens when we add more complex cases.</p>
                          </div>
                        </div>
                      )}

                      {/* Step 3 */}
                      {adaboostStep === 2 && (
                        <div className="visual-step">
                          <h6>Step 3: Next Iteration - Mistakes Appear!</h6>
                          <p><strong>Context:</strong> We add more challenging students that don't fit the simple "Score > 70" rule. The first stump now makes mistakes.</p>
                          <div className="stump-visualization">
                            <div className="stump-node">
                              <div className="stump-question">Score > 70? (Same rule as before)</div>
                              <div className="stump-branches">
                                <div className="stump-branch">Yes → Pass</div>
                                <div className="stump-branch">No → Fail</div>
                              </div>
                            </div>
                          </div>
                          <div className="prediction-results">
                            <h6>Same Stump with More Complex Students:</h6>
                            <div className="prediction-grid">
                              <div className="prediction-item">
                                <span className="patient-info">Student 1 (Score: 45, Homework: 90)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 2 (Score: 52, Homework: 85)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 3 (Score: 58, Homework: 100)</span>
                                <span className="prediction-result wrong highlighted"><Icon name="cross" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label actual-high">Actual: Pass</span>
                                <span className="mistake-explanation">Stump 1 only saw Score (58) and said Fail, but actual result is Pass because Homework (100) compensates!</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 4 (Score: 68, Homework: 75)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 5 (Score: 70, Homework: 60)</span>
                                <span className="prediction-result wrong highlighted"><Icon name="cross" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Pass</span>
                                <span className="actual-label actual-low">Actual: Fail</span>
                                <span className="mistake-explanation">Stump 1 only saw Score (70) and said Pass, but actual result is Fail because low Homework (60) requires higher test scores!</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 6 (Score: 48, Homework: 70)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                            </div>
                            <p className="iteration-summary"><strong>Key insight:</strong> The first stump (using only "Score > 70?") got <strong>2 students wrong</strong> (Students 3 and 5). These mistakes show that the first stump is too simple - it doesn't see homework grades! This is exactly what AdaBoost learns: we need more information to fix these mistakes.</p>
                          </div>
                        </div>
                      )}

                      {/* Step 4 */}
                      {adaboostStep === 3 && (
                        <div className="visual-step">
                          <h6>Step 4: Increase Weights for Mistakes</h6>
                          <div className="weight-update-explanation">
                            <p><strong>AdaBoost's adaptive mechanism:</strong> Students that were misclassified get their weights increased. This forces the next stump to pay more attention to them!</p>
                          </div>
                          <div className="weighted-samples">
                            <div className="sample-row">
                              <div className="sample-item light">
                                <span className="patient-label">Student 1</span>
                                <span className="weight-label">Weight: 0.10</span>
                                <span className="weight-change">↓ Decreased (was correct)</span>
                              </div>
                              <div className="sample-item light">
                                <span className="patient-label">Student 2</span>
                                <span className="weight-label">Weight: 0.10</span>
                                <span className="weight-change">↓ Decreased (was correct)</span>
                              </div>
                              <div className="sample-item heavy">
                                <span className="patient-label">Student 3</span>
                                <span className="weight-label">Weight: 0.40</span>
                                <span className="weight-change">↑ TRIPLED! (was wrong)</span>
                              </div>
                              <div className="sample-item light">
                                <span className="patient-label">Student 4</span>
                                <span className="weight-label">Weight: 0.10</span>
                                <span className="weight-change">↓ Decreased (was correct)</span>
                              </div>
                              <div className="sample-item heavy">
                                <span className="patient-label">Student 5</span>
                                <span className="weight-label">Weight: 0.40</span>
                                <span className="weight-change">↑ TRIPLED! (was wrong)</span>
                              </div>
                              <div className="sample-item light">
                                <span className="patient-label">Student 6</span>
                                <span className="weight-label">Weight: 0.10</span>
                                <span className="weight-change">↓ Decreased (was correct)</span>
                              </div>
                            </div>
                          </div>
                          <p><strong>Why this works:</strong> When the next stump trains, Students 3 and 5 "count more" because they have higher weights. The algorithm is forced to focus on correcting these mistakes!</p>
                        </div>
                      )}

                      {/* Step 5 */}
                      {adaboostStep === 4 && (
                        <div className="visual-step">
                          <h6>Step 5: Second Stump Corrects the Mistakes</h6>
                          <div className="stump-visualization">
                            <div className="stump-node">
                              <div className="stump-question">Homework > 80?</div>
                              <div className="stump-branches">
                                <div className="stump-branch">Yes → Pass</div>
                                <div className="stump-branch">No → Fail</div>
                              </div>
                            </div>
                          </div>
                          <div className="prediction-results">
                            <h6>Second Stump Predictions (Focusing on High-Weight Students):</h6>
                            <div className="prediction-grid">
                              <div className="prediction-item">
                                <span className="patient-info">Student 1 (Homework: 60)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 2 (Homework: 70)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                              <div className="prediction-item corrected">
                                <span className="patient-info">Student 3 (Homework: 100) ← High weight!</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Pass</span>
                                <span className="actual-label actual-high">Actual: Pass</span>
                                <span className="correction-badge"><Icon name="checkCircle" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> CORRECTED!</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 4 (Homework: 75)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Pass</span>
                              </div>
                              <div className="prediction-item corrected">
                                <span className="patient-info">Student 5 (Homework: 60) ← High weight!</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label actual-low">Actual: Fail</span>
                                <span className="correction-badge"><Icon name="checkCircle" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> CORRECTED!</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 6 (Homework: 70)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                            </div>
                            <p className="iteration-summary"><strong>Success!</strong> The second stump corrected the mistakes from the first stump! Student 3 (high homework) is now correctly identified as Pass, and Student 5 (low homework) is correctly identified as Fail. The weighted training forced this stump to focus on what the first stump got wrong.</p>
                          </div>
                        </div>
                      )}

                      {/* Step 6 */}
                      {adaboostStep === 5 && (
                        <div className="visual-step">
                          <h6>Step 6: Combine All Stumps with Weighted Voting</h6>
                          <div className="ensemble-visual">
                            <div className="ensemble-stumps">
                              <div className="stump-card">
                                <div className="stump-title">Stump 1</div>
                                <div className="stump-rule">"Score > 70?"</div>
                                <div className="stump-weight">Weight: 0.85</div>
                                <div className="stump-accuracy">Accuracy: 67%</div>
                              </div>
                              <div className="stump-card">
                                <div className="stump-title">Stump 2</div>
                                <div className="stump-rule">"Homework > 80?"</div>
                                <div className="stump-weight">Weight: 0.60</div>
                                <div className="stump-accuracy">Accuracy: 83%</div>
                              </div>
                              <div className="stump-card">
                                <div className="stump-title">Stump 3</div>
                                <div className="stump-rule">"Attendance > 90%?"</div>
                                <div className="stump-weight">Weight: 0.70</div>
                                <div className="stump-accuracy">Accuracy: 92%</div>
                              </div>
                            </div>
                            <div className="ensemble-plus">+</div>
                            <div className="final-vote">
                              <div className="vote-title">Final Prediction</div>
                              <div className="vote-explanation">Each stump votes, weighted by its accuracy. More accurate stumps have more influence!</div>
                              <div className="vote-formula">
                                Final = (0.85 × Stump1) + (0.60 × Stump2) + (0.70 × Stump3)
                              </div>
                            </div>
                          </div>
                          <p><strong>Final prediction:</strong> Each stump makes a prediction, but their votes are weighted by accuracy. More accurate stumps have more say. The final prediction is the weighted majority vote - making the ensemble more reliable than any single stump!</p>
                        </div>
                      )}
                    </div>

                    {/* Key Concepts - Only show after viewing all steps */}
                    {adaboostStep === 5 && (
                      <>
                        <div className="key-concepts-box" style={{ marginTop: '2rem' }}>
                          <h5><Icon name="settings" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Key Concepts</h5>
                          <div className="concepts-grid">
                            <div className="concept-item">
                              <strong>Decision Stumps:</strong> Very simple trees with just one split (one question). Much weaker than full trees, but AdaBoost combines many to make them strong.
                            </div>
                            <div className="concept-item">
                              <strong>Adaptive Weights:</strong> Mistakes from one round become the focus of the next round. The algorithm adapts to fix errors iteratively.
                            </div>
                            <div className="concept-item">
                              <strong>Weighted Voting:</strong> Not all stumps are equal. More accurate stumps get higher voting weights in the final prediction.
                            </div>
                            <div className="concept-item">
                              <strong>Sequential Learning:</strong> Trees are built one after another, each learning from the previous tree's mistakes. This is different from Random Forest where trees are independent.
                            </div>
                          </div>
                        </div>

                        {/* Real-World Example */}
                        <div className="real-world-example" style={{ marginTop: '2rem' }}>
                          <h5><Icon name="hospital" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Real-World Example: Student Pass/Fail Prediction</h5>
                          <div className="example-scenario">
                            <p><strong>Scenario:</strong> We want to predict if a student with test score of 70, homework grade of 90, and attendance of 85% will pass or fail.</p>
                            <div className="example-stumps">
                              <div className="example-stump">
                                <strong>Stump 1:</strong> "Score > 70?" → Yes → Votes <strong>Pass</strong> (weight: 0.85)
                              </div>
                              <div className="example-stump">
                                <strong>Stump 2:</strong> "Homework > 80?" → Yes → Votes <strong>Pass</strong> (weight: 0.90)
                              </div>
                              <div className="example-stump">
                                <strong>Stump 3:</strong> "Attendance > 90%?" → No → Votes <strong>Fail</strong> (weight: 0.60)
                              </div>
                            </div>
                            <div className="example-result">
                              <strong>Weighted Vote:</strong> (0.85 × Pass) + (0.90 × Pass) + (0.60 × Fail) = <strong>Pass</strong> wins!
                            </div>
                            <p><strong>Why it works:</strong> Even though Stump 3 says Fail, the first two stumps (which are more accurate and weighted higher) both say Pass. The ensemble makes a more reliable prediction than any single stump alone.</p>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Gradient Boosting Tab */}
              {activeBoostingTab === 'gradient' && (
                <div className="tab-panel">
                  <div className="algorithm-deep-dive">
                    <h4><Icon name="chart" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Gradient Boosting: Fixing Remaining Mistakes</h4>
                    <div className="gradient-explanation">
                      <div className="residual-demo">
                        <h5>How the "small corrections" work:</h5>
                        <div className="residual-steps">
                          <div className="residual-step">
                            <div className="step-label">Actual test scores (out of 100):</div>
                            <div className="values">
                              <div className="student-scores">
                                <div>Student 1: <strong>85</strong></div>
                                <div>Student 2: <strong>72</strong></div>
                                <div>Student 3: <strong>90</strong></div>
                                <div>Student 4: <strong>68</strong></div>
                                <div>Student 5: <strong>78</strong></div>
                              </div>
                            </div>
                            <div className="values-explanation">These are the real final test scores we want to predict</div>
                          </div>
                          <div className="residual-step">
                            <div className="step-label">First model predictions:</div>
                            <div className="values">
                              <div className="student-scores">
                                <div>Student 1: <strong>82</strong></div>
                                <div>Student 2: <strong>75</strong></div>
                                <div>Student 3: <strong>88</strong></div>
                                <div>Student 4: <strong>65</strong></div>
                                <div>Student 5: <strong>80</strong></div>
                              </div>
                            </div>
                            <div className="values-explanation">Our initial simple model tries to guess the scores</div>
                          </div>
                          <div className="residual-step">
                            <div className="step-label">Errors (Actual − Predicted):</div>
                            <div className="values">
                              <div className="student-scores">
                                <div>Student 1: <strong className="error-positive">+3</strong> (underestimated)</div>
                                <div>Student 2: <strong className="error-negative">-3</strong> (overestimated)</div>
                                <div>Student 3: <strong className="error-positive">+2</strong> (underestimated)</div>
                                <div>Student 4: <strong className="error-positive">+3</strong> (underestimated)</div>
                                <div>Student 5: <strong className="error-negative">-2</strong> (overestimated)</div>
                              </div>
                            </div>
                            <div className="values-explanation">Positive errors mean the model predicted too low. Negative errors mean it predicted too high.</div>
                          </div>
                          <div className="residual-step">
                            <div className="step-label">Next small tree learns patterns and adds corrections:</div>
                            <div className="correction-summary">
                              <div style={{ marginTop: '0.5rem' }}>
                                <strong>Corrections to apply:</strong>
                                <ul style={{ marginTop: '0.5rem', paddingLeft: '20px', marginBottom: '0' }}>
                                  <li>Students 1, 3, 4: <strong>Add +2 to +3 points</strong></li>
                                  <li>Student 2: <strong>Subtract 3 points</strong></li>
                                  <li>Student 5: <strong>Subtract 2 points</strong></li>
                                </ul>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="gradient-process">
                        <h5><Icon name="sync" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> The process in simple steps:</h5>
                        <div className="process-flow">
                          <div className="flow-step">
                            <div className="flow-icon"><Icon name="target" size={20} /></div>
                            <div className="flow-text">Start with simple predictions</div>
                          </div>
                          <div className="flow-arrow">→</div>
                          <div className="flow-step">
                            <div className="flow-icon"><Icon name="target" size={20} /></div>
                            <div className="flow-text">Find where predictions are wrong</div>
                          </div>
                          <div className="flow-arrow">→</div>
                          <div className="flow-step">
                            <div className="flow-icon"><Icon name="target" size={20} /></div>
                            <div className="flow-text">Train a small tree to fix those mistakes</div>
                          </div>
                          <div className="flow-arrow">→</div>
                          <div className="flow-step">
                            <div className="flow-icon"><Icon name="target" size={20} /></div>
                            <div className="flow-text">Add the small correction to the model</div>
                          </div>
                          <div className="flow-arrow">→</div>
                          <div className="flow-step">
                            <div className="flow-icon"><Icon name="target" size={20} /></div>
                            <div className="flow-text">Repeat until performance stops improving</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* XGBoost Tab */}
              {activeBoostingTab === 'xgboost' && (
                <div className="tab-panel">
                  <div className="algorithm-deep-dive">
                    <h4><Icon name="bolt" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> XGBoost: Extreme Optimization</h4>
                    <div className="xgboost-features">
                      <div className="feature-grid">
                        <div className="feature-card">
                          <div className="feature-icon"><Icon name="rocket" size={24} /></div>
                          <h6>Speed Optimizations</h6>
                          <p>Parallel processing and memory optimization for faster training</p>
                        </div>
                        <div className="feature-card">
                          <div className="feature-icon"><Icon name="shield" size={24} /></div>
                          <h6>Regularization</h6>
                          <p>Built-in overfitting prevention with L1 and L2 regularization</p>
                        </div>
                        <div className="feature-card">
                          <div className="feature-icon"><Icon name="chart" size={24} /></div>
                          <h6>Advanced Features</h6>
                          <p>Handles missing values and categorical features automatically</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Comparison Tab */}
              {activeBoostingTab === 'comparison' && (
                <div className="tab-panel">
                  <div className="algorithm-comparison">
                    <h4><Icon name="scale" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Boosting Algorithm Comparison</h4>
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
                      <h5><Icon name="target" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Key Insights:</h5>
                      <div className="insights-grid">
                        <div className="insight-card">
                          <strong><Icon name="bolt" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Speed:</strong> XGBoost is fastest, AdaBoost is moderate, Gradient Boosting is slowest
                        </div>
                        <div className="insight-card">
                          <strong><Icon name="target" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Accuracy:</strong> XGBoost usually wins, but Gradient Boosting is very close
                        </div>
                        <div className="insight-card">
                          <strong><Icon name="target" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Interpretability:</strong> AdaBoost is easiest to understand, XGBoost is most complex
                        </div>
                        <div className="insight-card">
                          <strong><Icon name="shield" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Overfitting:</strong> XGBoost has best protection, others need careful tuning
                        </div>
                      </div>
                    </div>

                    <div className="when-to-use">
                      <h5><Icon name="target" size={18} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> When to use which?</h5>
                      <ul>
                        <li><strong>AdaBoost:</strong> Smaller datasets; want simple, more explainable behavior.</li>
                        <li><strong>Gradient Boosting:</strong> Aim for top accuracy with careful tuning on medium data.</li>
                        <li><strong>XGBoost:</strong> Larger datasets; need speed and built‑in safeguards against overfitting.</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {currentStep.id === 'experiment' && (
          <div className="experiment-section">
            <h3><Icon name="experiment" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Performance Comparison on Stroke Dataset</h3>
            <p>We train multiple algorithms on the same stroke dataset and compare accuracy, precision, recall, and F1‑score.</p>
            
            <div className="experiment-controls">
              <button 
                className="run-experiment-btn"
                onClick={runExperiment}
                disabled={isRunningExperiment}
              >
                {isRunningExperiment ? <><Icon name="sync" size={16} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Running Experiment...</> : <><Icon name="rocket" size={16} style={{ marginRight: '6px', verticalAlign: 'middle' }} /> Run Performance Comparison</>}
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
                <h4><Icon name="chart" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Performance Results</h4>
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
                  <h4><Icon name="target" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Analysis</h4>
                  <div className="analysis-points">
                    <div className="analysis-point">
                      <strong><Icon name="target" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Best Performer:</strong> XGBoost achieved the highest accuracy (90.0%)
                    </div>
                    <div className="analysis-point">
                      <strong><Icon name="chart" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Improvement:</strong> Boosting algorithms significantly outperform single trees
                    </div>
                    <div className="analysis-point">
                      <strong><Icon name="bolt" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Speed vs Accuracy:</strong> XGBoost provides the best balance of performance and efficiency
                    </div>
                    <div className="analysis-point">
                      <strong><Icon name="target" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Key Insight:</strong> Ensemble methods work because they combine different perspectives
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
              onClick={() => {
                setCurrentSection(index);
                // Scroll to top when changing sections via indicator
                window.scrollTo({ top: 0, behavior: 'smooth' });
                if (contentRef.current) {
                  contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
              }}
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
