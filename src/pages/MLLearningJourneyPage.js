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
  const [activeBoostingTab, setActiveBoostingTab] = useState('overview'); // 'overview', 'adaboost', 'gradient', 'xgboost'
  const [adaboostStep, setAdaboostStep] = useState(0); // For AdaBoost step navigation
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const sectionRefs = useRef([]);
  const contentRef = useRef(null);
  const searchInputRef = useRef(null);

  // Alex's Learning Journey - Story-Driven Sections
  const learningSections = [
    {
      id: 'ml-intro',
      title: 'Chapter 1: Alex Discovers Machine Learning',
      titleIcon: 'ml',
      chapterNumber: 1,
      storyOpening: 'Meet Alex, a curious student who just got his first dataset about student performance. "How can I predict who will succeed?" Alex wonders. This is where Alex\'s Machine Learning journey begins...',
      content: 'Alex learns that Machine Learning helps discover patterns in data—just like a teacher who, after seeing thousands of student cases, can predict performance. Alex realizes this could solve his problem of predicting student success!',
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
      title: 'Chapter 2: Alex Builds His First Decision Tree',
      titleIcon: 'tree',
      chapterNumber: 2,
      storyOpening: 'Alex faces a challenge: "I need to make predictions, but where do I start?" Alex discovers Decision Trees—the perfect first step. Like deciding whether to bring an umbrella, Alex learns to ask simple questions that lead to smart decisions.',
      content: 'Alex builds his first Decision Tree, learning to ask simple yes/no questions one at a time to reach predictions. Just like deciding whether to bring an umbrella based on weather conditions! Alex realizes these trees are the foundation for more powerful techniques...',
      analogy: 'Like deciding whether to go out: "Is it raining?" → If yes, check "Do we have umbrella?" → Final decision: Go out or stay home',
      concepts: [
        {
          term: 'Entropy',
          description: 'How mixed or uncertain the groups are',
          example: 'Like checking the weather where half the days are rainy and half are sunny = very mixed (high entropy). But if almost every day is rainy = not mixed at all (low entropy)',
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
          example: 'Asking "Is it raining?" helps decide about bringing an umbrella really well - that\'s big information gain! A bad question like "Is it Tuesday?" wouldn\'t help predict the weather at all',
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
          example: 'For weather decisions, we split by asking "Is it raining?" → creates two groups: rainy days and not-rainy days. Then we can ask more questions like "Is it windy?" to split those groups further',
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
      title: 'Chapter 3: Alex Discovers the Power of Teams',
      titleIcon: 'team',
      chapterNumber: 3,
      storyOpening: 'Alex realizes his single Decision Tree makes mistakes. "What if I could combine multiple trees?" Alex wonders. This leads to a breakthrough: Ensemble Learning—where many trees work together, just like a team of teachers giving opinions!',
      content: 'Alex learns that combining many trees and letting them vote creates better predictions than any single tree alone. This reduces individual biases and dramatically improves accuracy—like multiple teachers collaborating to determine the best grade!',
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
      title: 'Chapter 4: Alex Learns from Mistakes',
      titleIcon: 'rocket',
      chapterNumber: 4,
      storyOpening: 'Alex notices something powerful: "What if each new tree focused on fixing the mistakes of the previous ones?" This is Boosting—where models learn from their errors, just like reviewing test questions you got wrong and studying those patterns more carefully.',
      content: 'Alex discovers Boosting, where trees are built one after another. Each new tree focuses more on examples that were previously misclassified, steadily reducing errors. Alex realizes this adaptive approach is the key to building truly powerful models!',
      analogy: 'Like reviewing test questions you got wrong last time and studying those patterns more carefully',
      concepts: [
        {
          algorithm: 'AdaBoost',
          description: 'Builds an ensemble of weak learners (simple decision trees) that adaptively focus on previously misclassified examples',
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
      title: 'Chapter 5: Alex Puts It All Together',
      titleIcon: 'experiment',
      chapterNumber: 5,
      storyOpening: 'Alex has learned so much! Now it\'s time to put everything together. "Which algorithm works best for my problem?" Alex decides to run experiments, comparing different methods—just like testing different study methods to find what works best.',
      content: 'Alex conducts his first comprehensive experiment, training multiple models and comparing their performance. This is where theory meets practice, and Alex discovers which techniques work best for real-world predictions!',
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

    // Tree data for weather/umbrella decision tree
    const treeData = {
      name: "Is it raining?",
      children: [
        {
          name: "Do we have umbrella?",
          branchLabel: "Yes",
          children: [
            { name: "Go out", icon: "🚶" },
            { name: "Don't go", icon: "🏠" }
          ]
        },
        {
          name: "Go out",
          branchLabel: "No",
          icon: "🚶"
        }
      ]
    };

    const width = 2000;
    const height = 1200;
    
    const svgElement = d3.select("#decision-tree-svg")
      .append("svg")
      .attr("viewBox", [0, 0, width, height])
      .attr("preserveAspectRatio", "xMidYMid meet");

    // Add gradient definitions with enhanced colors
    const defs = svgElement.append("defs");
    
    // Main node gradient - more vibrant purple-blue gradient
    const gradient = defs.append("linearGradient")
      .attr("id", "nodeGradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "100%");
    
    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#7c3aed");
    
    gradient.append("stop")
      .attr("offset", "50%")
      .attr("stop-color", "#667eea");
    
    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#8b5cf6");
    
    // Glow filter for nodes
    const glowFilter = defs.append("filter")
      .attr("id", "glow")
      .attr("x", "-50%")
      .attr("y", "-50%")
      .attr("width", "200%")
      .attr("height", "200%");
    
    glowFilter.append("feGaussianBlur")
      .attr("stdDeviation", "4")
      .attr("result", "coloredBlur");
    
    const feMerge = glowFilter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");
    
    // Link gradient
    const linkGradient = defs.append("linearGradient")
      .attr("id", "linkGradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");
    
    linkGradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", "#667eea")
      .attr("stop-opacity", "0.9");
    
    linkGradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", "#8b5cf6")
      .attr("stop-opacity", "0.9");

    const svg = svgElement
      .append("g")
      .attr("transform", "translate(250,100)");

    // Vertical tree layout - swap width and height for vertical orientation
    // Significantly increased spacing between branches and nodes
    const treeLayout = d3.tree()
      .size([width - 500, height - 200])
      .separation((a, b) => {
        // Add much more separation between nodes
        // Siblings get significantly more space, parents and children get more vertical space
        if (a.parent === b.parent) {
          return 3.0; // Much more space between siblings (branches)
        }
        return 2.5; // Much more space between parent and child
      });
    
    const root = d3.hierarchy(treeData);
    treeLayout(root);

    // Links - for vertical tree, with enhanced styling
    svg.selectAll(".link")
      .data(root.links())
      .join("path")
      .attr("class", "link")
      .attr("d", d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y))
      .attr("stroke-width", 4)
      .attr("stroke", "url(#linkGradient)")
      .attr("opacity", 0.8)
      .attr("fill", "none")
      .style("filter", "drop-shadow(0 2px 4px rgba(102, 126, 234, 0.4))")
      .on("mouseenter", function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("stroke-width", 5)
          .attr("opacity", 1)
          .style("filter", "drop-shadow(0 3px 6px rgba(102, 126, 234, 0.6))");
      })
      .on("mouseleave", function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr("stroke-width", 4)
          .attr("opacity", 0.8)
          .style("filter", "drop-shadow(0 2px 4px rgba(102, 126, 234, 0.4))");
      });

    // Calculate node dimensions - larger boxes with better spacing
    const getNodeWidth = (text) => {
      const baseWidth = 300;
      const minWidth = 350;
      const charWidth = 15;
      return Math.max(minWidth, Math.min(baseWidth + text.length * charWidth, 500));
    };

    const getNodeHeight = (hasIcon) => hasIcon ? 130 : 110;

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
      
      // Create rectangle with enhanced styling
      const rect = nodeGroup.append("rect")
        .attr("x", -nodeWidth / 2)
        .attr("y", -nodeHeight / 2)
        .attr("width", nodeWidth)
        .attr("height", nodeHeight)
        .attr("fill", "url(#nodeGradient)")
        .attr("stroke", "rgba(255, 255, 255, 0.4)")
        .attr("stroke-width", 2.5)
        .attr("rx", 16)
        .attr("ry", 16)
        .style("cursor", "pointer")
        .style("filter", "drop-shadow(0 4px 12px rgba(102, 126, 234, 0.5))")
        .on("mouseover", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill", "#8b5cf6")
            .attr("stroke", "rgba(255, 255, 255, 0.7)")
            .attr("stroke-width", 3.5)
            .style("filter", "drop-shadow(0 6px 20px rgba(139, 92, 246, 0.7))")
            .attr("rx", 18)
            .attr("ry", 18);
        })
        .on("mouseout", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill", "url(#nodeGradient)")
            .attr("stroke", "rgba(255, 255, 255, 0.4)")
            .attr("stroke-width", 2.5)
            .style("filter", "drop-shadow(0 4px 12px rgba(102, 126, 234, 0.5))")
            .attr("rx", 16)
            .attr("ry", 16);
        });

      // Add icon if available
      if (d.data.icon) {
        nodeGroup.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", -20)
          .attr("font-size", "120px")
          .style("font-size", "120px")
          .text(d.data.icon);
      }

      // Add text - much larger font size with better styling and shadow
      nodeGroup.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", d.data.icon ? 35 : 18)
        .attr("fill", "#ffffff")
        .attr("font-size", "72px")
        .attr("font-weight", "700")
        .attr("font-family", "Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif")
        .style("font-size", "72px")
        .style("font-weight", "700")
        .style("pointer-events", "none")
        .style("text-shadow", "0 2px 4px rgba(0, 0, 0, 0.5), 0 1px 2px rgba(0, 0, 0, 0.7)")
        .style("letter-spacing", "0.8px")
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
          if (d.target.parent.data.name === "Is it raining?") {
            labelText = index === 0 ? "Yes" : "No";
          } else if (d.target.parent.data.name === "Do we have umbrella?") {
            labelText = index === 0 ? "Yes" : "No";
          }
        }
        
        if (!labelText) return;
        
        // Create background circle/ellipse for label with enhanced styling
        labelGroup.append("ellipse")
          .attr("cx", midX + 60)
          .attr("cy", midY)
          .attr("rx", 50)
          .attr("ry", 35)
          .attr("fill", "#667eea")
          .attr("stroke", "rgba(255, 255, 255, 0.6)")
          .attr("stroke-width", 3.5)
          .attr("opacity", 0.95)
          .style("filter", "drop-shadow(0 3px 8px rgba(102, 126, 234, 0.6))")
          .on("mouseenter", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("rx", 52)
              .attr("ry", 37)
              .attr("opacity", 1)
              .style("filter", "drop-shadow(0 4px 10px rgba(102, 126, 234, 0.8))");
          })
          .on("mouseleave", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("rx", 50)
              .attr("ry", 35)
              .attr("opacity", 0.95)
              .style("filter", "drop-shadow(0 3px 8px rgba(102, 126, 234, 0.6))");
          });
        
        // Add label text - larger font size with shadow
        labelGroup.append("text")
          .attr("x", midX + 60)
          .attr("y", midY)
          .attr("text-anchor", "middle")
          .attr("dy", "0.35em")
          .attr("fill", "#ffffff")
          .attr("font-size", "36px")
          .attr("font-weight", "800")
          .attr("font-family", "Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif")
          .style("text-shadow", "0 2px 4px rgba(0, 0, 0, 0.6), 0 1px 2px rgba(0, 0, 0, 0.8)")
          .style("letter-spacing", "0.8px")
          .text(labelText);
      });

    // Cleanup function
    return () => {
      d3.select("#decision-tree-svg").selectAll("*").remove();
    };
  }, [currentSection]);

  // Build searchable index from all sections and concepts
  const buildSearchIndex = () => {
    const index = [];
    learningSections.forEach((section, sectionIndex) => {
      // Add section title
      index.push({
        type: 'section',
        title: section.title,
        sectionIndex,
        id: section.id,
        matchText: section.title.toLowerCase()
      });
      
      // Add section content and analogy
      if (section.content) {
        index.push({
          type: 'section',
          title: section.title,
          sectionIndex,
          id: section.id,
          matchText: section.content.toLowerCase()
        });
      }
      if (section.analogy) {
        index.push({
          type: 'section',
          title: section.title,
          sectionIndex,
          id: section.id,
          matchText: section.analogy.toLowerCase()
        });
      }
      
      // Add concepts
      if (section.concepts) {
        section.concepts.forEach((concept, conceptIndex) => {
          const conceptName = concept.term || concept.type || concept.algorithm || '';
          const conceptDesc = concept.description || '';
          
          // For boosting algorithms, add algorithm-specific entries with tab info
          if (section.id === 'boosting-algorithms' && concept.algorithm) {
            const algoName = concept.algorithm.toLowerCase();
            // Map algorithm names to tab names
            let tabName = null;
            if (algoName.includes('adaboost')) {
              tabName = 'adaboost';
            } else if (algoName.includes('gradient')) {
              tabName = 'gradient';
            } else if (algoName.includes('xgboost')) {
              tabName = 'xgboost';
            }
            
            // Add multiple variations for better matching
            const variations = [
              conceptName.toLowerCase(),
              algoName,
              algoName.replace(/\s+/g, ''), // "gradient boosting" -> "gradientboosting"
              algoName.replace(/\s+/g, '-'), // "gradient boosting" -> "gradient-boosting"
            ];
            
            variations.forEach(variation => {
              index.push({
                type: 'concept',
                title: conceptName,
                sectionIndex,
                conceptIndex,
                id: section.id,
                description: conceptDesc,
                matchText: `${variation} ${conceptDesc}`.toLowerCase(),
                tabName: tabName // Store tab name for navigation
              });
            });
          } else {
            // Regular concept
            index.push({
              type: 'concept',
              title: conceptName,
              sectionIndex,
              conceptIndex,
              id: section.id,
              description: conceptDesc,
              matchText: `${conceptName} ${conceptDesc}`.toLowerCase()
            });
          }
        });
      }
    });
    return index;
  };

  // Search functionality
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setSearchResults([]);
      setShowSearchResults(false);
      return;
    }

    const index = buildSearchIndex();
    const query = searchQuery.toLowerCase().trim();
    
    // Normalize query (remove spaces, hyphens for better matching)
    const normalizedQuery = query.replace(/\s+/g, '').replace(/-/g, '');
    
    const results = index.filter(item => {
      const matchText = item.matchText;
      const normalizedMatch = matchText.replace(/\s+/g, '').replace(/-/g, '');
      const titleMatch = item.title.toLowerCase();
      const normalizedTitle = titleMatch.replace(/\s+/g, '').replace(/-/g, '');
      
      return matchText.includes(query) || 
             normalizedMatch.includes(normalizedQuery) ||
             titleMatch.includes(query) ||
             normalizedTitle.includes(normalizedQuery);
    })
    // Remove duplicates based on sectionIndex, conceptIndex, and tabName
    .filter((item, index, self) => 
      index === self.findIndex(t => 
        t.sectionIndex === item.sectionIndex && 
        t.conceptIndex === item.conceptIndex &&
        t.tabName === item.tabName
      )
    )
    // Prioritize exact matches and algorithm-specific results
    .sort((a, b) => {
      const aExact = a.title.toLowerCase() === query || a.title.toLowerCase().includes(query);
      const bExact = b.title.toLowerCase() === query || b.title.toLowerCase().includes(query);
      if (aExact && !bExact) return -1;
      if (!aExact && bExact) return 1;
      if (a.tabName && !b.tabName) return -1;
      if (!a.tabName && b.tabName) return 1;
      return 0;
    })
    .slice(0, 10); // Limit to 10 results

    setSearchResults(results);
    setShowSearchResults(results.length > 0);
  }, [searchQuery]);

  // Navigate to search result
  const handleSearchSelect = (result) => {
    setCurrentSection(result.sectionIndex);
    
    // If it's a boosting algorithm concept, set the appropriate tab
    if (result.id === 'boosting-algorithms' && result.tabName) {
      setActiveBoostingTab(result.tabName);
      if (result.tabName === 'adaboost') {
        setAdaboostStep(0); // Reset AdaBoost step when navigating
      }
    }
    
    setSearchQuery('');
    setShowSearchResults(false);
    
    // Small delay to ensure tab is set before scrolling
    setTimeout(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      if (contentRef.current) {
        contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
  };

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl/Cmd + K to focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
      // Escape to close search
      if (e.key === 'Escape') {
        setShowSearchResults(false);
        searchInputRef.current?.blur();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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
      {/* Search Navbar */}
      <div className="ml-search-navbar">
        <div className="search-container">
          <div className="search-input-wrapper">
            <Icon name="search" size={20} className="search-icon" />
            <input
              ref={searchInputRef}
              type="text"
              className="search-input"
              placeholder="Search concepts... (Ctrl+K)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => searchQuery && setShowSearchResults(true)}
            />
            {searchQuery && (
              <button 
                className="search-clear"
                onClick={() => {
                  setSearchQuery('');
                  setShowSearchResults(false);
                  searchInputRef.current?.focus();
                }}
              >
                <Icon name="close" size={16} />
              </button>
            )}
          </div>
          {showSearchResults && searchResults.length > 0 && (
            <div className="search-results-dropdown">
              {searchResults.map((result, idx) => (
                <div
                  key={idx}
                  className="search-result-item"
                  onClick={() => handleSearchSelect(result)}
                >
                  <div className="search-result-header">
                    <Icon 
                      name={result.type === 'section' ? 'target' : 'brain'} 
                      size={16} 
                      className="search-result-icon"
                    />
                    <span className="search-result-title">{result.title}</span>
                    <span className="search-result-type">{result.type}</span>
                  </div>
                  {result.description && (
                    <div className="search-result-description">{result.description}</div>
                  )}
                  <div className="search-result-section">
                    {learningSections[result.sectionIndex]?.title}
                  </div>
                </div>
              ))}
            </div>
          )}
          {showSearchResults && searchQuery && searchResults.length === 0 && (
            <div className="search-results-dropdown">
              <div className="search-result-empty">
                <Icon name="search" size={24} />
                <p>No results found for "{searchQuery}"</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Story Hero Section - Only show on first section */}
      {currentSection === 0 && (
        <div className="hero-section animated-intro story-hero">
          <div className="hero-content">
            <div className="story-character-intro">
              <div className="character-avatar">👤</div>
              <h2 className="character-name">Meet Alex</h2>
            </div>
            <h1 className="hero-title animated-title">
              <span className="hero-title-main slide-in-right">Alex's Machine Learning</span>
              <span className="hero-title-sub slide-in-right-delay">Journey</span>
            </h1>
            <p className="hero-description fade-in story-intro-text">
              Follow Alex, a curious student, as he discovers Machine Learning step by step. Each chapter tells the story of how Alex learns to predict student performance—from first concepts to advanced techniques. Let's join Alex on this journey!
            </p>
            <div className="story-preview fade-in-delay">
              <p className="story-hook">
                <strong>Alex's Problem:</strong> "I have data about students—their test scores, homework grades, and attendance. Can I predict who will succeed? Where do I even start?"
              </p>
              <p className="story-promise">
                Join Alex as he discovers Decision Trees, learns about Ensemble Learning, masters Boosting algorithms, and finally conducts experiments to solve this challenge!
              </p>
            </div>
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
                <text x="50" y="270" fill="rgba(255,255,255,0.7)" fontSize="20" textAnchor="middle" className="graph-label">
                  <animate attributeName="opacity" values="0;1" dur="0.5s" begin="0.5s" fill="freeze"/>
                  Trees
                </text>
                <text x="350" y="120" fill="rgba(255,255,255,0.7)" fontSize="20" textAnchor="middle" className="graph-label">
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

      {/* Story Progress Bar */}
      <div className={`journey-progress ${isVisible ? 'animate-in' : ''}`}>
        <div className="progress-header">
          <h3>Alex's Journey Progress</h3>
          <div className="progress-stats">
            <span className="current-step">Chapter {currentStep.chapterNumber || currentSection + 1}</span>
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
          
          {/* Story Opening */}
          {currentStep.storyOpening && (
            <div className={`story-opening ${animationStep ? 'animate-in' : ''}`}>
              <div className="story-narrative">
                <div className="story-quote-icon">💭</div>
                <p className="story-text">{currentStep.storyOpening}</p>
              </div>
            </div>
          )}
          
          <p className={`section-description ${animationStep ? 'animate-in' : ''}`}>
            {currentStep.content}
          </p>
          {currentStep.analogy && (
            <div className={`analogy-box ${animationStep ? 'animate-in' : ''}`}>
              <div className="analogy-icon"><Icon name="lightbulb" size={24} /></div>
              <div className="analogy-text">
                <strong>Alex thinks:</strong> {currentStep.analogy}
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
            <h3><Icon name="tree" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Alex Builds His First Decision Tree</h3>
            <div className="story-moment">
              <p className="story-context">
                "I need a simple way to make predictions," Alex thinks. "What if I could build something like a flowchart that asks yes/no questions? That would be perfect for starting my journey!"
              </p>
            </div>
            <p className="section-intro" style={{ fontSize: '1.1rem', color: 'rgba(255, 255, 255, 0.85)', marginBottom: '2rem', lineHeight: '1.6', textAlign: 'center' }}>
              Alex learns that a decision tree is like a flowchart that helps make decisions by asking questions. To understand this, Alex starts with a simple example: deciding whether to bring an umbrella based on the weather!
            </p>
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
              <h4><Icon name="cloud" size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Real-World Example: Weather Decision Tree</h4>
              <p className="example-intro">
                Every morning, you face a simple decision: should you bring an umbrella? This decision tree shows how asking just a few yes/no questions about the weather helps you make the right choice.
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
                      <strong>First Question:</strong> "Is it raining?" — this splits the decision into two paths: rainy or not rainy.
                    </div>
                  </div>
                  <div className="explanation-step">
                    <div className="step-number">2</div>
                    <div className="step-content">
                      <strong>If Not Raining:</strong> Go out! No need to worry about an umbrella.
                    </div>
                  </div>
                  <div className="explanation-step">
                    <div className="step-number">3</div>
                    <div className="step-content">
                      <strong>If Raining:</strong> Ask "Do we have umbrella?" — If yes, go out with it. If no, don't go. This decision tree helps us make a smart choice every day!
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
              <h3><Icon name="team" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Alex's Breakthrough: The Power of Teams</h3>
              <div className="story-moment">
                <p className="story-context">
                  "My single tree makes mistakes," Alex realizes. "But what if I could combine multiple trees? Like asking several teachers for their opinion instead of just one—the consensus would be much more reliable!"
                </p>
              </div>
              <p className="ensemble-intro-text">
                Alex discovers that ensemble learning combines multiple decision trees to create a more accurate and reliable prediction system. 
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
            <h3><Icon name="rocket" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Alex's Discovery: Boosting Algorithms</h3>
            <div className="story-moment">
              <p className="story-context">
                "I noticed my ensemble works well," Alex thinks, "but what if I could make each new model focus on fixing the mistakes of the previous ones? That would be like learning from my errors—getting better with each iteration!"
              </p>
            </div>
            
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
                      <p><strong>AdaBoost (Adaptive Boosting)</strong> combines multiple <strong>weak learners</strong> (simple decision trees) into one powerful model. Each tree is a simple decision tree with just <strong>one decision</strong> (e.g., "Score > 70?" → Pass or Fail).</p>
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
                          <p><strong>Every student starts equal:</strong> All 6 students have weight = 1/6. The first decision tree trains on this balanced dataset. At this point, we haven't made any predictions yet.</p>
                        </div>
                      )}

                      {/* Step 2 */}
                      {adaboostStep === 1 && (
                        <div className="visual-step">
                          <h6>Step 2: First Tree Makes Predictions</h6>
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
                            <h6>First Tree Predictions:</h6>
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
                          <p><strong>Context:</strong> We add more challenging students that don't fit the simple "Score > 70" rule. The first tree now makes mistakes.</p>
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
                            <h6>Same Tree with More Complex Students:</h6>
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
                                <span className="mistake-explanation">Tree 1 only saw Score (58) and said Fail, but actual result is Pass because Homework (100) compensates!</span>
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
                                <span className="mistake-explanation">Tree 1 only saw Score (70) and said Pass, but actual result is Fail because low Homework (60) requires higher test scores!</span>
                              </div>
                              <div className="prediction-item">
                                <span className="patient-info">Student 6 (Score: 48, Homework: 70)</span>
                                <span className="prediction-result correct"><Icon name="check" size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Predicted: Fail</span>
                                <span className="actual-label">Actual: Fail</span>
                              </div>
                            </div>
                            <p className="iteration-summary"><strong>Key insight:</strong> The first tree (using only "Score > 70?") got <strong>2 students wrong</strong> (Students 3 and 5). These mistakes show that the first tree is too simple - it doesn't see homework grades! This is exactly what AdaBoost learns: we need more information to fix these mistakes.</p>
                          </div>
                        </div>
                      )}

                      {/* Step 4 */}
                      {adaboostStep === 3 && (
                        <div className="visual-step">
                          <h6>Step 4: Increase Weights for Mistakes</h6>
                          <div className="weight-update-explanation">
                            <p><strong>AdaBoost's adaptive mechanism:</strong> Students that were misclassified get their weights increased. This forces the next tree to pay more attention to them!</p>
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
                          <p><strong>Why this works:</strong> When the next tree trains, Students 3 and 5 "count more" because they have higher weights. The algorithm is forced to focus on correcting these mistakes!</p>
                        </div>
                      )}

                      {/* Step 5 */}
                      {adaboostStep === 4 && (
                        <div className="visual-step">
                          <h6>Step 5: Second Tree Corrects the Mistakes</h6>
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
                            <h6>Second Tree Predictions (Focusing on High-Weight Students):</h6>
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
                            <p className="iteration-summary"><strong>Success!</strong> The second tree corrected the mistakes from the first tree! Student 3 (high homework) is now correctly identified as Pass, and Student 5 (low homework) is correctly identified as Fail. The weighted training forced this tree to focus on what the first tree got wrong.</p>
                          </div>
                        </div>
                      )}

                      {/* Step 6 */}
                      {adaboostStep === 5 && (
                        <div className="visual-step">
                          <h6>Step 6: Combine All Trees with Weighted Voting</h6>
                          <div className="ensemble-visual">
                            <div className="ensemble-stumps">
                              <div className="stump-card">
                                <div className="stump-title">Tree 1</div>
                                <div className="stump-rule">"Score > 70?"</div>
                                <div className="stump-weight">Weight: 0.85</div>
                                <div className="stump-accuracy">Accuracy: 67%</div>
                              </div>
                              <div className="stump-card">
                                <div className="stump-title">Tree 2</div>
                                <div className="stump-rule">"Homework > 80?"</div>
                                <div className="stump-weight">Weight: 0.60</div>
                                <div className="stump-accuracy">Accuracy: 83%</div>
                              </div>
                              <div className="stump-card">
                                <div className="stump-title">Tree 3</div>
                                <div className="stump-rule">"Attendance > 90%?"</div>
                                <div className="stump-weight">Weight: 0.70</div>
                                <div className="stump-accuracy">Accuracy: 92%</div>
                              </div>
                            </div>
                            <div className="ensemble-plus">+</div>
                            <div className="final-vote">
                              <div className="vote-title">Final Prediction</div>
                              <div className="vote-explanation">Each tree votes, weighted by its accuracy. More accurate trees have more influence!</div>
                              <div className="vote-formula">
                                Final = (0.85 × Tree1) + (0.60 × Tree2) + (0.70 × Tree3)
                              </div>
                            </div>
                          </div>
                          <p><strong>Final prediction:</strong> Each tree makes a prediction, but their votes are weighted by accuracy. More accurate trees have more say. The final prediction is the weighted majority vote - making the ensemble more reliable than any single tree!</p>
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
                              <strong>Simple Decision Trees:</strong> Very simple trees with just one split (one question). Much weaker than full trees, but AdaBoost combines many to make them strong.
                            </div>
                            <div className="concept-item">
                              <strong>Adaptive Weights:</strong> Mistakes from one round become the focus of the next round. The algorithm adapts to fix errors iteratively.
                            </div>
                            <div className="concept-item">
                              <strong>Weighted Voting:</strong> Not all trees are equal. More accurate trees get higher voting weights in the final prediction.
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
                                <strong>Tree 1:</strong> "Score > 70?" → Yes → Votes <strong>Pass</strong> (weight: 0.85)
                              </div>
                              <div className="example-stump">
                                <strong>Tree 2:</strong> "Homework > 80?" → Yes → Votes <strong>Pass</strong> (weight: 0.90)
                              </div>
                              <div className="example-stump">
                                <strong>Tree 3:</strong> "Attendance > 90%?" → No → Votes <strong>Fail</strong> (weight: 0.60)
                              </div>
                            </div>
                            <div className="example-result">
                              <strong>Weighted Vote:</strong> (0.85 × Pass) + (0.90 × Pass) + (0.60 × Fail) = <strong>Pass</strong> wins!
                            </div>
                            <p><strong>Why it works:</strong> Even though Tree 3 says Fail, the first two trees (which are more accurate and weighted higher) both say Pass. The ensemble makes a more reliable prediction than any single tree alone.</p>
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

            </div>
          </div>
        )}

        {currentStep.id === 'experiment' && (
          <div className="experiment-section">
            <div className="story-moment">
              <p className="story-context">
                "I've learned so much!" Alex realizes. "Decision Trees, Ensembles, Boosting... but which one works best for my problem? It's time to run experiments and find out!"
              </p>
            </div>
            <h3><Icon name="experiment" size={24} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Alex's Experiment: Performance Comparison</h3>
            <p>Alex trains multiple algorithms on the same stroke dataset to compare accuracy, precision, recall, and F1‑score. "This will show me which technique works best!" Alex thinks.</p>
            
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
