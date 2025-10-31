import React from 'react';

/**
 * Flaticon Animated Icon Component
 * 
 * Usage:
 * 1. Download animated icons from https://www.flaticon.com/animated-icons-most-downloaded
 * 2. Place SVG/Lottie files in public/icons/flaticon/ directory
 * 3. Use: <FlaticonIcon name="rocket" type="svg" />
 * 
 * For Lottie animations, use type="lottie" and ensure lottie-react is installed
 */
const FlaticonIcon = ({ 
  name, 
  type = 'svg', // 'svg' or 'lottie'
  size = 24,
  className = '',
  style = {},
  ...props 
}) => {
  if (type === 'lottie') {
    // For Lottie animations (requires lottie-react package)
    try {
      const Lottie = require('lottie-react');
      const animationData = require(`../../public/icons/flaticon/${name}.json`);
      
      return (
        <div 
          className={`flaticon-icon lottie-icon ${className}`}
          style={{ width: size, height: size, ...style }}
          {...props}
        >
          <Lottie animationData={animationData} loop={true} autoplay={true} />
        </div>
      );
    } catch (error) {
      console.warn(`Lottie animation not found: ${name}.json`);
      return null;
    }
  } else {
    // For SVG files
    return (
      <img
        src={`/icons/flaticon/${name}.svg`}
        alt={name}
        className={`flaticon-icon svg-icon ${className}`}
        style={{ width: size, height: size, ...style }}
        {...props}
      />
    );
  }
};

export default FlaticonIcon;

