#!/usr/bin/env python3
"""
Test script for Morning Stock Analysis
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from morning_stock_analysis import MorningStockAnalyzer

def test_morning_analysis():
    """Test the morning stock analysis functionality"""
    print("Testing Morning Stock Analysis...")
    
    try:
        # Initialize analyzer
        analyzer = MorningStockAnalyzer()
        
        # Run a test analysis
        print("Running test analysis...")
        analysis_result = analyzer.run_morning_analysis()
        
        # Format and print results
        report = analyzer.format_analysis_report(analysis_result)
        print("\n" + "="*80)
        print("TEST RESULTS:")
        print("="*80)
        print(report)
        
        # Save results
        analyzer.save_analysis_to_file(analysis_result, "test_morning_analysis.txt")
        print("\nTest completed successfully! Results saved to test_morning_analysis.txt")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_morning_analysis() 