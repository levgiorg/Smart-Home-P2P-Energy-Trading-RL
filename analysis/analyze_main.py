from enhanced_analyzer import EnhancedRunAnalyzer
import matplotlib.pyplot as plt

def main():
    analyzer = EnhancedRunAnalyzer("ml-outputs")
    
    # Plot detailed comparison of top runs
    analyzer.plot_top_runs_detailed()
    
    # Print detailed analysis
    analyzer.print_detailed_analysis()
    
    plt.show()

if __name__ == "__main__":
    main()