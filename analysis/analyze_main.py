import matplotlib.pyplot as plt

from enhanced_analyzer import EnhancedRunAnalyzer


def main():
    analyzer = EnhancedRunAnalyzer("ml-outputs/7000 episodes : 10 houses")
    
    # Plot detailed comparison of top runs
    analyzer.plot_top_runs_detailed()
    
    # Print detailed analysis
    analyzer.print_detailed_analysis()
    
    plt.show()

if __name__ == "__main__":
    main()