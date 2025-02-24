import matplotlib.pyplot as plt

from enhanced_analyzer import EnhancedRunAnalyzer
from cartel_analyzer import AntiCartelAnalyzer


def Enhanced_main():
    analyzer = EnhancedRunAnalyzer("ml-outputs")
    
    # Plot detailed comparison of top runs
    analyzer.plot_top_runs_detailed()
    
    # Print detailed analysis
    analyzer.print_detailed_analysis()
    
    plt.show()

def Cartel_main():
    # Initialize analyzer with your output directory
    analyzer = AntiCartelAnalyzer("ml-outputs")
    
    # Generate comparison plots
    fig = analyzer.compare_mechanisms()
    
    # Print statistical analysis
    analyzer.print_statistical_analysis()
    
    # Save plot to file in this directory 
    fig.savefig("cartel_comparison.png")

    # Show plots
    plt.show()

if __name__ == "__main__":
    Cartel_main()