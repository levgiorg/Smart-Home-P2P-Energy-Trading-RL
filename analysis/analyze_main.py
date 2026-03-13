import matplotlib.pyplot as plt
from cartel_analyzer import AntiCartelAnalyzer
from enhanced_analyzer import EnhancedRunAnalyzer


def enhanced_main() -> None:
    analyzer = EnhancedRunAnalyzer("ml-outputs2")

    # Plot detailed comparison of top runs
    analyzer.plot_top_runs_detailed()

    # Print detailed analysis
    analyzer.print_detailed_analysis()

    # Save plot to file in this directory
    plt.savefig("enhanced_comparison.png")

    plt.show()


def cartel_main() -> None:
    # Initialize analyzer with your output directory
    analyzer = AntiCartelAnalyzer("ml-outputs2")

    # Generate comparison plots
    fig = analyzer.compare_mechanisms()

    # Print statistical analysis
    analyzer.print_statistical_analysis()

    # Save plot to file in this directory
    fig.savefig("cartel_comparison.png")

    # Show plots
    plt.show()


if __name__ == "__main__":
    cartel_main()
