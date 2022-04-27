from pathlib import Path

src: Path = Path(__file__).absolute().parent.parent
project: Path = src.parent
data: Path = project / 'data'
scripts: Path = project / 'scripts'
tests: Path = project / 'tests'
datasets: Path = data / 'datasets'
test_results: Path = data / 'test_results'
tsne_plots: Path = test_results / 'tsne_plots'
logs: Path = data / 'logs'

for i in list(vars().values()):
    if isinstance(i, Path):
        i.mkdir(parents=True, exist_ok=True)

