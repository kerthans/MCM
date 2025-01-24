from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console()

def log_info(message):
    console.print(f"[bold blue]INFO:[/bold blue] {message}")

def log_success(message):
    console.print(f"[bold green]SUCCESS:[/bold green] {message}")

def log_error(message):
    console.print(f"[bold red]ERROR:[/bold red] {message}")

def create_progress():
    return Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    )