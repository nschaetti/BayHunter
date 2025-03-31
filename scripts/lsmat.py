import scipy.io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

console = Console()


def load_mat_file(mat_file):
    """Charge le fichier .mat et retourne son contenu sous forme de dictionnaire."""
    try:
        data = scipy.io.loadmat(mat_file)
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception as e:
        console.print(f"[bold red]❌ Erreur lors du chargement de {mat_file} : {e}[/bold red]")
        return None


def show_variable(mat_data, var_name):
    """Affiche le contenu d'une variable spécifique."""
    if var_name not in mat_data:
        console.print(f"[bold red]❌ Erreur : La variable '{var_name}' n'existe pas.[/bold red]")
        return

    value = mat_data[var_name]

    console.print(Panel.fit(f"[bold cyan]📜 Contenu de la variable : {var_name}[/bold cyan]\n", title="🔍 Détails"))

    if isinstance(value, np.ndarray):
        console.print(f"[bold green]Type:[/bold green] np.ndarray")
        console.print(f"[bold green]Shape:[/bold green] {value.shape}")
        console.print(f"[bold green]DType:[/bold green] {value.dtype}")
        console.print(f"[bold green]Valeurs (premiers éléments) :[/bold green]\n{value[:5]}")
    elif isinstance(value, dict):
        console.print(f"[bold green]Type:[/bold green] dict")
        console.print(f"[bold green]Clés disponibles:[/bold green] {list(value.keys())}")
    else:
        console.print(f"[bold green]Type:[/bold green] {type(value)}")
        console.print(f"[bold green]Valeur :[/bold green] {value}")


def display_mat_table(mat_data, mat_file):
    """Affiche la liste des variables du fichier .mat sous forme de tableau."""
    console.print(Panel.fit(f"[bold cyan]📂 Contenu du fichier : {mat_file}[/bold cyan]\n", title="🔍 Analyse"))

    table = Table(title="📜 Variables dans le fichier .mat", highlight=True)
    table.add_column("Nom", style="bold yellow")
    table.add_column("Type", style="bold cyan")
    table.add_column("Shape", style="bold green")
    table.add_column("DType", style="bold magenta")

    for key, value in mat_data.items():
        if isinstance(value, np.ndarray):
            table.add_row(key, "np.ndarray", str(value.shape), str(value.dtype))
        else:
            table.add_row(key, str(type(value)), "-", "-")

    console.print(table)


def plot_data(mat_data):
    """Permet de visualiser une relation entre deux variables avec Matplotlib."""
    console.print("\n📊 [bold cyan]Visualisation des données avec Matplotlib[/bold cyan]")

    x_var = console.input("[bold yellow]Choisissez la variable pour l'axe X : [/bold yellow]").strip()
    y_var = console.input("[bold yellow]Choisissez la variable pour l'axe Y : [/bold yellow]").strip()

    if x_var not in mat_data or y_var not in mat_data:
        console.print(f"[bold red]❌ Erreur : L'une des variables '{x_var}' ou '{y_var}' n'existe pas.[/bold red]")
        return

    x_data = mat_data[x_var]
    y_data = mat_data[y_var]

    if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
        console.print(f"[bold red]❌ Erreur : Les deux variables doivent être des `np.ndarray`.[/bold red]")
        return

    # Sélectionner la dimension à afficher si nécessaire
    if x_data.ndim > 1:
        console.print(f"[bold cyan]Variable X '{x_var}' : Shape {x_data.shape}[/bold cyan]")
        dim_x = int(console.input(
            f"[bold green]Sélectionnez la dimension à afficher pour '{x_var}' (0-{x_data.shape[0] - 1}) : [/bold green]"))
        x_data = x_data[dim_x, :]

    if y_data.ndim > 1:
        console.print(f"[bold cyan]Variable Y '{y_var}' : Shape {y_data.shape}[/bold cyan]")
        dim_y = int(console.input(
            f"[bold green]Sélectionnez la dimension à afficher pour '{y_var}' (0-{y_data.shape[0] - 1}) : [/bold green]"))
        y_data = y_data[dim_y, :]

    if x_data.shape[0] != y_data.shape[0]:
        console.print(
            f"[bold red]❌ Erreur : Les dimensions de {x_var} et {y_var} ne correspondent pas après sélection ({x_data.shape[0]} vs {y_data.shape[0]}).[/bold red]")
        return

    title = console.input(
        "[bold green]Titre du graphique (par défaut: 'Visualisation') : [/bold green]").strip() or "Visualisation"
    xlabel = console.input(f"[bold green]Label de l'axe X (par défaut: {x_var}) : [/bold green]").strip() or x_var
    ylabel = console.input(f"[bold green]Label de l'axe Y (par défaut: {y_var}) : [/bold green]").strip() or y_var

    try:
        xlim = list(map(float, console.input("[bold green]xlim (min max, facultatif) : [/bold green]").split()))
    except ValueError:
        xlim = None

    try:
        ylim = list(map(float, console.input("[bold green]ylim (min max, facultatif) : [/bold green]").split()))
    except ValueError:
        ylim = None

    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, label=f"{y_var} vs {x_var}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if xlim and len(xlim) == 2:
        plt.xlim(xlim)
    if ylim and len(ylim) == 2:
        plt.ylim(ylim)

    plt.grid(True)
    plt.show()


def interactive_shell(mat_data, mat_file):
    """Boucle interactive permettant d'afficher et de visualiser les variables du fichier .mat."""
    session = PromptSession()
    commands = ["ls", "exit", "plot"] + list(mat_data.keys())
    completer = WordCompleter(commands, ignore_case=True)

    console.print(
        "\n💻 [bold cyan]Mode interactif activé ![/bold cyan] Tape 'ls' pour voir les variables, 'show <nom>' pour afficher une variable, 'plot' pour visualiser des données ou 'exit' pour quitter.")

    while True:
        try:
            command = session.prompt("🔹 Commande > ", completer=completer).strip()

            if command == "ls":
                display_mat_table(mat_data, mat_file)
            elif command.startswith("show "):
                var_name = command.split(" ", 1)[1]
                show_variable(mat_data, var_name)
            elif command == "plot":
                plot_data(mat_data)
            elif command == "exit":
                console.print("[bold yellow]👋 Fin de l'exploration du fichier .mat.[/bold yellow]")
                break
            else:
                console.print(
                    "[bold red]❌ Commande inconnue. Utilisez 'ls', 'show <nom_variable>', 'plot' ou 'exit'.[/bold red]")

        except KeyboardInterrupt:
            console.print("\n[bold yellow]👋 Fin de l'exploration du fichier .mat.[/bold yellow]")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Affiche et explore le contenu d'un fichier .mat de manière interactive.")
    parser.add_argument("--file", type=str, help="Fichier .mat à inspecter")

    args = parser.parse_args()
    mat_data = load_mat_file(args.file)

    if mat_data:
        display_mat_table(mat_data, args.file)
        interactive_shell(mat_data, args.file)