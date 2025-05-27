import marimo as mo

# Inicializa la app
app = mo.App()

# Crea un componente interactivo
slider = mo.ui.slider(1, 10, value=5)

# Asigna el componente a la app (usa @= en lugar de @)
app @= mo.md(f"**Valor del slider:** {slider}")

if __name__ == "__main__":
    app.run()