import marimo

__generated_with = "0.13.11"
app = marimo.App()


@app.cell
def _():
    import etl_tools as etl
    import pandas as pd
    import os
    return etl, os


@app.cell
def _(etl):
    tickets = etl.get_sp500_tickets()
    return (tickets,)


@app.cell
def _(etl, tickets):
    tickets_1 = etl.object_to_string(tickets)
    return (tickets_1,)


@app.cell
def _(tickets_1):
    tickets_1.info()
    return


app._unparsable_cell(
    r"""
    !pwd
    """,
    name="_"
)


@app.cell
def _(os):
    output_dir = "/home/sistemas/m/octavo/data/sp500"
    os.makedirs(output_dir, exist_ok=True)
    return


if __name__ == "__main__":
    app.run()

