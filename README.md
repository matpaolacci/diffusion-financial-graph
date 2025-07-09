# Synthetic Generation of Financial Transaction Graphs Using Diffusion Models for Anti-Money Laundering

First off, install Miniconda, move to the project directory, and then run the following command:

```bash
conda env create --file environments/notebook_env.yml
```
Then open the [Notebook](notebook.ipynb) and run it selecting the jupyter kernel in the environement _notebook-env_.

The notebook was developed using Visual Studio Code, so some plots may not render correctly in other environments (e.g., Google Colab) due to differences in how interactive output is handled.

To create the environment to run the DiGress model run the following command:
```bash
conda env create --file environments/digress_env.yml
```

To start the training, run the following commands from the root of the project:
```bash
python src/main.py dataset=financial
```

If you are using vscode and if you want to run in debug mode create the launch file at `.vscode/launch.json` and add the following json content:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Digress",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/main.py",
      "args": ["dataset=financial"],
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  ]
}
```