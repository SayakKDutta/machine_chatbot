import sqlite3
import os
import plotly.graph_objs as go
from utils import convert_to_json, json_to_markdown_table

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "Fetch data from SQLite database",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "Complete and correct SQL query to fulfil user request.",
                    }
                },
                "required": ["sql_query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_chart",
            "description": "Plot Bar or Linechart to visualize the result of sql query",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "description": "Plot type either bar or line or scatter",
                    },
                    "x_values": {
                        "type": "array",
                        "description": "List of x values for plotting",
                        "items": {
                            "type": "string"
                        }
                    },
                    "y_values": {
                        "type": "array",
                        "description": "List of y axis values for plotting",
                        "items": {
                            "type": "number"
                        }
                    },
                    "plot_title": {
                        "type": "string",
                        "description": "Descriptive title for the plot",
                    },
                    "x_label": {
                        "type": "string",
                        "description": "Label for the x-axis",
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Label for the y-axis",
                    }
                },
                "required": ["plot_type","x_values","y_values","plot_title","x_label","y_label"],
            },
        }
    }
]


async def run_sqlite_query(sql_query, markdown=True):
    connection = None
    try:
        db_path = os.path.join(os.path.dirname(__file__), '../data/ai4i2020.db')
        connection = sqlite3.connect(db_path)

        cursor = connection.cursor()
        cursor.execute(sql_query)

        column_names = [desc[0] for desc in cursor.description]
        result = cursor.fetchall()

        if markdown:
            json_data = convert_to_json(result, column_names)
            markdown_data = json_to_markdown_table(json_data)
            return markdown_data

        return result, column_names
    except sqlite3.Error as error:
        print("Error while executing the query:", error)
        if markdown:
            return f"Error while executing the query: {error}"
        return [], []

    finally:
        if connection:
            cursor.close()
            connection.close()
            print("SQLite connection is closed")

async def plot_chart(x_values, y_values, plot_title, x_label, y_label, plot_type='line', save_path="tmp/tmp.png"):
    if len(x_values) != len(y_values):
        raise ValueError("Lengths of x_values and y_values must be the same.")

    if plot_type == 'bar':
        trace = go.Bar(x=x_values, y=y_values, marker=dict(color='#24C8BF', line=dict(width=1)))
    elif plot_type == 'scatter':
        trace = go.Scatter(x=x_values, y=y_values, mode='markers', marker=dict(color='#df84ff', size=10, opacity=0.7, line=dict(width=1)))
    elif plot_type == 'line':
        trace = go.Scatter(x=x_values, y=y_values, mode='lines+markers', marker=dict(color='#ff9900', size=8, line=dict(width=1)), line=dict(width=2, color='#ff9900'))

    layout = go.Layout(
        title=f'{plot_title} {plot_type.capitalize()} Chart',
        title_font=dict(size=20, family='Arial', color='#333'),
        xaxis=dict(title=x_label, titlefont=dict(size=18), tickfont=dict(size=14), gridcolor='#f0f0f0'),
        yaxis=dict(title=y_label, titlefont=dict(size=18), tickfont=dict(size=14), gridcolor='#f0f0f0'),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor='#f8f8f8',
        paper_bgcolor='#f8f8f8'
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig
