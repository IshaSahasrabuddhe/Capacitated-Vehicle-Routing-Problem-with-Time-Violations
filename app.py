import json
import gradio as gr
import pandas as pd
from solver import (
    generate_random_instance,
    solve_vrp_tw,
    plot_solution,
    parse_uploaded_csv,
    make_template_dataframe,
)

TITLE = "Ride-Sharing Optimizer (Capacitated VRP) — Gradio Demo"
DESC = """
This demo assigns **stops (riders)** to **drivers (vehicles)** with a simple, fast heuristic:
**Sweep clustering** → **Greedy routing** → **2-opt improvement**.  
You can **generate a sample** dataset or **upload a CSV** with columns:
`id,x,y,demand,tw_start,tw_end,service`.
- **Capacity** = max riders per vehicle (sum of `demand` per route).
- **Time windows** are *soft* (violations are reported in metrics).
- Distances are Euclidean on the X-Y plane for clarity.
💡 Tip: Start with the generator, then switch to your CSV.
"""

FOOTER = "Made with ❤️ using Gradio. No native dependencies; runs quickly on Spaces."


# -----------------------------
# Helper function: Format Metrics
# -----------------------------


def format_metrics(metrics_dict):
    # Build per-route table
    routes = range(1, len(metrics_dict["per_route_distance"]) + 1)
    route_rows = [
        f"| Route {i} | {metrics_dict['per_route_distance'][i-1]:.2f} | {metrics_dict['per_route_load'][i-1]} |"
        for i in routes
    ]
    route_table = "\n".join(route_rows)

    metrics = f"""
### 🚗 Vehicle Routing Summary
**Vehicles used:** {metrics_dict['vehicles_used']}  
**Vehicle capacity:** {metrics_dict['capacity']}  
**Total distance:** {metrics_dict['total_distance']:.2f} units  

---

**📊 Per-route Performance**

| Route | Distance | Load |
|:------|----------:|------:|
{route_table}

---

**⏰ Time Window Report:**  
- Total lateness count: {metrics_dict['time_window_report']['total_lateness_count']}  
- Total lateness: {metrics_dict['time_window_report']['total_lateness']:.2f}  
- Max lateness: {metrics_dict['time_window_report']['max_lateness']:.2f}  
- Status: ✅ {metrics_dict['time_window_report']['status']}

---

**🧩 Notes:**  
{metrics_dict['note']}
"""
    return metrics


# -----------------------------
# Functions
# -----------------------------
def run_generator(n_clients, n_vehicles, capacity, spread, demand_min, demand_max, seed):
    df = generate_random_instance(
        n_clients=n_clients,
        n_vehicles=n_vehicles,
        capacity=capacity,
        spread=spread,
        demand_min=demand_min,
        demand_max=demand_max,
        seed=seed,
    )
    depot = (0.0, 0.0)
    sol = solve_vrp_tw(df, depot=depot, n_vehicles=n_vehicles, capacity=capacity, speed=10.0)

    # ✅ plot_solution returns a PIL image
    img = plot_solution(df, sol, depot=depot)

    route_table = sol["assignments_table"]
    metrics = format_metrics(sol["metrics"])  # 🪄 Prettify metrics

    return img, route_table, metrics, df


def run_csv(file, n_vehicles, capacity):
    if file is None:
        raise gr.Error("Please upload a CSV first.")
    try:
        df = parse_uploaded_csv(file)
    except Exception as e:
        raise gr.Error(f"CSV parsing error: {e}")

    depot = (0.0, 0.0)
    sol = solve_vrp_tw(df, depot=depot, n_vehicles=n_vehicles, capacity=capacity, speed=10.0)
    img = plot_solution(df, sol, depot=depot)

    route_table = sol["assignments_table"]
    metrics = format_metrics(sol["metrics"])  # 🪄 Prettify metrics

    return img, route_table, metrics


def download_template():
    return make_template_dataframe()


# -----------------------------
# UI Layout
# -----------------------------
with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESC)

    with gr.Tab("🔀 Generate sample"):
        with gr.Row():
            with gr.Column():
                n_clients = gr.Slider(5, 200, value=30, step=1, label="Number of riders (clients)")
                n_vehicles = gr.Slider(1, 20, value=4, step=1, label="Number of drivers (vehicles)")
                capacity = gr.Slider(1, 50, value=10, step=1, label="Vehicle capacity (sum of demand)")
                spread = gr.Slider(10, 200, value=50, step=1, label="Spatial spread (larger = wider map)")
                demand_min = gr.Slider(1, 5, value=1, step=1, label="Min demand per stop")
                demand_max = gr.Slider(1, 10, value=3, step=1, label="Max demand per stop")
                seed = gr.Slider(0, 9999, value=42, step=1, label="Random seed")
                run_btn = gr.Button("🚗 Generate & Optimize", variant="primary")
            with gr.Column():
                img = gr.Image(type="pil", label="Route Visualization", interactive=False)

        with gr.Row():
            route_df = gr.Dataframe(label="Route assignments (per stop)", wrap=True)
            metrics = gr.Markdown(label="Metrics Summary")  # 🧩 Changed from Code → Markdown
        with gr.Accordion("Show generated dataset", open=False):
            data_out = gr.Dataframe(label="Generated input data")

        run_btn.click(
            fn=run_generator,
            inputs=[n_clients, n_vehicles, capacity, spread, demand_min, demand_max, seed],
            outputs=[img, route_df, metrics, data_out],
        )

    with gr.Tab("📄 Upload CSV"):
        with gr.Row():
            with gr.Column():
                file = gr.File(label="Upload CSV (id,x,y,demand,tw_start,tw_end,service)")
                dl_tmp = gr.Button("Get CSV Template")
                n_vehicles2 = gr.Slider(1, 50, value=5, step=1, label="Number of drivers (vehicles)")
                capacity2 = gr.Slider(1, 200, value=15, step=1, label="Vehicle capacity (sum of demand)")
                run_btn2 = gr.Button("📈 Optimize uploaded data", variant="primary")
            with gr.Column():
                img2 = gr.Image(type="pil", label="Route Visualization", interactive=False)

        with gr.Row():
            route_df2 = gr.Dataframe(label="Route assignments (per stop)")
            metrics2 = gr.Markdown(label="Metrics Summary")  # 🧩 Changed from Code → Markdown

        run_btn2.click(
            fn=run_csv,
            inputs=[file, n_vehicles2, capacity2],
            outputs=[img2, route_df2, metrics2],
        )

        def _tmpl():
            return gr.File.update(value=None), download_template()

        dl_tmp.click(fn=_tmpl, outputs=[file, route_df2], inputs=None)

    gr.Markdown(f"---\n{FOOTER}")

if __name__ == "__main__":
    demo.launch()